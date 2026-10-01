# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Versioned WER and entity evaluation for VoiceCodeBench.

The deterministic lattice-WER implementation below is a direct, dependency-
free transcription of the MIT-licensed official scorer at revision
``af309592118b83fdf9b0ad896b564a142657aac8``. CTEM and TSR can use the offline
``deterministic_annotations_v1`` verifier, or the pinned release GPT-5.5
verifier in the default live-fill mode. NVIDIA Inference Hub uses the same
prompt and schema with a distinct endpoint, model alias, ID, and digest.
The verifier ID and digest are carried into every metric artifact because
values from these verifiers are not interchangeable.
"""

import asyncio
import importlib.util
import json
import os
import re
import sys
import typing
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any

from nemo_skills.evaluation.evaluator.base import BaseEvaluator, BaseEvaluatorConfig
from nemo_skills.evaluation.evaluator.voice_code_bench_deterministic import (
    DETERMINISTIC_VERIFIER_DIGEST,
    DETERMINISTIC_VERIFIER_ID,
    DETERMINISTIC_VERIFIER_MODEL,
    verify_deterministic_entity_matches,
)
from nemo_skills.utils import nested_dataclass

OFFICIAL_REVISION = "af309592118b83fdf9b0ad896b564a142657aac8"
FORMAT_INVARIANT_WER_REFERENCE = "acoustic_or_canonical_entity_lattice"
PINNED_VERIFIER_ID = "openai_gpt_5_5_v1"
NVIDIA_HUB_VERIFIER_ID = "nvidia_inference_hub_gpt_5_5_v1"
NVIDIA_HUB_VERIFIER_MODEL = "openai/openai/gpt-5.5"
NVIDIA_HUB_VERIFIER_ENDPOINT = "https://inference-api.nvidia.com/v1/chat/completions"
WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
TEMPLATE_ENTITY_RE = re.compile(r"<([^>]+)>")


@dataclass(frozen=True)
class WerResult:
    reference_words: int
    hypothesis_words: int
    substitutions: int
    deletions: int
    insertions: int

    @property
    def wer(self) -> float:
        if self.reference_words == 0:
            return 0.0
        return (self.substitutions + self.deletions + self.insertions) / self.reference_words


def tokenize_words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def _find_subsequence(tokens: list[str], subsequence: list[str], start: int = 0) -> int | None:
    width = len(subsequence)
    for index in range(start, len(tokens) - width + 1):
        if tokens[index : index + width] == subsequence:
            return index
    return None


def _reference_chunks(datapoint: dict[str, Any], reference: list[str]) -> list[list[list[str]]]:
    entities = datapoint.get("entities")
    transcripts = datapoint.get("transcripts")
    if not isinstance(entities, list):
        raise TypeError("VoiceCodeBench datapoint has invalid entities")
    if not isinstance(transcripts, dict) or not isinstance(transcripts.get("template"), str):
        raise TypeError("VoiceCodeBench datapoint is missing its template transcript")

    entities_by_id = {entity["id"]: entity for entity in entities}
    entity_ids = TEMPLATE_ENTITY_RE.findall(transcripts["template"])
    if len(entity_ids) != len(entities) or set(entity_ids) != set(entities_by_id):
        entity_ids = list(entities_by_id)

    chunks: list[list[list[str]]] = []
    cursor = 0
    for entity_id in entity_ids:
        entity = entities_by_id[entity_id]
        acoustic_tokens = tokenize_words(entity["acoustic"])
        canonical_tokens = tokenize_words(entity["canonical"])
        start = _find_subsequence(reference, acoustic_tokens, cursor)
        if start is None:
            raise ValueError(
                f"Acoustic transcript does not contain entity {entity_id} after token {cursor}"
            )
        if start > cursor:
            chunks.append([reference[cursor:start]])
        alternatives = [acoustic_tokens]
        if canonical_tokens != acoustic_tokens:
            alternatives.append(canonical_tokens)
        chunks.append(alternatives)
        cursor = start + len(acoustic_tokens)
    if cursor < len(reference):
        chunks.append([reference[cursor:]])
    return chunks


def _advance_state(
    initial_cost: list[int],
    initial_sub: list[int],
    initial_del: list[int],
    initial_ins: list[int],
    reference: list[str],
    hypothesis: list[str],
) -> tuple[list[int], list[int], list[int], list[int]]:
    cols = len(hypothesis) + 1
    prev_cost = list(initial_cost)
    prev_sub = list(initial_sub)
    prev_del = list(initial_del)
    prev_ins = list(initial_ins)
    for reference_word in reference:
        curr_cost = [prev_cost[0] + 1] + [0] * (cols - 1)
        curr_sub = [0] * cols
        curr_del = [prev_del[0] + 1] + [0] * (cols - 1)
        curr_ins = [0] * cols
        for col, hypothesis_word in enumerate(hypothesis, start=1):
            increment = int(reference_word != hypothesis_word)
            best = (
                prev_cost[col - 1] + increment,
                prev_sub[col - 1] + increment,
                prev_del[col - 1],
                prev_ins[col - 1],
            )
            deletion = (prev_cost[col] + 1, prev_sub[col], prev_del[col] + 1, prev_ins[col])
            if deletion[0] < best[0]:
                best = deletion
            insertion = (
                curr_cost[col - 1] + 1,
                curr_sub[col - 1],
                curr_del[col - 1],
                curr_ins[col - 1] + 1,
            )
            if insertion[0] < best[0]:
                best = insertion
            curr_cost[col], curr_sub[col], curr_del[col], curr_ins[col] = best
        prev_cost, prev_sub, prev_del, prev_ins = curr_cost, curr_sub, curr_del, curr_ins
    return prev_cost, prev_sub, prev_del, prev_ins


def compute_format_invariant_wer(datapoint: dict[str, Any], hypothesis_text: str) -> WerResult:
    reference = tokenize_words(datapoint["transcripts"]["acoustic"])
    hypothesis = tokenize_words(hypothesis_text)
    chunks = _reference_chunks(datapoint, reference)
    cols = len(hypothesis) + 1
    prev_cost = list(range(cols))
    prev_sub = [0] * cols
    prev_del = [0] * cols
    prev_ins = list(range(cols))
    for alternatives in chunks:
        candidates = [
            _advance_state(prev_cost, prev_sub, prev_del, prev_ins, alternative, hypothesis)
            for alternative in alternatives
        ]
        prev_cost, prev_sub, prev_del, prev_ins = map(list, candidates[0])
        for cost, substitutions, deletions, insertions in candidates[1:]:
            for col in range(cols):
                if cost[col] < prev_cost[col]:
                    prev_cost[col] = cost[col]
                    prev_sub[col] = substitutions[col]
                    prev_del[col] = deletions[col]
                    prev_ins[col] = insertions[col]
    return WerResult(
        reference_words=len(reference),
        hypothesis_words=len(hypothesis),
        substitutions=prev_sub[-1],
        deletions=prev_del[-1],
        insertions=prev_ins[-1],
    )


def _official_root(data_dir: str) -> Path:
    root = Path(data_dir) / "voice-code-bench"
    revision_path = root / "NEMO_SKILLS_REVISION"
    revision = revision_path.read_text(encoding="utf-8").strip()
    if revision != OFFICIAL_REVISION:
        raise ValueError(
            f"VoiceCodeBench scorer revision mismatch: expected {OFFICIAL_REVISION}, found {revision}"
        )
    return root


def _ensure_typing_not_required() -> None:
    """Make the pinned Python 3.11 scorer importable in Python 3.10 images."""
    if not hasattr(typing, "NotRequired"):
        from typing_extensions import NotRequired

        setattr(typing, "NotRequired", NotRequired)


def _load_official_package(data_dir: str) -> tuple[ModuleType, ModuleType, str]:
    root = _official_root(data_dir)
    _ensure_typing_not_required()
    package_dir = root / "scripts" / "voice_code_bench"
    init_path = package_dir / "__init__.py"
    package_name = f"_nemo_skills_voice_code_bench_{OFFICIAL_REVISION[:12]}"
    if package_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package_name, init_path, submodule_search_locations=[str(package_dir)]
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load official VoiceCodeBench package from {package_dir}")
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        spec.loader.exec_module(package)
    metrics = __import__(f"{package_name}.metrics", fromlist=["metrics"])
    verifier = __import__(f"{package_name}.verifier", fromlist=["verifier"])
    return metrics, verifier, str(root)


@nested_dataclass(kw_only=True)
class VoiceCodeBenchEvaluatorConfig(BaseEvaluatorConfig):
    entity_verifier_mode: str = "live-fill"
    verifier_id: str = PINNED_VERIFIER_ID
    verifier_provider: str = "nvidia-inference-hub"
    verifier_cache_path: str | None = None
    verifier_concurrency: int = 10


class VoiceCodeBenchEvaluator(BaseEvaluator):
    """Batch evaluator using the pinned official entity verifier and scorer."""

    def __init__(self, config: dict[str, Any], num_parallel_requests: int = 10):
        super().__init__(config, num_parallel_requests)
        self.eval_config = VoiceCodeBenchEvaluatorConfig(**config)
        modes = {"off", "deterministic", "live", "replay", "live-fill"}
        if self.eval_config.entity_verifier_mode not in modes:
            raise ValueError(
                "entity_verifier_mode must be off, deterministic, live, replay, or live-fill"
            )
        if self.eval_config.verifier_provider not in {"openai", "nvidia-inference-hub"}:
            raise ValueError("verifier_provider must be openai or nvidia-inference-hub")

    async def eval_full(self) -> None:
        if not self.eval_config.input_file or not self.eval_config.data_dir:
            raise ValueError("VoiceCodeBench evaluation requires input_file and data_dir")
        official_root = str(_official_root(self.eval_config.data_dir))
        official_verifier: ModuleType | None = None
        verifier_config = None
        verifier_provider = self.eval_config.verifier_provider
        api_key_name = (
            "NVIDIA_INFERENCE_HUB_API_KEY" if verifier_provider == "nvidia-inference-hub" else "OPENAI_API_KEY"
        )
        api_key = os.environ.get(api_key_name, "").strip()
        if self.eval_config.entity_verifier_mode in {"live", "replay", "live-fill"}:
            _, official_verifier, official_root = _load_official_package(self.eval_config.data_dir)
            verifier_config = official_verifier.load_verifier_config(self.eval_config.verifier_id)
            if verifier_provider == "nvidia-inference-hub":
                if verifier_config.id != PINNED_VERIFIER_ID or verifier_config.model != "gpt-5.5":
                    raise ValueError("NVIDIA Inference Hub requires the pinned GPT-5.5 verifier config")
                verifier_config = replace(
                    verifier_config,
                    id=NVIDIA_HUB_VERIFIER_ID,
                    model=NVIDIA_HUB_VERIFIER_MODEL,
                    endpoint_or_api=NVIDIA_HUB_VERIFIER_ENDPOINT,
                )
        input_path = Path(self.eval_config.input_file)
        cache_path = Path(
            self.eval_config.verifier_cache_path
            or f"{self.eval_config.input_file}.voice-code-verifier-cache.json"
        )
        cache = None
        if official_verifier is not None and self.eval_config.entity_verifier_mode in {"replay", "live-fill"}:
            cache = official_verifier.VerifierCache(cache_path)
        if self.eval_config.entity_verifier_mode in {"live", "live-fill"} and not api_key:
            raise RuntimeError(
                f"VoiceCodeBench GPT-5.5 CTEM/TSR requires {api_key_name} for {verifier_provider}. "
                "Use entity_verifier_mode=replay with a complete cache, "
                "or explicitly use off for WER-only diagnostics."
            )

        rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        semaphore = asyncio.Semaphore(self.eval_config.verifier_concurrency)

        async def evaluate_row(row: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                return await asyncio.to_thread(
                    self._evaluate_row,
                    row,
                    official_verifier,
                    verifier_config,
                    cache,
                    api_key,
                    official_root,
                )

        evaluated = await asyncio.gather(*(evaluate_row(row) for row in rows))
        temp_path = input_path.with_name(input_path.name + "-tmp")
        with temp_path.open("w", encoding="utf-8") as output:
            for row in evaluated:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temp_path, input_path)

    def _evaluate_row(
        self,
        row: dict[str, Any],
        official_verifier: ModuleType | None,
        verifier_config: Any,
        cache: Any,
        api_key: str,
        official_root: str,
    ) -> dict[str, Any]:
        if row.get("voice_code_bench_revision") != OFFICIAL_REVISION:
            raise ValueError(f"Unexpected VoiceCodeBench row revision for {row.get('audio_id')}")
        datapoint = row["voice_code_bench"]
        generation = str(row.get("generation") or "").strip()
        wer = compute_format_invariant_wer(datapoint, generation)
        updates: dict[str, Any] = {
            "voice_code_wer_reference": FORMAT_INVARIANT_WER_REFERENCE,
            "voice_code_wer_errors": wer.substitutions + wer.deletions + wer.insertions,
            "voice_code_wer_ref_words": wer.reference_words,
            "voice_code_wer_hyp_words": wer.hypothesis_words,
            "voice_code_wer_substitutions": wer.substitutions,
            "voice_code_wer_deletions": wer.deletions,
            "voice_code_wer_insertions": wer.insertions,
            "voice_code_official_root": official_root,
        }
        if self.eval_config.entity_verifier_mode == "off":
            updates["voice_code_entity_verifier_status"] = "off"
            updates["is_correct"] = wer.wer == 0.0
            return {**row, **updates}

        if self.eval_config.entity_verifier_mode == "deterministic":
            matches = verify_deterministic_entity_matches(datapoint, generation)
            verifier_id = DETERMINISTIC_VERIFIER_ID
            verifier_model = DETERMINISTIC_VERIFIER_MODEL
            verifier_digest = DETERMINISTIC_VERIFIER_DIGEST
        else:
            if official_verifier is None or verifier_config is None:
                raise RuntimeError("VoiceCodeBench entity verifier was not initialized")
            matches = official_verifier.verify_entity_matches(
                datapoint,
                generation,
                verifier_config,
                {"OPENAI_API_KEY": api_key} if api_key else {},
                cache=cache,
                cache_mode=self.eval_config.entity_verifier_mode,
            )
            verifier_id = verifier_config.id
            verifier_model = verifier_config.model
            verifier_digest = verifier_config.digest
        correct_by_type: dict[str, int] = {}
        total_by_type: dict[str, int] = {}
        for entity, match in zip(datapoint["entities"], matches, strict=True):
            entity_type = entity["type"]
            total_by_type[entity_type] = total_by_type.get(entity_type, 0) + 1
            correct_by_type[entity_type] = correct_by_type.get(entity_type, 0) + int(match["present"])
        correct = sum(correct_by_type.values())
        total = len(datapoint["entities"])
        updates.update(
            {
                "voice_code_entity_verifier_status": "complete",
                "voice_code_verifier_id": verifier_id,
                "voice_code_verifier_model": verifier_model,
                "voice_code_verifier_config_digest": verifier_digest,
                "voice_code_entity_matches": matches,
                "voice_code_correct_entities": correct,
                "voice_code_total_entities": total,
                "voice_code_task_success": correct == total,
                "voice_code_correct_by_type": correct_by_type,
                "voice_code_total_by_type": total_by_type,
                "is_correct": correct == total,
            }
        )
        return {**row, **updates}
