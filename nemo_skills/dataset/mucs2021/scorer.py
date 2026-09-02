# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Protocol-faithful MUCS 2021 Subtask 2 blind-test scorer.

This is a direct, dependency-free implementation of the released
``inference/compute_wer.py`` protocol. Both references and hypotheses are
lowercased and whitespace-tokenized. The punctuation token map is applied to
hypotheses only for Normal WER. The language word map is then applied to those
already punctuation-mapped hypotheses only for Transliterated WER. Empty
references are skipped exactly as in the release.
"""

from __future__ import annotations

import hashlib
from functools import cache
from pathlib import Path
from typing import Any

from nemo_skills.evaluation.evaluator.base import BaseEvaluator, BaseEvaluatorConfig
from nemo_skills.evaluation.metrics.base import BaseMetrics, as_int, as_percentage
from nemo_skills.utils import nested_dataclass

ASSETS_DIR = Path(__file__).parent / "assets"
ASSET_CONTRACTS = {
    "punct.mapping.txt": ("43927a9aea99289d552bdc509406680688f67e25d52fb6f9debcd27c1e31b1ef", 76),
    "hin2engwords.txt": ("af49f1a95114021e678a0e86c7a9c4e85c8b91072ebd538d76293e73b8b5968e", 1315),
    "ben2engwords.txt": ("8e53663a8b839e042eb9e430877fc9f783efb056d169de2e4c8d7b1fe28e5b80", 1177),
    "official_compute_wer.py": ("fc22eefa193f23dfee0b23d4078233cc9e3cb2c95690ccf48f39c5dfe8d388b6", 99),
}
WORD_MAPPING_ASSET = {
    "hindi-english": "hin2engwords.txt",
    "bengali-english": "ben2engwords.txt",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_mapping(path: Path) -> dict[str, str]:
    """Read a released mapping with the official first-key-wins behavior."""
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            fields = raw_line.strip().split()
            if not fields:
                raise ValueError(f"Blank mapping row at {path}:{line_number}")
            mapping.setdefault(fields[0], " ".join(fields[1:]))
    return mapping


@cache
def load_official_mapping(asset_name: str) -> dict[str, str]:
    """Load one vendored official map after checksum and row validation."""
    if asset_name not in ASSET_CONTRACTS:
        raise ValueError(f"Unknown MUCS scorer asset: {asset_name}")
    expected_sha, expected_rows = ASSET_CONTRACTS[asset_name]
    path = ASSETS_DIR / asset_name
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise ValueError(f"MUCS scorer asset drift for {asset_name}: {actual_sha} != {expected_sha}")
    actual_rows = sum(1 for _ in path.open(encoding="utf-8"))
    if actual_rows != expected_rows:
        raise ValueError(f"MUCS scorer row drift for {asset_name}: {actual_rows} != {expected_rows}")
    return _read_mapping(path)


def whitespace_tokens(text: object) -> list[str]:
    """Match ``line.strip().lower().split()`` from the official script."""
    return str(text).strip().lower().split()


def apply_mapping(tokens: list[str], mapping: dict[str, str]) -> list[str]:
    """Match the official join-after-token-replacement operation exactly."""
    return " ".join(mapping.get(token, token) for token in tokens).split()


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    """Levenshtein distance equivalent to the jiwer version in the release."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_token in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_token in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + (ref_token != hyp_token),
                )
            )
        previous = current
    return previous[-1]


def evaluate_mucs2021_sample(data_point: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one row while retaining additive corpus counts."""
    language = str(data_point.get("extra_fields", {}).get("mucs_language", ""))
    if language not in WORD_MAPPING_ASSET:
        raise ValueError(f"Missing or invalid extra_fields.mucs_language: {language!r}")

    reference = whitespace_tokens(data_point["expected_answer"])
    generation = str(data_point.get("generation") or "").strip()
    hypothesis = whitespace_tokens(generation)
    punctuated_hypothesis = apply_mapping(hypothesis, load_official_mapping("punct.mapping.txt"))
    transliterated_hypothesis = apply_mapping(
        punctuated_hypothesis,
        load_official_mapping(WORD_MAPPING_ASSET[language]),
    )

    # The official loop continues before calling jiwer when the reference is
    # empty. Hypothesis insertions therefore do not contribute to either WER.
    if not reference:
        normal_errors = 0
        transliterated_errors = 0
        skipped = True
    else:
        normal_errors = edit_distance(reference, punctuated_hypothesis)
        transliterated_errors = edit_distance(reference, transliterated_hypothesis)
        skipped = False

    ref_words = len(reference)
    return {
        "generation": generation,
        "text": " ".join(reference),
        "pred_text": " ".join(punctuated_hypothesis),
        "transliterated_pred_text": " ".join(transliterated_hypothesis),
        "wer": normal_errors / ref_words if ref_words else None,
        "twer": transliterated_errors / ref_words if ref_words else None,
        "wer_errors": normal_errors,
        "twer_errors": transliterated_errors,
        "ref_words": ref_words,
        "skipped_empty_reference": skipped,
    }


@nested_dataclass(kw_only=True)
class MUCS2021EvaluatorConfig(BaseEvaluatorConfig):
    pass


class MUCS2021Evaluator(BaseEvaluator):
    async def eval_single(self, data_point: dict[str, Any]) -> dict[str, Any]:
        return evaluate_mucs2021_sample(data_point)


class MUCS2021Metrics(BaseMetrics):
    """Corpus WER/tWER matching the official per-utterance weighted sum."""

    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        super().__init__(compute_no_answer=compute_no_answer)
        if max_k != 1:
            raise ValueError("MUCS2021Metrics supports one greedy generation only")
        self.max_k = 1

    def reset(self):
        super().reset()
        self.wer_errors = 0
        self.twer_errors = 0
        self.ref_words = 0
        self.scored_entries = 0
        self.skipped_empty_references = 0
        self.no_answer = 0

    def _get_score_dict(self, prediction):
        return {"exact": prediction.get("wer_errors") == 0 and not prediction.get("skipped_empty_reference")}

    def get_incorrect_sample(self, prediction):
        prediction = prediction.copy()
        ref_words = int(prediction.get("ref_words", 0))
        prediction.update(
            wer_errors=max(1, ref_words),
            twer_errors=max(1, ref_words),
            skipped_empty_reference=False,
        )
        return prediction

    def update(self, predictions):
        if len(predictions) != 1:
            raise ValueError("MUCS2021Metrics expects exactly one prediction per sample")
        super().update(predictions)
        prediction = predictions[0]
        self.wer_errors += int(prediction["wer_errors"])
        self.twer_errors += int(prediction["twer_errors"])
        self.ref_words += int(prediction["ref_words"])
        if prediction["skipped_empty_reference"]:
            self.skipped_empty_references += 1
        else:
            self.scored_entries += 1
        if not (prediction.get("generation") or "").strip():
            self.no_answer += 1

    def get_metrics(self):
        if self.ref_words <= 0:
            raise ValueError("MUCS 2021 has no non-empty reference words to score")
        values = {
            "wer": round(100.0 * self.wer_errors / self.ref_words, 2),
            "twer": round(100.0 * self.twer_errors / self.ref_words, 2),
            "wer_errors": self.wer_errors,
            "twer_errors": self.twer_errors,
            "ref_words": self.ref_words,
            "scored_entries": self.scored_entries,
            "skipped_empty_references": self.skipped_empty_references,
            "no_answer": 100.0 * self.no_answer / self.total,
        }
        self.update_common_metrics(values)
        return {"pass@1": values}

    def evaluations_to_print(self):
        return ["pass@1"]

    def metrics_to_print(self):
        return {
            "wer": as_percentage,
            "twer": as_percentage,
            "no_answer": as_percentage,
            "wer_errors": as_int,
            "twer_errors": as_int,
            "ref_words": as_int,
            "scored_entries": as_int,
            "skipped_empty_references": as_int,
            "num_entries": as_int,
        }
