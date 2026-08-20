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

"""Mode-selectable evaluator for Rich Transcription ASR."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from nemo_skills.evaluation.evaluator.audio import (
    AudioEvaluator,
    AudioEvaluatorConfig,
    _wer_with_counts,
    coerce_text_field,
    extract_asr_text,
    strip_helpful_prefixes,
)
from nemo_skills.evaluation.evaluator.base import BaseEvaluator
from nemo_skills.evaluation.evaluator.open_asr_normalizer import (
    OpenASREnglishTextNormalizer,
)
from nemo_skills.utils import nested_dataclass

from .rich_transcription_asr_metrics import extract_tokens, score_event_tokens

ANGLE_TAG_RE = re.compile(r"(<[^>]*>)")
EVALUATION_MODES = ("event", "style", "event_style", "all")
EVALUATION_MODE_ALIASES = {
    "1": "event",
    "event": "event",
    "events": "event",
    "2": "style",
    "style": "style",
    "styles": "style",
    "3": "event_style",
    "event_style": "event_style",
    "event+style": "event_style",
    "all_special": "event_style",
    "4": "all",
    "all": "all",
    "all_tokens": "all",
}
SPECIAL_SCOPE_BY_MODE = {
    "event": "event",
    "style": "style",
    "event_style": "all_special",
}


@nested_dataclass(kw_only=True)
class RichTranscriptionASREvaluatorConfig(AudioEvaluatorConfig):
    """Evaluator configuration; event-only WER is the default contract."""

    evaluation_mode: str = "event"


def normalize_evaluation_mode(evaluation_mode: str | int) -> str:
    """Resolve numeric and descriptive mode aliases to one canonical name."""
    key = str(evaluation_mode).strip().lower()
    if key not in EVALUATION_MODE_ALIASES:
        raise ValueError(
            f"Unsupported evaluation mode {evaluation_mode!r}; expected one of {EVALUATION_MODES} or 1/2/3/4"
        )
    return EVALUATION_MODE_ALIASES[key]


@lru_cache(maxsize=1)
def _normalizer() -> OpenASREnglishTextNormalizer:
    return OpenASREnglishTextNormalizer()


def normalize_rich_transcription_text(text: str) -> str:
    """Normalize ordinary words while retaining every inline angle tag."""
    pieces = []
    normalizer = _normalizer()
    for part in ANGLE_TAG_RE.split(text or ""):
        if not part:
            continue
        if part.startswith("<") and part.endswith(">"):
            pieces.append(part)
        else:
            normalized = normalizer(part)
            if normalized:
                pieces.append(normalized)
    return " ".join(pieces).strip()


def _score_special_mode(reference: str, hypothesis: str, scope: str) -> dict[str, Any]:
    metrics = score_event_tokens(
        [reference], [hypothesis], token_type=scope, exclude_flat=False
    )
    normalized_reference = " ".join(
        extract_tokens(reference, scope, exclude_flat=False)
    )
    normalized_hypothesis = " ".join(
        extract_tokens(hypothesis, scope, exclude_flat=False)
    )
    return {
        "wer": metrics.wer,
        "wer_errors": metrics.errors,
        "wer_ref_words": metrics.ref_tokens,
        "wer_substitutions": metrics.substitutions,
        "wer_insertions": metrics.insertions,
        "wer_deletions": metrics.deletions,
        "text": normalized_reference,
        "pred_text": normalized_hypothesis,
    }


def evaluate_rich_transcription_asr_text(
    reference: str,
    hypothesis: str,
    evaluation_mode: str | int = "event",
) -> dict[str, Any]:
    """Compute WER after retaining only the tokens selected by the mode."""
    evaluation_mode = normalize_evaluation_mode(evaluation_mode)
    if evaluation_mode == "all":
        normalized_reference = normalize_rich_transcription_text(reference)
        normalized_hypothesis = normalize_rich_transcription_text(hypothesis)
        metrics = _wer_with_counts(normalized_reference, normalized_hypothesis)
        metrics["text"] = normalized_reference
        metrics["pred_text"] = normalized_hypothesis
    else:
        metrics = _score_special_mode(
            reference, hypothesis, SPECIAL_SCOPE_BY_MODE[evaluation_mode]
        )

    metrics["evaluation_mode"] = evaluation_mode
    metrics["is_correct"] = None if metrics["wer"] is None else metrics["wer"] < 0.5
    return metrics


def evaluate_rich_transcription_asr_sample(
    sample: dict[str, Any],
    evaluation_mode: str | int = "event",
) -> dict[str, Any]:
    """Return per-sample fields needed for selected-mode corpus WER."""
    reference = coerce_text_field(sample.get("expected_answer", sample.get("text", "")))
    hypothesis = coerce_text_field(sample.get("generation", ""))
    updates = evaluate_rich_transcription_asr_text(
        reference, hypothesis, evaluation_mode
    )
    updates["predicted_answer"] = hypothesis

    duration = sample.get("audio_duration", sample.get("duration"))
    if duration and float(duration) > 0 and reference and hypothesis:
        duration = float(duration)
        updates["ref_char_rate"] = (len(reference) / duration) * 60.0
        updates["hyp_char_rate"] = (len(hypothesis) / duration) * 60.0
        updates["char_rate_diff"] = abs(
            updates["hyp_char_rate"] - updates["ref_char_rate"]
        )
    return updates


class RichTranscriptionASREvaluator(AudioEvaluator):
    """Evaluate one of four Rich Transcription ASR token scopes."""

    def __init__(self, config: dict, num_parallel_requests: int = 10):
        BaseEvaluator.__init__(self, config, num_parallel_requests)
        self.eval_config = RichTranscriptionASREvaluatorConfig(**self.config)
        self.eval_config.evaluation_mode = normalize_evaluation_mode(
            self.eval_config.evaluation_mode
        )

    async def eval_single(self, data_point: dict[str, Any]) -> dict[str, Any]:
        sample = dict(data_point)
        generation = extract_asr_text(coerce_text_field(sample.get("generation", "")))
        if self.eval_config.strip_helpful_prefixes:
            generation = strip_helpful_prefixes(generation)
        sample["generation"] = generation
        return evaluate_rich_transcription_asr_sample(
            sample, self.eval_config.evaluation_mode
        )
