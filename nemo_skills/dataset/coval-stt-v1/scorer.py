# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Protocol-faithful quality scorer for the fixed Coval STT v1 profile.

Coval normalizes each reference/hypothesis with EnglishTextNormalizer, computes
one WER ratio per utterance, and stores that ratio as a percentage. Its public
reporting aggregation averages those observation values, so utterance-macro WER
is the official quality metric. Corpus-micro WER is retained here only as a
diagnostic. This implementation owns deterministic Levenshtein DP and does not
depend on a potentially incompatible jiwer runtime.
"""

from __future__ import annotations

from functools import cache
from importlib.metadata import version
from typing import Any

from nemo_skills.evaluation.evaluator.base import BaseEvaluator, BaseEvaluatorConfig
from nemo_skills.evaluation.metrics.base import BaseMetrics, as_int, as_percentage
from nemo_skills.utils import nested_dataclass

NORMALIZER_DISTRIBUTION = "whisper-normalizer"
NORMALIZER_VERSION = "0.1.12"


@cache
def _english_text_normalizer():
    actual_version = version(NORMALIZER_DISTRIBUTION)
    if actual_version != NORMALIZER_VERSION:
        raise RuntimeError(
            f"Coval STT v1 requires {NORMALIZER_DISTRIBUTION}=={NORMALIZER_VERSION}; found {actual_version}"
        )
    from whisper_normalizer.english import EnglishTextNormalizer

    return EnglishTextNormalizer()


def normalize_text(text: object) -> str:
    """Apply Coval's pinned quality normalizer exactly once."""
    return _english_text_normalizer()(str(text))


def word_error_counts(reference: list[str], hypothesis: list[str]) -> dict[str, int]:
    """Return deterministic Levenshtein S/I/D counts with Coval tie ordering."""
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    distances = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
    for ref_index in range(ref_len + 1):
        distances[ref_index][0] = ref_index
    for hyp_index in range(hyp_len + 1):
        distances[0][hyp_index] = hyp_index
    for ref_index in range(1, ref_len + 1):
        for hyp_index in range(1, hyp_len + 1):
            if reference[ref_index - 1] == hypothesis[hyp_index - 1]:
                distances[ref_index][hyp_index] = distances[ref_index - 1][hyp_index - 1]
            else:
                distances[ref_index][hyp_index] = 1 + min(
                    distances[ref_index - 1][hyp_index - 1],
                    distances[ref_index][hyp_index - 1],
                    distances[ref_index - 1][hyp_index],
                )

    substitutions = insertions = deletions = 0
    ref_index, hyp_index = ref_len, hyp_len
    while ref_index > 0 or hyp_index > 0:
        if ref_index > 0 and hyp_index > 0 and reference[ref_index - 1] == hypothesis[hyp_index - 1]:
            ref_index -= 1
            hyp_index -= 1
        elif (
            ref_index > 0
            and hyp_index > 0
            and distances[ref_index][hyp_index] == distances[ref_index - 1][hyp_index - 1] + 1
        ):
            substitutions += 1
            ref_index -= 1
            hyp_index -= 1
        elif hyp_index > 0 and distances[ref_index][hyp_index] == distances[ref_index][hyp_index - 1] + 1:
            insertions += 1
            hyp_index -= 1
        elif ref_index > 0 and distances[ref_index][hyp_index] == distances[ref_index - 1][hyp_index] + 1:
            deletions += 1
            ref_index -= 1
        else:
            raise RuntimeError("Coval STT v1 DP backtrace reached an inconsistent state")

    errors = substitutions + insertions + deletions
    if errors != distances[ref_len][hyp_len]:
        raise RuntimeError("Coval STT v1 DP error-count invariant failed")
    return {
        "errors": errors,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "ref_words": ref_len,
    }


def evaluate_coval_stt_v1_sample(data_point: dict[str, Any]) -> dict[str, Any]:
    reference = normalize_text(data_point["expected_answer"])
    generation = str(data_point.get("generation") or "").strip()
    hypothesis = normalize_text(generation)
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    if not reference_words:
        raise ValueError("Coval STT v1 reference normalized to an empty string")
    counts = word_error_counts(reference_words, hypothesis_words)
    wer = counts["errors"] / counts["ref_words"]
    return {
        "generation": generation,
        "text": reference,
        "pred_text": hypothesis,
        "coval_wer": wer,
        "wer": wer,
        "wer_errors": counts["errors"],
        "wer_substitutions": counts["substitutions"],
        "wer_insertions": counts["insertions"],
        "wer_deletions": counts["deletions"],
        "wer_ref_words": counts["ref_words"],
        "missing_prediction": not bool(generation),
        "is_correct": counts["errors"] == 0,
    }


@nested_dataclass(kw_only=True)
class CovalSTTV1EvaluatorConfig(BaseEvaluatorConfig):
    pass


class CovalSTTV1Evaluator(BaseEvaluator):
    async def eval_single(self, data_point: dict[str, Any]) -> dict[str, Any]:
        return evaluate_coval_stt_v1_sample(data_point)


class CovalSTTV1Metrics(BaseMetrics):
    """Coval utterance-macro WER plus explicitly diagnostic corpus WER."""

    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        super().__init__(compute_no_answer=compute_no_answer)
        if max_k != 1:
            raise ValueError("CovalSTTV1Metrics supports one greedy generation only")
        self.max_k = 1

    def reset(self):
        super().reset()
        self.macro_wer_sum = 0.0
        self.wer_errors = 0
        self.wer_ref_words = 0
        self.wer_substitutions = 0
        self.wer_insertions = 0
        self.wer_deletions = 0
        self.missing_predictions = 0

    def _get_score_dict(self, prediction):
        return {"exact": int(prediction.get("wer_errors", 1)) == 0}

    def get_incorrect_sample(self, prediction):
        prediction = prediction.copy()
        ref_words = max(1, int(prediction.get("wer_ref_words", 0)))
        prediction.update(
            coval_wer=1.0,
            wer=1.0,
            wer_errors=ref_words,
            wer_substitutions=0,
            wer_insertions=0,
            wer_deletions=ref_words,
            wer_ref_words=ref_words,
            missing_prediction=True,
            generation="",
            is_correct=False,
        )
        return prediction

    def update(self, predictions):
        if len(predictions) != 1:
            raise ValueError("CovalSTTV1Metrics expects exactly one prediction per sample")
        super().update(predictions)
        prediction = predictions[0]
        self.macro_wer_sum += float(prediction["coval_wer"])
        self.wer_errors += int(prediction["wer_errors"])
        self.wer_ref_words += int(prediction["wer_ref_words"])
        self.wer_substitutions += int(prediction["wer_substitutions"])
        self.wer_insertions += int(prediction["wer_insertions"])
        self.wer_deletions += int(prediction["wer_deletions"])
        self.missing_predictions += int(bool(prediction["missing_prediction"]))

    def get_metrics(self):
        if self.total <= 0 or self.wer_ref_words <= 0:
            raise ValueError("Coval STT v1 has no predictions/reference words to score")
        values = {
            # Official Coval quality rollup: AVG(per-observation WER percent).
            "wer": round(100.0 * self.macro_wer_sum / self.total, 2),
            # Diagnostic only: this is not Coval's published profile aggregation.
            "corpus_wer_diagnostic": round(100.0 * self.wer_errors / self.wer_ref_words, 2),
            "coverage": round(100.0 * (self.total - self.missing_predictions) / self.total, 2),
            "missing_predictions": self.missing_predictions,
            "scored_entries": self.total,
            "wer_errors": self.wer_errors,
            "wer_ref_words": self.wer_ref_words,
            "wer_substitutions": self.wer_substitutions,
            "wer_insertions": self.wer_insertions,
            "wer_deletions": self.wer_deletions,
        }
        self.update_common_metrics(values)
        return {"pass@1": values}

    def evaluations_to_print(self):
        return ["pass@1"]

    def metrics_to_print(self):
        return {
            "wer": as_percentage,
            "corpus_wer_diagnostic": as_percentage,
            "coverage": as_percentage,
            "missing_predictions": as_int,
            "scored_entries": as_int,
            "wer_errors": as_int,
            "wer_ref_words": as_int,
            "wer_substitutions": as_int,
            "wer_insertions": as_int,
            "wer_deletions": as_int,
            "num_entries": as_int,
        }
