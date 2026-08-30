# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Official aggregate WER/OIWER scoring for public Monsoon ASR splits."""

from __future__ import annotations

from functools import cache
from importlib.metadata import version
from typing import Any

from nemo_skills.evaluation.evaluator.base import BaseEvaluator, BaseEvaluatorConfig
from nemo_skills.evaluation.evaluator.open_asr_normalizer import OpenASREnglishTextNormalizer
from nemo_skills.evaluation.metrics.base import BaseMetrics, as_int, as_percentage
from nemo_skills.utils import nested_dataclass

OIWER_DISTRIBUTION = "voi-oiwer"
OIWER_VERSION = "0.1.4"


@cache
def _english_normalizer() -> OpenASREnglishTextNormalizer:
    return OpenASREnglishTextNormalizer()


@cache
def _official_oiwer():
    actual_version = version(OIWER_DISTRIBUTION)
    if actual_version != OIWER_VERSION:
        raise RuntimeError(f"Monsoon hi-IN requires {OIWER_DISTRIBUTION}=={OIWER_VERSION}; found {actual_version}")
    from voi_oiwer import oiwer

    return oiwer


def word_error_counts(reference: list[str], hypothesis: list[str]) -> dict[str, int]:
    """Return deterministic Levenshtein S/I/D counts."""
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
            raise RuntimeError("Monsoon WER DP backtrace reached an inconsistent state")
    return {
        "errors": substitutions + insertions + deletions,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "ref_words": ref_len,
    }


def _evaluate_english(data_point: dict[str, Any], generation: str) -> dict[str, Any]:
    reference = _english_normalizer()(str(data_point["expected_answer"]))
    hypothesis = _english_normalizer()(generation)
    reference_words = reference.split()
    if not reference_words:
        raise ValueError("Monsoon en-IN reference normalized to an empty string")
    counts = word_error_counts(reference_words, hypothesis.split())
    return {
        "monsoon_metric": "wer",
        "text": reference,
        "pred_text": hypothesis,
        **{f"monsoon_{name}": value for name, value in counts.items()},
    }


def _evaluate_hindi(data_point: dict[str, Any], generation: str) -> dict[str, Any]:
    lattice = data_point.get("reference_lattice")
    if not isinstance(lattice, list) or not lattice or any(not isinstance(slot, list) or not slot for slot in lattice):
        raise ValueError("Monsoon hi-IN requires a nonempty list-of-lists reference_lattice")
    result = _official_oiwer()(generation, lattice, "hindi")
    score, counts, ref_words = float(result[0]), result[4], int(result[5])
    if len(counts) != 3 or ref_words <= 0:
        raise ValueError("voi-oiwer returned an invalid count/reference contract")
    insertions, deletions, substitutions = (int(value) for value in counts)
    errors = insertions + deletions + substitutions
    if abs(score - 100.0 * errors / ref_words) > 1e-6:
        raise ValueError("voi-oiwer score/count invariant failed")
    return {
        "monsoon_metric": "oiwer",
        "text": data_point.get("expected_answer", ""),
        "pred_text": generation,
        "monsoon_errors": errors,
        "monsoon_substitutions": substitutions,
        "monsoon_insertions": insertions,
        "monsoon_deletions": deletions,
        "monsoon_ref_words": ref_words,
        "oiwer_utterance_diagnostic": score / 100.0,
    }


def evaluate_monsoon_sample(data_point: dict[str, Any]) -> dict[str, Any]:
    generation = str(data_point.get("generation") or "").strip()
    language = data_point.get("monsoon_language")
    if language == "en-in":
        values = _evaluate_english(data_point, generation)
    elif language == "hi-in":
        values = _evaluate_hindi(data_point, generation)
    else:
        raise ValueError(f"Unsupported Monsoon language: {language!r}")
    values.update(
        generation=generation,
        missing_prediction=not bool(generation),
        is_correct=values["monsoon_errors"] == 0,
    )
    return values


@nested_dataclass(kw_only=True)
class MonsoonASREvaluatorConfig(BaseEvaluatorConfig):
    pass


class MonsoonASREvaluator(BaseEvaluator):
    async def eval_single(self, data_point: dict[str, Any]) -> dict[str, Any]:
        return evaluate_monsoon_sample(data_point)


class MonsoonASRMetrics(BaseMetrics):
    """Corpus-micro official WER or OIWER with raw count evidence."""

    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        super().__init__(compute_no_answer=compute_no_answer)
        if max_k != 1:
            raise ValueError("MonsoonASRMetrics supports one greedy generation only")
        self.max_k = 1

    def reset(self):
        super().reset()
        self.metric_name: str | None = None
        self.errors = 0
        self.ref_words = 0
        self.substitutions = 0
        self.insertions = 0
        self.deletions = 0
        self.missing_predictions = 0

    def _get_score_dict(self, prediction):
        return {"exact": int(prediction.get("monsoon_errors", 1)) == 0}

    def get_incorrect_sample(self, prediction):
        prediction = prediction.copy()
        ref_words = max(1, int(prediction.get("monsoon_ref_words", 0)))
        prediction.update(
            monsoon_errors=ref_words,
            monsoon_substitutions=0,
            monsoon_insertions=0,
            monsoon_deletions=ref_words,
            monsoon_ref_words=ref_words,
            missing_prediction=True,
            generation="",
            is_correct=False,
        )
        return prediction

    def update(self, predictions):
        if len(predictions) != 1:
            raise ValueError("MonsoonASRMetrics expects exactly one prediction per sample")
        prediction = predictions[0]
        metric_name = str(prediction["monsoon_metric"])
        if metric_name not in {"wer", "oiwer"}:
            raise ValueError(f"Unsupported Monsoon metric: {metric_name!r}")
        if self.metric_name is not None and metric_name != self.metric_name:
            raise ValueError("Monsoon WER and OIWER rows must be summarized as separate profiles")
        self.metric_name = metric_name
        super().update(predictions)
        self.errors += int(prediction["monsoon_errors"])
        self.ref_words += int(prediction["monsoon_ref_words"])
        self.substitutions += int(prediction["monsoon_substitutions"])
        self.insertions += int(prediction["monsoon_insertions"])
        self.deletions += int(prediction["monsoon_deletions"])
        self.missing_predictions += int(bool(prediction["missing_prediction"]))

    def get_metrics(self):
        if self.total <= 0 or self.ref_words <= 0 or self.metric_name is None:
            raise ValueError("Monsoon ASR has no predictions/reference words to score")
        values = {
            self.metric_name: round(100.0 * self.errors / self.ref_words, 2),
            "coverage": round(100.0 * (self.total - self.missing_predictions) / self.total, 2),
            "missing_predictions": self.missing_predictions,
            "scored_entries": self.total,
            "errors": self.errors,
            "ref_words": self.ref_words,
            "substitutions": self.substitutions,
            "insertions": self.insertions,
            "deletions": self.deletions,
        }
        self.update_common_metrics(values)
        return {"pass@1": values}

    def evaluations_to_print(self):
        return ["pass@1"]

    def metrics_to_print(self):
        return {
            "wer": as_percentage,
            "oiwer": as_percentage,
            "coverage": as_percentage,
            "missing_predictions": as_int,
            "scored_entries": as_int,
            "errors": as_int,
            "ref_words": as_int,
            "substitutions": as_int,
            "insertions": as_int,
            "deletions": as_int,
            "num_entries": as_int,
        }
