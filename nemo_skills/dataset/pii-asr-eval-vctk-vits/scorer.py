# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Deterministic metrics for typed PII redaction from speech."""

from __future__ import annotations

import re
from collections import Counter
from functools import cache
from importlib.metadata import version
from typing import Any

from nemo_skills.evaluation.evaluator.base import BaseEvaluator, BaseEvaluatorConfig
from nemo_skills.evaluation.metrics.base import BaseMetrics, as_int, as_percentage
from nemo_skills.utils import nested_dataclass

NORMALIZER_DISTRIBUTION = "whisper-normalizer"
NORMALIZER_VERSION = "0.1.12"
_PLACEHOLDER_RE = re.compile(r"<[a-z]+(?:-[a-z]+)*>")


@cache
def _english_text_normalizer():
    actual_version = version(NORMALIZER_DISTRIBUTION)
    if actual_version != NORMALIZER_VERSION:
        raise RuntimeError(f"PII-ASR requires {NORMALIZER_DISTRIBUTION}=={NORMALIZER_VERSION}; found {actual_version}")
    from whisper_normalizer.english import EnglishTextNormalizer

    return EnglishTextNormalizer()


def normalize_with_placeholders(text: object) -> str:
    value = str(text)
    tokens: list[str] = []
    cursor = 0
    for match in _PLACEHOLDER_RE.finditer(value):
        normalized = _english_text_normalizer()(value[cursor : match.start()]).strip()
        if normalized:
            tokens.extend(normalized.split())
        tokens.append(match.group(0))
        cursor = match.end()
    normalized = _english_text_normalizer()(value[cursor:]).strip()
    if normalized:
        tokens.extend(normalized.split())
    return " ".join(tokens)


def word_error_counts(reference: list[str], hypothesis: list[str]) -> dict[str, int]:
    ref_len, hyp_len = len(reference), len(hypothesis)
    distances = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
    for index in range(ref_len + 1):
        distances[index][0] = index
    for index in range(hyp_len + 1):
        distances[0][index] = index
    for ref_index in range(1, ref_len + 1):
        for hyp_index in range(1, hyp_len + 1):
            if reference[ref_index - 1] == hypothesis[hyp_index - 1]:
                distances[ref_index][hyp_index] = distances[ref_index - 1][hyp_index - 1]
            else:
                distances[ref_index][hyp_index] = 1 + min(distances[ref_index - 1][hyp_index - 1], distances[ref_index][hyp_index - 1], distances[ref_index - 1][hyp_index])
    substitutions = insertions = deletions = 0
    ref_index, hyp_index = ref_len, hyp_len
    while ref_index > 0 or hyp_index > 0:
        if ref_index > 0 and hyp_index > 0 and reference[ref_index - 1] == hypothesis[hyp_index - 1]:
            ref_index -= 1
            hyp_index -= 1
        elif ref_index > 0 and hyp_index > 0 and distances[ref_index][hyp_index] == distances[ref_index - 1][hyp_index - 1] + 1:
            substitutions += 1
            ref_index -= 1
            hyp_index -= 1
        elif hyp_index > 0 and distances[ref_index][hyp_index] == distances[ref_index][hyp_index - 1] + 1:
            insertions += 1
            hyp_index -= 1
        else:
            deletions += 1
            ref_index -= 1
    return {"errors": substitutions + insertions + deletions, "substitutions": substitutions, "insertions": insertions, "deletions": deletions, "ref_words": ref_len}


def _compact_normalized(text: object) -> str:
    return re.sub(r"[^a-z0-9]", "", _english_text_normalizer()(str(text)).casefold())


def evaluate_sample(data_point: dict[str, Any]) -> dict[str, Any]:
    reference = normalize_with_placeholders(data_point["expected_answer"])
    generation = str(data_point.get("generation") or "").strip()
    hypothesis = normalize_with_placeholders(generation)
    counts = word_error_counts(reference.split(), hypothesis.split())
    expected = Counter(data_point["expected_placeholders"])
    predicted = Counter(_PLACEHOLDER_RE.findall(generation))
    placeholder_tp = sum((expected & predicted).values())
    placeholder_fp = sum((predicted - expected).values())
    placeholder_fn = sum((expected - predicted).values())
    compact_generation = _compact_normalized(generation)
    entity_texts = data_point.get("pii_entity_texts") or []
    leaked_entities = sum(bool(value) and value in compact_generation for value in (_compact_normalized(text) for text in entity_texts))
    return {"generation": generation, "text": reference, "pred_text": hypothesis, "wer_errors": counts["errors"], "wer_substitutions": counts["substitutions"], "wer_insertions": counts["insertions"], "wer_deletions": counts["deletions"], "wer_ref_words": counts["ref_words"], "placeholder_tp": placeholder_tp, "placeholder_fp": placeholder_fp, "placeholder_fn": placeholder_fn, "placeholder_exact": expected == predicted, "entity_count": len(entity_texts), "leaked_entities": leaked_entities, "sample_has_entity_leakage": leaked_entities > 0, "missing_prediction": not bool(generation), "is_correct": counts["errors"] == 0 and expected == predicted}


@nested_dataclass(kw_only=True)
class PIIASRRedactionEvaluatorConfig(BaseEvaluatorConfig):
    pass


class PIIASRRedactionEvaluator(BaseEvaluator):
    async def eval_single(self, data_point: dict[str, Any]) -> dict[str, Any]:
        return evaluate_sample(data_point)


class PIIASRRedactionMetrics(BaseMetrics):
    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        super().__init__(compute_no_answer=compute_no_answer)
        if max_k != 1:
            raise ValueError("PIIASRRedactionMetrics supports one greedy generation only")

    def reset(self):
        super().reset()
        self.wer_errors = self.wer_ref_words = 0
        self.wer_substitutions = self.wer_insertions = self.wer_deletions = 0
        self.placeholder_tp = self.placeholder_fp = self.placeholder_fn = 0
        self.placeholder_exact_samples = 0
        self.entity_count = self.leaked_entities = self.samples_with_entity_leakage = 0
        self.exact_matches = self.missing_predictions = 0

    def _get_score_dict(self, prediction):
        return {"exact": bool(prediction.get("is_correct", False))}

    def get_incorrect_sample(self, prediction):
        prediction = prediction.copy()
        expected = prediction.get("expected_placeholders") or []
        entities = prediction.get("pii_entity_texts") or []
        ref_words = max(1, len(normalize_with_placeholders(prediction.get("expected_answer", "")).split()))
        prediction.update(generation="", pred_text="", wer_errors=ref_words, wer_substitutions=0, wer_insertions=0, wer_deletions=ref_words, wer_ref_words=ref_words, placeholder_tp=0, placeholder_fp=0, placeholder_fn=len(expected), placeholder_exact=False, entity_count=len(entities), leaked_entities=0, sample_has_entity_leakage=False, missing_prediction=True, is_correct=False)
        return prediction

    def update(self, predictions):
        if len(predictions) != 1:
            raise ValueError("PIIASRRedactionMetrics expects exactly one prediction per sample")
        super().update(predictions)
        prediction = predictions[0]
        for name in ("wer_errors", "wer_ref_words", "wer_substitutions", "wer_insertions", "wer_deletions", "placeholder_tp", "placeholder_fp", "placeholder_fn", "entity_count", "leaked_entities"):
            setattr(self, name, getattr(self, name) + int(prediction[name]))
        self.placeholder_exact_samples += int(bool(prediction["placeholder_exact"]))
        self.samples_with_entity_leakage += int(bool(prediction["sample_has_entity_leakage"]))
        self.exact_matches += int(bool(prediction["is_correct"]))
        self.missing_predictions += int(bool(prediction["missing_prediction"]))

    def get_metrics(self):
        if self.total <= 0 or self.wer_ref_words <= 0 or self.entity_count <= 0:
            raise ValueError("PII-ASR has no predictions/references/entities to score")
        precision_denominator = self.placeholder_tp + self.placeholder_fp
        recall_denominator = self.placeholder_tp + self.placeholder_fn
        precision = self.placeholder_tp / precision_denominator if precision_denominator else 0.0
        recall = self.placeholder_tp / recall_denominator if recall_denominator else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        values = {"wer": round(100.0 * self.wer_errors / self.wer_ref_words, 2), "placeholder_precision": round(100.0 * precision, 2), "placeholder_recall": round(100.0 * recall, 2), "placeholder_f1": round(100.0 * f1, 2), "placeholder_exact_match": round(100.0 * self.placeholder_exact_samples / self.total, 2), "redacted_exact_match": round(100.0 * self.exact_matches / self.total, 2), "entity_leakage_rate": round(100.0 * self.leaked_entities / self.entity_count, 2), "samples_with_entity_leakage": self.samples_with_entity_leakage, "coverage": round(100.0 * (self.total - self.missing_predictions) / self.total, 2), "missing_predictions": self.missing_predictions, "scored_entries": self.total, "entity_count": self.entity_count, "leaked_entities": self.leaked_entities, "placeholder_tp": self.placeholder_tp, "placeholder_fp": self.placeholder_fp, "placeholder_fn": self.placeholder_fn, "wer_errors": self.wer_errors, "wer_ref_words": self.wer_ref_words, "wer_substitutions": self.wer_substitutions, "wer_insertions": self.wer_insertions, "wer_deletions": self.wer_deletions}
        self.update_common_metrics(values)
        return {"pass@1": values}

    def evaluations_to_print(self):
        return ["pass@1"]

    def metrics_to_print(self):
        percentages = {name: as_percentage for name in ("wer", "placeholder_precision", "placeholder_recall", "placeholder_f1", "placeholder_exact_match", "redacted_exact_match", "entity_leakage_rate", "coverage")}
        integers = {name: as_int for name in ("samples_with_entity_leakage", "missing_predictions", "scored_entries", "entity_count", "leaked_entities", "placeholder_tp", "placeholder_fp", "placeholder_fn", "wer_errors", "wer_ref_words", "wer_substitutions", "wer_insertions", "wer_deletions", "num_entries")}
        return percentages | integers
