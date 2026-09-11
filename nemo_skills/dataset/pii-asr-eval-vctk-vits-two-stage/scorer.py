# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Final-redaction metrics plus first-stage ASR diagnostics."""

from __future__ import annotations

import importlib
from typing import Any

from nemo_skills.evaluation.evaluator.base import BaseEvaluator
from nemo_skills.evaluation.metrics.base import as_int, as_percentage

_base = importlib.import_module("nemo_skills.dataset.pii-asr-eval-vctk-vits.scorer")


def evaluate_sample(data_point: dict[str, Any]) -> dict[str, Any]:
    values = _base.evaluate_sample(data_point)
    reference = _base.normalize_with_placeholders(data_point["expected_asr"])
    hypothesis = _base.normalize_with_placeholders(data_point.get("intermediate_asr") or "")
    counts = _base.word_error_counts(reference.split(), hypothesis.split())
    values.update(
        first_stage_asr=data_point.get("intermediate_asr") or "",
        first_stage_wer_errors=counts["errors"],
        first_stage_wer_substitutions=counts["substitutions"],
        first_stage_wer_insertions=counts["insertions"],
        first_stage_wer_deletions=counts["deletions"],
        first_stage_wer_ref_words=counts["ref_words"],
        first_stage_missing=not bool(data_point.get("intermediate_asr")),
    )
    return values


class PIIASRTwoStageEvaluator(BaseEvaluator):
    async def eval_single(self, data_point: dict[str, Any]) -> dict[str, Any]:
        return evaluate_sample(data_point)


class PIIASRTwoStageMetrics(_base.PIIASRRedactionMetrics):
    def reset(self):
        super().reset()
        self.first_stage_wer_errors = self.first_stage_wer_ref_words = 0
        self.first_stage_wer_substitutions = self.first_stage_wer_insertions = self.first_stage_wer_deletions = 0
        self.first_stage_missing = 0

    def update(self, predictions):
        super().update(predictions)
        prediction = predictions[0]
        for name in (
            "first_stage_wer_errors",
            "first_stage_wer_ref_words",
            "first_stage_wer_substitutions",
            "first_stage_wer_insertions",
            "first_stage_wer_deletions",
        ):
            setattr(self, name, getattr(self, name) + int(prediction[name]))
        self.first_stage_missing += int(bool(prediction["first_stage_missing"]))

    def get_metrics(self):
        result = super().get_metrics()
        if self.first_stage_wer_ref_words <= 0:
            raise ValueError("PII-ASR two-stage evaluation has no first-stage ASR references")
        result["pass@1"].update(
            first_stage_asr_wer=round(100.0 * self.first_stage_wer_errors / self.first_stage_wer_ref_words, 2),
            first_stage_missing=self.first_stage_missing,
            first_stage_wer_errors=self.first_stage_wer_errors,
            first_stage_wer_ref_words=self.first_stage_wer_ref_words,
            first_stage_wer_substitutions=self.first_stage_wer_substitutions,
            first_stage_wer_insertions=self.first_stage_wer_insertions,
            first_stage_wer_deletions=self.first_stage_wer_deletions,
        )
        return result

    def metrics_to_print(self):
        return super().metrics_to_print() | {
            "first_stage_asr_wer": as_percentage,
            "first_stage_missing": as_int,
            "first_stage_wer_errors": as_int,
            "first_stage_wer_ref_words": as_int,
            "first_stage_wer_substitutions": as_int,
            "first_stage_wer_insertions": as_int,
            "first_stage_wer_deletions": as_int,
        }
