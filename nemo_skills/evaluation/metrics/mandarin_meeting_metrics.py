# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Corpus cpCER aggregation for speaker-attributed Mandarin meetings."""

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_int, as_percentage


class MandarinMeetingMetrics(BaseMetrics):
    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        super().__init__(compute_no_answer=compute_no_answer)
        if max_k != 1:
            raise ValueError("MandarinMeetingMetrics supports one greedy generation only")
        self.max_k = 1
        self.errors = 0
        self.substitutions = 0
        self.insertions = 0
        self.deletions = 0
        self.ref_chars = 0
        self.session_scores: list[float] = []

    def _get_score_dict(self, prediction):
        return {"correct": bool(prediction["is_correct"])}

    def get_incorrect_sample(self, prediction):
        prediction = prediction.copy()
        prediction["is_correct"] = False
        return prediction

    def update(self, predictions):
        if len(predictions) != 1:
            raise ValueError("MandarinMeetingMetrics expects exactly one prediction per sample")
        super().update(predictions)
        prediction = predictions[0]
        self.errors += int(prediction["cpcer_errors"])
        self.substitutions += int(prediction["cpcer_substitutions"])
        self.insertions += int(prediction["cpcer_insertions"])
        self.deletions += int(prediction["cpcer_deletions"])
        self.ref_chars += int(prediction["cpcer_ref_chars"])
        self.session_scores.append(float(prediction["cpcer"]))
        answer = (prediction.get("generation") or "").strip() or None
        self._compute_pass_at_k(predictions, [answer])
        self._compute_majority_at_k(predictions, [answer])

    def get_metrics(self):
        metrics = super().get_metrics()
        for values in metrics.values():
            values["success_rate"] = values.pop("correct", 0.0)
            values["cpcer"] = round(100.0 * self.errors / self.ref_chars, 2)
            values["cpcer_macro"] = round(100.0 * sum(self.session_scores) / len(self.session_scores), 2)
            values["cpcer_errors"] = self.errors
            values["cpcer_substitutions"] = self.substitutions
            values["cpcer_insertions"] = self.insertions
            values["cpcer_deletions"] = self.deletions
            values["cpcer_ref_chars"] = self.ref_chars
        return metrics

    def evaluations_to_print(self):
        return ["pass@1"]

    def metrics_to_print(self):
        return {
            "cpcer": as_percentage,
            "cpcer_macro": as_percentage,
            "success_rate": as_percentage,
            "cpcer_errors": as_int,
            "cpcer_substitutions": as_int,
            "cpcer_insertions": as_int,
            "cpcer_deletions": as_int,
            "cpcer_ref_chars": as_int,
            "num_entries": as_int,
        }
