# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Corpus-level official ASCEND MER (primary) and CER (secondary)."""

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_int, as_percentage


class ASCENDMetrics(BaseMetrics):
    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        super().__init__(compute_no_answer=compute_no_answer)
        if max_k != 1:
            raise ValueError("ASCENDMetrics supports one greedy generation only")
        self.max_k = 1
        self.mer_errors = 0
        self.mer_ref_units = 0
        self.cer_errors = 0
        self.cer_ref_chars = 0

    def _get_score_dict(self, prediction):
        return {"correct": bool(prediction["is_correct"])}

    def get_incorrect_sample(self, prediction):
        prediction = prediction.copy()
        prediction["is_correct"] = False
        return prediction

    def update(self, predictions):
        if len(predictions) != 1:
            raise ValueError("ASCENDMetrics expects exactly one prediction per sample")
        super().update(predictions)
        prediction = predictions[0]
        self.mer_errors += int(prediction["mer_errors"])
        self.mer_ref_units += int(prediction["mer_ref_units"])
        self.cer_errors += int(prediction["cer_errors"])
        self.cer_ref_chars += int(prediction["cer_ref_chars"])
        answer = (prediction.get("generation") or "").strip() or None
        self._compute_pass_at_k(predictions, [answer])
        self._compute_majority_at_k(predictions, [answer])

    def get_metrics(self):
        metrics = super().get_metrics()
        for values in metrics.values():
            values["success_rate"] = values.pop("correct", 0.0)
            values["mer"] = round(100.0 * self.mer_errors / self.mer_ref_units, 2)
            values["cer"] = round(100.0 * self.cer_errors / self.cer_ref_chars, 2)
            values["mer_errors"] = self.mer_errors
            values["mer_ref_units"] = self.mer_ref_units
            values["cer_errors"] = self.cer_errors
            values["cer_ref_chars"] = self.cer_ref_chars
        return metrics

    def evaluations_to_print(self):
        return ["pass@1"]

    def metrics_to_print(self):
        return {
            "mer": as_percentage,
            "cer": as_percentage,
            "success_rate": as_percentage,
            "mer_errors": as_int,
            "mer_ref_units": as_int,
            "cer_errors": as_int,
            "cer_ref_chars": as_int,
            "num_entries": as_int,
        }
