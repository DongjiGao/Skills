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

"""Corpus-level ASCEND ASR-adapted and literal-upstream MER/CER.

The legacy ``mer`` and ``cer`` result keys retain their historical
symmetric-normalized meaning. Explicitly named fields distinguish that
audio-LLM adaptation from the official code's raw-hypothesis behavior.
"""

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
        self.upstream_mer_errors = 0
        self.upstream_mer_ref_units = 0
        self.upstream_cer_errors = 0
        self.upstream_cer_ref_chars = 0

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
        self.upstream_mer_errors += int(prediction["mer_upstream_literal_errors"])
        self.upstream_mer_ref_units += int(prediction["mer_upstream_literal_ref_units"])
        self.upstream_cer_errors += int(prediction["cer_upstream_literal_errors"])
        self.upstream_cer_ref_chars += int(prediction["cer_upstream_literal_ref_chars"])
        self._compute_pass_at_k(predictions, [answer])
        self._compute_majority_at_k(predictions, [answer])

    def get_metrics(self):
        metrics = super().get_metrics()
        for values in metrics.values():
            values["success_rate"] = values.pop("correct", 0.0)
            values["mer"] = round(100.0 * self.mer_errors / self.mer_ref_units, 2)
            values["cer"] = round(100.0 * self.cer_errors / self.cer_ref_chars, 2)
            values["mer_asr_adapted"] = values["mer"]
            values["cer_asr_adapted"] = values["cer"]
            values["mer_errors"] = self.mer_errors
            values["mer_ref_units"] = self.mer_ref_units
            values["cer_errors"] = self.cer_errors
            values["cer_ref_chars"] = self.cer_ref_chars
            values["mer_asr_adapted_errors"] = self.mer_errors
            values["mer_asr_adapted_ref_units"] = self.mer_ref_units
            values["cer_asr_adapted_errors"] = self.cer_errors
            values["cer_asr_adapted_ref_chars"] = self.cer_ref_chars
            values["mer_upstream_literal"] = round(100.0 * self.upstream_mer_errors / self.upstream_mer_ref_units, 2)
            values["cer_upstream_literal"] = round(100.0 * self.upstream_cer_errors / self.upstream_cer_ref_chars, 2)
            values["mer_upstream_literal_errors"] = self.upstream_mer_errors
            values["mer_upstream_literal_ref_units"] = self.upstream_mer_ref_units
            values["cer_upstream_literal_errors"] = self.upstream_cer_errors
            values["cer_upstream_literal_ref_chars"] = self.upstream_cer_ref_chars
        return metrics

    def evaluations_to_print(self):
        return ["pass@1"]

    def metrics_to_print(self):
        return {
            "mer_asr_adapted": as_percentage,
            "cer_asr_adapted": as_percentage,
            "mer_upstream_literal": as_percentage,
            "cer_upstream_literal": as_percentage,
            "success_rate": as_percentage,
            "mer_asr_adapted_errors": as_int,
            "mer_asr_adapted_ref_units": as_int,
            "cer_asr_adapted_errors": as_int,
            "cer_asr_adapted_ref_chars": as_int,
            "mer_upstream_literal_errors": as_int,
            "mer_upstream_literal_ref_units": as_int,
            "cer_upstream_literal_errors": as_int,
            "cer_upstream_literal_ref_chars": as_int,
            "num_entries": as_int,
        }
