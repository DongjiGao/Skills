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

"""Accuracy aggregation for deterministic MMAR/MMAU set-token scoring."""

from nemo_skills.evaluation.metrics.base import as_int, as_percentage
from nemo_skills.evaluation.metrics.math_metrics import MathMetrics


class SetTokenMCQMetrics(MathMetrics):
    def __init__(self, compute_no_answer: bool = True):
        super().__init__(
            compute_no_answer=compute_no_answer,
            question_key="question",
            answer_key="predicted_answer",
        )

    def _get_score_dict(self, prediction: dict) -> dict[str, bool]:
        return {"accuracy": bool(prediction.get("symbolic_correct", False))}

    def metrics_to_print(self):
        metrics = {
            "num_entries": as_int,
            "avg_tokens": as_int,
            "gen_seconds": as_int,
            "accuracy": as_percentage,
        }
        if self.compute_no_answer:
            metrics["no_answer"] = as_percentage
        return metrics
