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

"""Official leaf-level AIR-Bench Foundation accuracy reporting.

The public scorer prints one accuracy for each ``task_name_dataset_name`` leaf
and does not define an overall Foundation accuracy. Consequently a calculator
that observes multiple leaves reports only the official ``total_sum`` and
``fail_num`` counts; it never invents a micro or macro aggregate.
"""

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_int, as_percentage


class AIRBenchFoundationMetrics(BaseMetrics):
    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        if max_k != 1:
            raise ValueError("AIRBenchFoundationMetrics supports one greedy generation only")
        super().__init__(compute_no_answer=compute_no_answer)

    def reset(self):
        super().reset()
        self.max_k = 1
        self.official_total = 0
        self.official_correct = 0
        self.fail_num = 0
        self.leaves: set[str] = set()

    def _get_score_dict(self, prediction):
        return {"correct": bool(prediction.get("symbolic_correct", False))}

    def get_incorrect_sample(self, prediction):
        prediction = prediction.copy()
        prediction["official_parse_valid"] = True
        prediction["symbolic_correct"] = False
        prediction["match"] = 0
        return prediction

    def update(self, predictions):
        if len(predictions) != 1:
            raise ValueError("AIRBenchFoundationMetrics expects exactly one prediction per sample")
        super().update(predictions)
        prediction = predictions[0]
        leaf = prediction.get("air_bench_foundation_leaf", prediction.get("subset_for_metrics"))
        if not isinstance(leaf, str) or not leaf:
            raise ValueError("AIR-Bench Foundation prediction is missing its official leaf identifier")
        self.leaves.add(leaf)
        if bool(prediction.get("official_parse_valid", False)):
            self.official_total += 1
            self.official_correct += int(bool(prediction.get("symbolic_correct", False)))
        else:
            self.fail_num += 1

    def get_metrics(self):
        values = {}
        self.update_common_metrics(values)
        values.update(
            {
                "total_sum": self.official_total,
                "correct_num": self.official_correct,
                "fail_num": self.fail_num,
                "num_leaves": len(self.leaves),
            }
        )
        # ComputeMetrics creates one calculator per subset and an _all_
        # calculator. Only the single-leaf calculators reproduce an official
        # accuracy; the mixed _all_ calculator intentionally has none.
        if len(self.leaves) == 1:
            values["accuracy"] = (
                round(100.0 * self.official_correct / self.official_total, 2) if self.official_total else 0.0
            )
        return {"pass@1": values}

    def evaluations_to_print(self):
        return ["pass@1"]

    def metrics_to_print(self):
        return {
            "accuracy": as_percentage,
            "total_sum": as_int,
            "correct_num": as_int,
            "fail_num": as_int,
            "num_entries": as_int,
        }
