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

"""Micro-average selected-mode WER over the D12a and D8 members of a split."""

from __future__ import annotations

from typing import Any


def _aggregate(benchmarks: dict[str, dict[str, Any]], eval_mode: str) -> dict[str, Any]:
    metrics = [
        benchmark[eval_mode]
        for benchmark in benchmarks.values()
        if eval_mode in benchmark
    ]
    if not metrics:
        return {}

    evaluation_modes = {str(item.get("evaluation_mode", "event")) for item in metrics}
    if len(evaluation_modes) != 1:
        raise ValueError(
            f"Cannot aggregate mixed evaluation modes: {sorted(evaluation_modes)}"
        )

    output = {
        "evaluation_mode": next(iter(evaluation_modes)),
        "num_entries": sum(int(item.get("num_entries", 0) or 0) for item in metrics),
        "wer_errors": sum(int(item.get("wer_errors", 0) or 0) for item in metrics),
        "substitutions": sum(
            int(item.get("substitutions", 0) or 0) for item in metrics
        ),
        "insertions": sum(int(item.get("insertions", 0) or 0) for item in metrics),
        "deletions": sum(int(item.get("deletions", 0) or 0) for item in metrics),
        "ref_words": sum(int(item.get("ref_words", 0) or 0) for item in metrics),
    }
    if output["ref_words"] > 0:
        output["wer"] = round(100.0 * output["wer_errors"] / output["ref_words"], 2)
    return output


def compute_score(
    combined_metrics: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Compute one micro-averaged score for every reported evaluation aggregate."""
    eval_modes = sorted(
        {
            eval_mode
            for benchmark_metrics in combined_metrics.values()
            for eval_mode in benchmark_metrics
        }
    )
    return {
        eval_mode: aggregate
        for eval_mode in eval_modes
        if (aggregate := _aggregate(combined_metrics, eval_mode))
    }
