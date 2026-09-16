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

"""Exact deterministic AIR-Bench Foundation evaluator.

This reproduces the public ``score_foundation.py`` parser. Parse failures are
flagged and excluded from each leaf's official accuracy denominator, exactly as
in the source scorer; they remain visible through ``fail_num``.
"""

import json
import os

from tqdm import tqdm

from nemo_skills.evaluation.evaluator.base import BaseEvaluatorConfig
from nemo_skills.utils import nested_dataclass


def official_foundation_prediction(response: str) -> str | None:
    """Apply the exact public first-character/second-last-character parser."""
    predict = response.strip().replace("\n", "")
    if predict == "None" or not predict:
        return None
    if predict[0] in "ABCD":
        return predict[0]
    # The source scorer calls out outputs such as ``The answer is A.``.
    if len(predict) > 1 and predict[-2] in "ABCD":
        return predict[-2]
    return None


def official_foundation_ground_truth(data_point: dict) -> str:
    """Map answer text to A-D in the exact source-scorer precedence order."""
    answer = data_point.get("answer_gt")
    for letter in "abcd":
        if answer == data_point.get(f"choice_{letter}"):
            return letter.upper()
    raise ValueError(f"AIR-Bench answer_gt does not match any choice: {answer!r}")


def evaluate_air_bench_foundation_sample(data_point: dict) -> dict:
    generation = data_point.get("generation")
    response = generation if isinstance(generation, str) else ""
    predicted_answer = official_foundation_prediction(response)
    ground_truth = official_foundation_ground_truth(data_point)
    parse_valid = predicted_answer is not None
    is_correct = parse_valid and predicted_answer == ground_truth
    leaf = f"{data_point['task_name']}_{data_point['dataset_name']}"
    return {
        "response": response,
        "predicted_answer": predicted_answer,
        "official_ground_truth": ground_truth,
        "official_parse_valid": parse_valid,
        "symbolic_correct": is_correct,
        "match": int(is_correct),
        "air_bench_foundation_leaf": leaf,
        "subset_for_metrics": leaf,
    }


@nested_dataclass(kw_only=True)
class AIRBenchFoundationEvaluatorConfig(BaseEvaluatorConfig):
    pass


def eval_air_bench_foundation(cfg) -> None:
    """Grade an AIR-Bench result JSONL in place with the official parser."""
    eval_config = AIRBenchFoundationEvaluatorConfig(**cfg)
    if not eval_config.input_file:
        raise ValueError("AIR-Bench Foundation evaluation requires input_file")

    with open(eval_config.input_file, "rt", encoding="utf-8") as source:
        data = [json.loads(line) for line in source if line.strip()]

    temporary_path = f"{eval_config.input_file}.tmp"
    try:
        with open(temporary_path, "wt", encoding="utf-8") as output:
            for sample in tqdm(data):
                sample.update(evaluate_air_bench_foundation_sample(sample))
                output.write(json.dumps(sample, ensure_ascii=False) + "\n")
        # Match other batch evaluators' in-place result contract without ever
        # exposing a partially scored JSONL.
        os.replace(temporary_path, eval_config.input_file)
    finally:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
