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

"""Exact public set-token MCQ evaluator shared by MMAR and MMAU.

The implementation follows the official public functions verbatim in meaning:
https://github.com/ddlBoJack/MMAR/blob/main/code/evaluation.py
https://github.com/Sakshi113/MMAU/blob/main/evaluation.py
"""

import json
import re

from tqdm import tqdm

from nemo_skills.evaluation.evaluator.base import BaseEvaluatorConfig
from nemo_skills.utils import nested_dataclass


def official_set_token_match(answer: str, prediction: str, choices: list[str]) -> bool:
    """Return the exact deterministic correctness decision used by MMAR/MMAU."""

    def tokenize(text: str) -> set[str]:
        return set(re.findall(r"\b\w+\b", text.lower()))

    prediction_tokens = tokenize(prediction)
    answer_tokens = tokenize(answer)
    if not prediction_tokens:
        return False

    incorrect_tokens: set[str] = set()
    for choice in choices:
        choice_tokens = tokenize(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)

    return answer_tokens.issubset(prediction_tokens) and prediction_tokens.isdisjoint(incorrect_tokens)


@nested_dataclass(kw_only=True)
class SetTokenMCQEvaluatorConfig(BaseEvaluatorConfig):
    official_output_key: str = "answer_prediction"


def eval_set_token_mcq(cfg) -> None:
    """Grade generations in place and retain each benchmark's official output key."""
    eval_config = SetTokenMCQEvaluatorConfig(**cfg)
    if not eval_config.input_file:
        raise ValueError("set-token-mcq evaluation requires input_file")

    with open(eval_config.input_file, "rt", encoding="utf-8") as fin:
        data = [json.loads(line) for line in fin if line.strip()]

    with open(eval_config.input_file, "wt", encoding="utf-8") as fout:
        for sample in tqdm(data):
            generation = sample.get("generation")
            prediction = generation if isinstance(generation, str) else ""
            answer = sample.get("answer", sample.get("expected_answer"))
            choices = sample.get("choices")
            if not isinstance(answer, str):
                raise TypeError("set-token-mcq sample requires string answer/expected_answer")
            if not isinstance(choices, list) or not all(isinstance(choice, str) for choice in choices):
                raise TypeError("set-token-mcq sample requires a list of string choices")

            is_correct = official_set_token_match(answer, prediction, choices)
            sample[eval_config.official_output_key] = prediction
            sample["predicted_answer"] = prediction.strip() or None
            sample["symbolic_correct"] = is_correct
            sample["match"] = int(is_correct)
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
