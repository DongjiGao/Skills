# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import json
import logging
import re
from typing import Union

from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


def read_predictions(predictions, line_idx, file_handles):
    data = []
    for file_idx, prediction in enumerate(predictions):
        try:
            prediction_dict = json.loads(prediction)
        except Exception as e:
            LOG.error("\n\n ***** Error reading line %s in file %s: %s", line_idx + 1, file_handles[file_idx].name, e)
            raise
        data.append(prediction_dict)

    return data


def is_correct_judgement(judgement, return_none=False) -> Union[bool, None]:
    """Parse judgement text to determine correctness.

    Supports multiple formats:
    - "Judgement: Yes/No" (standard format, also handles markdown bold)
    - "\\boxed{Correct/Incorrect}" (IMO AnswerBench format)
    - "<points>N out of 7</points>" (IMO ProofBench format: 6-7 = correct, 0-1 = incorrect)
    """
    if judgement:
        match = re.search(r"\*{0,2}Judgement\*{0,2}\s*:", judgement, re.IGNORECASE)
        if match:
            verdict = judgement[match.end() :].strip().lstrip("*").strip()
            if verdict.lower().startswith("yes"):
                return True
            if verdict.lower().startswith("no"):
                return False

        boxed_match = re.search(r"\\boxed\s*\{\s*(Correct|Incorrect)\s*\}", judgement, re.IGNORECASE)
        if boxed_match:
            verdict = boxed_match.group(1).lower()
            return verdict == "correct"

        points_match = re.search(r"<points>\s*(\d+)\s*out of 7\s*</points>", judgement, re.IGNORECASE)
        if points_match:
            points = int(points_match.group(1))
            return points >= 6

    if return_none:
        return None
    return False
