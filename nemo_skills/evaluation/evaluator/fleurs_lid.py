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

"""Exact-label evaluator for FLEURS language identification."""

from nemo_skills.dataset.fleurs.languages import HR7_LID_LOCALES, get_lang_group
from nemo_skills.evaluation.evaluator.language_id import (
    LanguageIDEvaluator,
    LanguageIDEvaluatorConfig,
    evaluate_language_id_sample,
)
from nemo_skills.utils import nested_dataclass


def evaluate_fleurs_lid_sample(data_point: dict) -> dict:
    result = evaluate_language_id_sample(data_point, valid_labels=HR7_LID_LOCALES)
    result["locale"] = result["label"]
    result["lang_group"] = get_lang_group(result["label"])
    result["label_group"] = result["lang_group"]
    return result


@nested_dataclass(kw_only=True)
class FleursLIDEvaluatorConfig(LanguageIDEvaluatorConfig):
    pass


class FleursLIDEvaluator(LanguageIDEvaluator):
    async def eval_single(self, data_point: dict) -> dict:
        return evaluate_fleurs_lid_sample(data_point)
