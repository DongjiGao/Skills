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

from nemo_skills.evaluation.evaluator import is_evaluator_registered
from nemo_skills.evaluation.evaluator.language_id import evaluate_language_id_sample
from nemo_skills.evaluation.metrics.language_id_metrics import LanguageIDMetrics
from nemo_skills.evaluation.metrics.map_metrics import get_metrics


def _evaluated(label: str, generation: str, group: str | None = None) -> dict:
    sample = {"expected_answer": label, "generation": generation, "extra_fields": {"label_group": group}}
    return {**sample, **evaluate_language_id_sample(sample)}


def test_language_id_exact_label_contract_and_optional_vocabulary():
    valid = {"en", "de", "fr"}
    exact = evaluate_language_id_sample({"expected_answer": "en", "generation": " en\n"}, valid)
    wrong_case = evaluate_language_id_sample({"expected_answer": "en", "generation": "EN"}, valid)
    assert exact["lid_correct"] is True
    assert exact["valid_label"] is True
    assert wrong_case["lid_correct"] is False
    assert wrong_case["valid_label"] is False


def test_language_id_generic_micro_and_label_macro():
    metrics = LanguageIDMetrics(max_k=1)
    for prediction in [
        _evaluated("en", "en", "g1"),
        _evaluated("en", "de", "g1"),
        _evaluated("de", "de", "g1"),
        _evaluated("fr", "de", "g2"),
    ]:
        metrics.update([prediction])
    output = metrics.get_metrics()["pass@1"]
    assert output["accuracy_micro"] == 50.0
    assert output["accuracy_macro"] == 50.0
    assert output["num_labels"] == 3
    assert output["accuracy::en"] == 50.0
    assert output["accuracy::de"] == 100.0
    assert output["accuracy::fr"] == 0.0
    assert output["accuracy_group::g1"] == 66.67
    assert output["accuracy_group_macro::g1"] == 75.0


def test_language_id_registration():
    assert is_evaluator_registered("language_id")
    assert isinstance(get_metrics("language_id", max_k=1), LanguageIDMetrics)
