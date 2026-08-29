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

import importlib

from nemo_skills.dataset.fleurs.languages import FLEURS_LANGUAGES
from nemo_skills.evaluation.evaluator import is_evaluator_registered
from nemo_skills.evaluation.evaluator.fleurs_lid import evaluate_fleurs_lid_sample
from nemo_skills.evaluation.metrics.fleurs_lid_metrics import FleursLIDMetrics
from nemo_skills.evaluation.metrics.map_metrics import get_metrics

prepare = importlib.import_module("nemo_skills.dataset.fleurs-lid.prepare")


def _evaluated(locale: str, generation: str) -> dict:
    sample = {"expected_answer": locale, "generation": generation}
    return {**sample, **evaluate_fleurs_lid_sample(sample)}


def test_fleurs_lid_has_exact_official_label_space_and_pin():
    assert len(prepare.CANONICAL_LABELS) == 102
    assert tuple(sorted(FLEURS_LANGUAGES)) == prepare.CANONICAL_LABELS
    assert len({metadata["group"] for metadata in FLEURS_LANGUAGES.values()}) == 7
    assert prepare.HF_REVISION == "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
    assert "en_us" in prepare.PROMPT and "yue_hant_hk" in prepare.PROMPT


def test_fleurs_lid_evaluator_requires_exact_canonical_output():
    assert evaluate_fleurs_lid_sample({"expected_answer": "en_us", "generation": " en_us\n"})["lid_correct"]
    invalid = evaluate_fleurs_lid_sample({"expected_answer": "en_us", "generation": "EN_US"})
    assert invalid["valid_label"] is False
    assert invalid["lid_correct"] is False


def test_fleurs_lid_micro_locale_macro_and_geographic_reporting():
    metrics = FleursLIDMetrics(max_k=1)
    for prediction in [
        _evaluated("en_us", "en_us"),
        _evaluated("en_us", "fr_fr"),
        _evaluated("fr_fr", "fr_fr"),
        _evaluated("ja_jp", "ko_kr"),
    ]:
        metrics.update([prediction])
    output = metrics.get_metrics()["pass@1"]
    assert output["accuracy_micro"] == 50.0
    assert output["accuracy_macro"] == 50.0
    assert output["accuracy::en_us"] == 50.0
    assert output["accuracy::fr_fr"] == 100.0
    assert output["accuracy::ja_jp"] == 0.0
    assert output["accuracy_group::western_european_we"] == 66.67
    assert output["accuracy_group_macro::western_european_we"] == 75.0
    assert output["accuracy_group::chinese_japanase_korean_cjk"] == 0.0


def test_fleurs_lid_registration():
    assert is_evaluator_registered("fleurs_lid")
    assert isinstance(get_metrics("fleurs_lid", max_k=1), FleursLIDMetrics)
