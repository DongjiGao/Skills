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

from importlib import import_module

import pytest

pytest.importorskip("jiwer")
pytest.importorskip("kaldialign")
pytest.importorskip("num2words")
pytest.importorskip("whisper_normalizer")

from nemo_skills.evaluation.evaluator.audio import AudioEvaluatorConfig, evaluate_sample
from nemo_skills.pipeline.summarize_results import _set_asr_leaderboard_multilingual_macro_wer


prepare = import_module("nemo_skills.dataset.asr-leaderboard-multilingual.prepare")


def test_multilingual_leaderboard_config_matrix_matches_official_public_set():
    assert set(prepare.DATASET_CONFIGS) == {
        "fleurs_de",
        "fleurs_fr",
        "fleurs_it",
        "fleurs_es",
        "fleurs_pt",
        "mcv_de",
        "mcv_es",
        "mcv_fr",
        "mcv_it",
        "mls_es",
        "mls_fr",
        "mls_it",
        "mls_pt",
    }
    assert prepare.LANGUAGE_DATASETS == {
        "de": ("fleurs_de", "mcv_de"),
        "fr": ("fleurs_fr", "mcv_fr", "mls_fr"),
        "it": ("fleurs_it", "mcv_it", "mls_it"),
        "es": ("fleurs_es", "mcv_es", "mls_es"),
        "pt": ("fleurs_pt", "mls_pt"),
    }


def test_open_asr_multilingual_sample_preserves_diacritics():
    config = AudioEvaluatorConfig(normalization_mode="multilingual")
    sample = {
        "task_type": "Multilingual-ASR",
        "expected_answer": "ete",
        "generation": "été",
        "extra_fields": {"src_lang": "fr", "remove_diacritics": False, "merge_compounds": True},
    }

    result = evaluate_sample(sample, config)

    assert result["text"] == "ete"
    assert result["pred_text"] == "été"
    assert result["wer"] == 1


def test_fleurs_default_multilingual_sample_still_strips_diacritics():
    config = AudioEvaluatorConfig(normalization_mode="multilingual")
    sample = {
        "task_type": "Multilingual-ASR",
        "expected_answer": "ete",
        "generation": "été",
        "extra_fields": {"src_lang": "fr"},
    }

    result = evaluate_sample(sample, config)

    assert result["text"] == "ete"
    assert result["pred_text"] == "ete"
    assert result["wer"] == 0


def test_open_asr_multilingual_sample_uses_kaldialign_compound_merging():
    config = AudioEvaluatorConfig(normalization_mode="multilingual")
    sample = {
        "task_type": "Multilingual-ASR",
        "expected_answer": "white paper",
        "generation": "whitepaper",
        "extra_fields": {
            "src_lang": "de",
            "remove_diacritics": False,
            "normalize_compound": True,
            "merge_compounds": True,
        },
    }

    result = evaluate_sample(sample, config)

    assert result["wer"] == 0
    assert result["wer_errors"] == 0


def test_multilingual_leaderboard_summary_adds_language_and_overall_means():
    metrics = {
        "_all_": {"pass@1": {"wer": 999.0, "num_entries": 60}},
        "fleurs_de": {
            "pass@1": {
                "wer": 10.0,
                "avg_tokens": 1,
                "gen_seconds": 2,
                "success_rate": 100.0,
                "no_answer": 0.0,
                "num_entries": 10,
                "substitutions": 1,
                "insertions": 2,
                "deletions": 3,
                "ref_words": 10,
            }
        },
        "mcv_de": {
            "pass@1": {
                "wer": 20.0,
                "avg_tokens": 3,
                "gen_seconds": 4,
                "success_rate": 50.0,
                "no_answer": 10.0,
                "num_entries": 30,
                "substitutions": 4,
                "insertions": 5,
                "deletions": 6,
                "ref_words": 30,
            }
        },
        "fleurs_fr": {"pass@1": {"wer": 30.0, "num_entries": 10}},
        "mcv_fr": {"pass@1": {"wer": 60.0, "num_entries": 10}},
        "mls_fr": {"pass@1": {"wer": 90.0, "num_entries": 10}},
    }

    _set_asr_leaderboard_multilingual_macro_wer(metrics)

    assert metrics["de"]["pass@1"]["wer"] == 15.0
    assert metrics["de"]["pass@1"]["avg_tokens"] == 2
    assert metrics["de"]["pass@1"]["gen_seconds"] == 6
    assert metrics["de"]["pass@1"]["success_rate"] == 62.5
    assert metrics["de"]["pass@1"]["no_answer"] == 7.5
    assert metrics["de"]["pass@1"]["substitutions"] == 5
    assert metrics["de"]["pass@1"]["ref_words"] == 40
    assert metrics["fr"]["pass@1"]["wer"] == 60.0
    assert metrics["_all_"]["pass@1"]["wer"] == 37.5
