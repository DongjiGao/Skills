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

from nemo_skills.evaluation.evaluator import get_evaluator_class
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
        "fleurs_nl",
        "fleurs_zh",
        "fleurs_ja",
        "mcv_de",
        "mcv_es",
        "mcv_fr",
        "mcv_it",
        "mcv_nl",
        "mls_es",
        "mls_fr",
        "mls_it",
        "mls_pt",
        "mls_nl",
        "monsoon_hi",
    }
    assert {
        spec.revision
        for name, spec in prepare.DATASET_CONFIGS.items()
        if name not in {"monsoon_hi", "fleurs_zh", "fleurs_ja"}
    } == {prepare.LEADERBOARD_REVISION}
    assert prepare.DATASET_CONFIGS["monsoon_hi"].revision == prepare.MONSOON_HINDI_CONTRACT.revision
    assert prepare.LANGUAGE_DATASETS == {
        "de": ("fleurs_de", "mcv_de"),
        "fr": ("fleurs_fr", "mcv_fr", "mls_fr"),
        "it": ("fleurs_it", "mcv_it", "mls_it"),
        "es": ("fleurs_es", "mcv_es", "mls_es"),
        "pt": ("fleurs_pt", "mls_pt"),
        "nl": ("fleurs_nl", "mcv_nl", "mls_nl"),
        "hi": ("monsoon_hi",),
        "zh": ("fleurs_zh",),
        "ja": ("fleurs_ja",),
    }


def test_hindi_monsoon_format_preserves_lattice_and_pinned_provenance(tmp_path):
    spec = prepare.DATASET_CONFIGS["monsoon_hi"]
    row = {
        "id": "clip-1",
        "lattice": [["नमस्ते"], ["दुनिया", "विश्व"]],
        "speaker_id": 7,
        "audio_length_s": 1.25,
        "audio": None,
    }

    result = prepare.format_entry(row, 0, "monsoon_hi", tmp_path, spec, with_audio=False)

    assert result["expected_answer"] == "नमस्ते दुनिया"
    assert result["reference_lattice"] == row["lattice"]
    assert result["monsoon_language"] == "hi-in"
    assert result["subset_for_metrics"] == "monsoon_hi"
    assert result["messages"][1]["audio"] == {
        "path": "/dataset/asr-leaderboard-multilingual/data/monsoon_hi/clip-1.wav",
        "duration": 1.25,
    }
    assert result["extra_fields"]["source_repo"] == spec.repo
    assert result["extra_fields"]["source_revision"] == spec.revision


def test_multilingual_profile_registers_mixed_wer_oiwer_evaluator():
    benchmark = import_module("nemo_skills.dataset.asr-leaderboard-multilingual")
    assert "OpenASRMultilingualEvaluator" in benchmark.EVAL_ARGS
    evaluator = get_evaluator_class(
        "nemo_skills.dataset.asr-leaderboard-multilingual.scorer::OpenASRMultilingualEvaluator",
        {"normalization_mode": "multilingual"},
    )
    assert evaluator.eval_config.normalization_mode == "multilingual"


def test_hindi_monsoon_row_uses_official_lattice_oiwer(monkeypatch):
    scorer = import_module("nemo_skills.dataset.voicearena-monsoon.scorer")
    mixed = import_module("nemo_skills.dataset.asr-leaderboard-multilingual.scorer")
    monkeypatch.setattr(
        scorer,
        "_official_oiwer",
        lambda: (
            lambda hypothesis, reference_lists, input_language: (50.0, [], [], [], (1, 0, 0), 2, [], reference_lists)
        ),
    )
    evaluator = mixed.OpenASRMultilingualEvaluator({"normalization_mode": "multilingual"})
    result = __import__("asyncio").run(
        evaluator.eval_single(
            {
                "monsoon_language": "hi-in",
                "reference_lattice": [["नमस्ते"], ["दुनिया", "विश्व"]],
                "expected_answer": "नमस्ते दुनिया",
                "generation": "नमस्ते",
            }
        )
    )
    assert result["monsoon_metric"] == "oiwer"
    assert result["wer"] == 0.5
    assert result["wer_errors"] == 1
    assert result["wer_ref_words"] == 2


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


@pytest.mark.parametrize("src_lang", ["cmn_hans_cn", "zh-CN"])
def test_configured_chinese_cer_ignores_only_han_adjacent_whitespace(src_lang):
    config = AudioEvaluatorConfig(normalization_mode="multilingual")
    sample = {
        "task_type": "Multilingual-ASR",
        "expected_answer": "你 好 use AI",
        "generation": "你好 use AI",
        "extra_fields": {
            "src_lang": src_lang,
            "use_cer": True,
            "cer_whitespace_mode": "han_adjacent",
        },
    }

    result = evaluate_sample(sample, config)

    # The mixed headline stores CJK CER under the historical wer* keys, while
    # the parallel cer* detail uses the same dataset-configured character stream.
    # Han-adjacent formatting spaces disappear, but the Latin word boundary stays.
    assert result["wer"] == result["cer"] == 0
    assert result["wer_errors"] == result["cer_errors"] == 0
    assert result["text"] == result["pred_text"] == "你好use ai"


def test_non_cer_multilingual_asr_preserves_word_boundaries():
    config = AudioEvaluatorConfig(normalization_mode="multilingual")
    sample = {
        "task_type": "Multilingual-ASR",
        "expected_answer": "alpha beta",
        "generation": "alphabeta",
        "extra_fields": {"src_lang": "en", "use_cer": False},
    }

    result = evaluate_sample(sample, config)

    assert result["wer"] > 0
    assert result["text"] == "alpha beta"
    assert result["pred_text"] == "alphabeta"


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
        "fleurs_nl": {"pass@1": {"wer": 7.0, "num_entries": 10}},
        "mcv_nl": {"pass@1": {"wer": 8.0, "num_entries": 10}},
        "mls_nl": {"pass@1": {"wer": 9.0, "num_entries": 10}},
        "monsoon_hi": {"pass@1": {"wer": 12.0, "num_entries": 10}},
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
    assert metrics["nl"]["pass@1"]["wer"] == 8.0
    assert metrics["hi"]["pass@1"]["wer"] == 12.0
    assert metrics["_all_"]["pass@1"]["wer"] == 23.75


def test_extended_macro_retains_official_language_comparison():
    metrics = {
        "_all_": {"pass@1": {}},
        "fleurs_de": {"pass@1": {"wer": 6, "num_entries": 1}},
        "fleurs_zh": {"pass@1": {"wer": 3, "num_entries": 1}},
        "fleurs_ja": {"pass@1": {"wer": 9, "num_entries": 1}},
    }
    _set_asr_leaderboard_multilingual_macro_wer(metrics)
    assert metrics["_all_"]["pass@1"]["wer_official_languages"] == 6
    assert metrics["_all_"]["pass@1"]["mixed_error_rate_extended"] == 6
    assert metrics["zh"]["pass@1"]["wer"] == 3
    assert metrics["ja"]["pass@1"]["wer"] == 9


@pytest.mark.parametrize(
    ("name", "locale", "text"), [("fleurs_zh", "cmn_hans_cn", "你好世界"), ("fleurs_ja", "ja_jp", "こんにちは世界")]
)
def test_fleurs_extensions_use_pinned_test_source_and_cer(name, locale, text, tmp_path, monkeypatch):
    import json

    fleurs = import_module("nemo_skills.dataset.fleurs.prepare")
    calls = []

    def load(src_locale, split, cache_dir):
        calls.append((src_locale, split))
        return ([{"id": 7, "transcription": text, "raw_transcription": text, "wav_filename": "sample.wav",
                  "samples": [0] * 16000, "sampling_rate": 16000}], {"rows": 1})
    monkeypatch.setattr(fleurs, "_load_product_locale_source", load)
    assert prepare.prepare_fleurs_extension(name, tmp_path, with_audio=False) == 1
    record = json.loads((tmp_path / f"{name}.jsonl").read_text())
    assert calls == [(locale, "test")]
    assert record["extra_fields"]["use_cer"] is True
    assert record["extra_fields"]["source_revision"] == fleurs.HF_REVISION
    assert record["expected_answer"] == text
    assert record["id"] == f"{name}-sample"
