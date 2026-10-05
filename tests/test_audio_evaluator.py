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

import pytest

pytest.importorskip("jiwer")
pytest.importorskip("whisper_normalizer")

from nemo_skills.evaluation.evaluator.audio import evaluate_asr, preprocess_asr_text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("kinda", "kinda"),
        ("sorta", "sorta"),
        ("dunno", "dunno"),
        ("'cause", "cause"),
        ("EBITDA's", "ebitda is"),
        ("I kinda sorta dunno 'cause EBITDA's up.", "i kinda sorta dunno cause ebitda is up"),
        ("We, um, reported 10 dollars [noise].", "we reported $ten"),
    ],
)
def test_audiobench_earnings_normalizer_matches_pinned_upstream_goldens(text, expected):
    assert preprocess_asr_text(text, mode="audiobench_earnings") == expected


def test_historical_audiobench_mode_is_not_changed_by_earnings_protocol():
    assert (
        preprocess_asr_text("I kinda sorta dunno 'cause EBITDA's up.", mode="audiobench")
        == "i kind of sort of do not know because ebitda is up"
    )


def test_audiobench_earnings_empty_generation_is_a_full_reference_error():
    result = evaluate_asr(
        "Revenue increased by twenty percent.",
        "",
        normalization_mode="audiobench_earnings",
    )

    # Soft generation failures stay in the corpus and contribute one error for
    # every normalized reference word, rather than disappearing from WER.
    assert result["wer"] == 1.0
    assert result["wer_errors"] == result["wer_ref_words"] == 4
    assert result["wer_ref_words"] == len(result["text"].split())


def test_apptek_callcenter_normalization_matches_dataset_specific_mappings():
    """End-to-end: prediction-only ``oh`` removal, reference-only ``~`` half-word
    cleanup, Whisper number normalization, and the AppTek word_mappings all line
    up so a faithful prediction scores 0 WER."""
    reference = "I called one eight hundred zero zero four nine eight nine and said he~ hello."
    hypothesis = "I called 1800 oh oh 4 989 and said hello."

    result = evaluate_asr(reference, hypothesis, normalization_mode="apptek_callcenter")

    assert result["wer"] == 0
    assert result["wer_errors"] == 0
    assert result["text"] == "i called 1800004989 and said hello"
    assert result["pred_text"] == "i called 1800004989 and said hello"


def test_apptek_callcenter_empty_reference_returns_none():
    """Empty references must yield wer=None so they are dropped at aggregation."""
    result = evaluate_asr("", "some hypothesis", normalization_mode="apptek_callcenter")

    assert result["wer"] is None
    assert result["is_correct"] is None
    assert result["text"] == ""


def test_apptek_callcenter_preprocess_forwards_prediction_flag():
    """Prediction normalization removes AppTek prediction-only filler words."""
    assert preprocess_asr_text("oh hello", mode="apptek_callcenter", is_prediction=True) == "hello"


def test_apptek_callcenter_hesitations_override_word_mappings():
    """Hesitation removals take precedence over broader word normalisations."""
    assert preprocess_asr_text("ahh hello", mode="apptek_callcenter") == "hello"


def test_resolve_bleu_tokenize_handles_locale_variants():
    from nemo_skills.evaluation.evaluator.audio import resolve_bleu_tokenize

    assert resolve_bleu_tokenize("zh-CN") == "zh"
    assert resolve_bleu_tokenize("zh_TW") == "zh"
    assert resolve_bleu_tokenize("cmn_hans_cn") == "zh"
    assert resolve_bleu_tokenize("ja-JP") == "ja-mecab"
    assert resolve_bleu_tokenize("ko_kr") == "ko-mecab"
    assert resolve_bleu_tokenize("th_th") == "flores200"
    assert resolve_bleu_tokenize("de") == "13a"
    assert resolve_bleu_tokenize(None) == "13a"
