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
pytest.importorskip("kaldialign")
pytest.importorskip("whisper_normalizer")

from nemo_skills.evaluation.evaluator.audio import evaluate_asr, preprocess_asr_text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("T V ah health care", "tv health care"),
        ("wi fi and e mail", "wifi and email"),
        ("Jane's book", "jane s book"),
        ("Katherine and Shaun said OK", "catherine and sean said okay"),
        ("uh um ah er erm", "er erm"),
    ],
)
def test_hf_leaderboard_normalizer_matches_pr142_examples(text, expected):
    assert preprocess_asr_text(text, mode="hf_leaderboard") == expected


@pytest.mark.parametrize(
    ("reference", "hypothesis", "reference_words"),
    [
        ("white paper", "whitepaper", 2),
        ("whitepaper", "white paper", 1),
        ("a health care plan", "a healthcare plan", 4),
    ],
)
def test_hf_leaderboard_kaldialign_merges_compounds(reference, hypothesis, reference_words):
    result = evaluate_asr(reference, hypothesis, normalization_mode="hf_leaderboard")

    assert result["wer"] == 0
    assert result["wer_errors"] == 0
    assert result["wer_ref_words"] == reference_words
    assert result["wer_substitutions"] == 0
    assert result["wer_insertions"] == 0
    assert result["wer_deletions"] == 0


def test_hf_leaderboard_empty_hypothesis_uses_kaldialign_counts():
    result = evaluate_asr("two words", "", normalization_mode="hf_leaderboard")

    assert result["wer"] == 1
    assert result["wer_errors"] == 2
    assert result["wer_ref_words"] == 2
    assert result["wer_deletions"] == 2
    assert result["pred_text"] == ""


def test_standard_mode_keeps_original_whisper_normalizer():
    assert preprocess_asr_text("Jane's book", mode="standard") == "jane is book"
