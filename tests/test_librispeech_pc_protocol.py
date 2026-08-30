# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden protocol tests for LibriSpeech-PC scoring and preparation."""

import pytest

from nemo_skills.evaluation.evaluator.audio import evaluate_asr_pc, punctuation_error_counts
from nemo_skills.evaluation.metrics.audio_metrics import AudioMetrics


@pytest.mark.parametrize(
    ("reference", "hypothesis", "expected"),
    [
        (
            "Hello, world.",
            "Hello world,.",
            {"correct": 1, "substitutions": 0, "deletions": 1, "insertions": 1},
        ),
        (
            "Let's eat, Bob.",
            "Lets eat, Bob.",
            {"correct": 2, "substitutions": 0, "deletions": 0, "insertions": 0},
        ),
        (
            "Let's eat, Bob!",
            "Let's eat Bob!",
            {"correct": 0, "substitutions": 0, "deletions": 1, "insertions": 0},
        ),
        (
            "Hi, Bob.",
            "Hi. Bob?",
            {"correct": 0, "substitutions": 2, "deletions": 0, "insertions": 0},
        ),
        (
            "Hi Bob",
            "Hi, Bob.",
            {"correct": 0, "substitutions": 0, "deletions": 0, "insertions": 2},
        ),
        (
            "Let's shout!",
            "Lets shout!",
            {"correct": 0, "substitutions": 0, "deletions": 0, "insertions": 0},
        ),
    ],
)
def test_per_matches_nemo_masked_position_golden_counts(reference, hypothesis, expected):
    """Goldens are from NeMo punct_er.py at bcc9ee94f1d96ca72e5273d5f7085e9925d662b5."""
    result = punctuation_error_counts(reference, hypothesis)
    actual = {name: result[f"per_{name}"] for name in expected}
    assert actual == expected
    expected_operations = sum(expected.values())
    expected_errors = expected_operations - expected["correct"]
    assert result["per_operations"] == expected_operations
    assert result["per_errors"] == expected_errors
    assert result["per"] == pytest.approx(expected_errors / expected_operations if expected_operations else 0.0)


def test_asr_pc_uses_dedicated_case_and_punctuation_tokens_without_tn_itn():
    numeric = evaluate_asr_pc("I paid two dollars.", "I paid 2 dollars.")
    assert numeric["wer"] == pytest.approx(1 / 4)
    assert numeric["wer_c"] == pytest.approx(1 / 4)
    assert numeric["wer_pc"] == pytest.approx(1 / 5)
    assert numeric["wer_errors"] == numeric["wer_c_errors"] == numeric["wer_pc_errors"] == 1

    case = evaluate_asr_pc("Hello, World.", "hello, world.")
    assert case["wer"] == 0
    assert case["wer_c"] == 1
    assert case["wer_pc"] == pytest.approx(1 / 2)
    assert case["per"] == 0

    apostrophe = evaluate_asr_pc("Let's go.", "Lets go.")
    assert apostrophe["wer"] == pytest.approx(1 / 2)
    assert apostrophe["wer_c"] == pytest.approx(1 / 2)
    assert apostrophe["per"] == 0


def test_asr_pc_metrics_are_corpus_micro_with_raw_counts():
    predictions = [
        evaluate_asr_pc("A.", "a"),
        evaluate_asr_pc(
            "One, two, three, four, five, six, seven, eight, nine.",
            "One, two, three, four, five, six, seven, eight, nine.",
        ),
    ]
    metrics = AudioMetrics(compute_no_answer=False)
    for prediction in predictions:
        prediction["generation"] = prediction["pred_text"]
        metrics.update([prediction])

    result = metrics.get_metrics()["pass@1"]
    assert result["wer_c"] == 10.0
    assert result["wer_c_errors"] == 1
    assert result["wer_c_ref_words"] == 10
    assert result["wer_pc"] == 10.0
    assert result["wer_pc_errors"] == 2
    assert result["wer_pc_ref_words"] == 20
    assert result["per"] == 10.0
    assert result["per_correct"] == 9
    assert result["per_deletions"] == 1
    assert result["per_errors"] == 1
    assert result["per_operations"] == 10
    assert result["wer_c_macro"] == result["wer_pc_macro"] == result["per_macro"] == 50.0
