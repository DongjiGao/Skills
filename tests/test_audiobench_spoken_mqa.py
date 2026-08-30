# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

from importlib import import_module

import pytest

from nemo_skills.dataset.audiobench.prepare import (
    AUDIOBENCH_OFFICIAL_REVISION,
    SPOKEN_MQA_DATASETS,
    SPOKEN_MQA_SOURCE_REVISION,
    create_manifest_entry,
    make_spoken_mqa_manifest_entry,
)
from nemo_skills.evaluation.evaluator.audio import (
    AudioEvaluatorConfig,
    clean_spoken_mqa_prediction,
    evaluate_sample,
    evaluate_spoken_mqa,
)
from nemo_skills.evaluation.metrics.audio_metrics import AudioMetrics


def _raw_sample(answer: list[str] | None = None) -> dict:
    return {
        "context": {"array": [0.0, 0.0], "sampling_rate": 16000},
        "context_transcript": "A context transcript.",
        "instruction": {"text": "What is the answer?"},
        "answer": {"text": answer or ["42"], "audio": None},
    }


def test_source_and_row_contracts_match_pinned_official_inputs():
    assert AUDIOBENCH_OFFICIAL_REVISION == "85dcdb88c41e6ebb2bcb3ce19b15d061dfef9dba"
    assert SPOKEN_MQA_SOURCE_REVISION == "a1f9ec6143c6ce12966d7c6b3283905d679645c9"
    assert {name: spec["rows"] for name, spec in SPOKEN_MQA_DATASETS.items()} == {
        "spoken-mqa_short_digit": 100,
        "spoken-mqa_long_digit": 173,
        "spoken-mqa_single_step_reasoning": 594,
        "spoken-mqa_multi_step_reasoning": 1402,
    }


def test_manifest_preserves_reference_list_and_official_protocol():
    entry = create_manifest_entry(
        sample=_raw_sample(["42", "42.0"]),
        audio_filename="sample.wav",
        duration=1.25,
        dataset_name="spoken-mqa_short_digit",
        sample_id=7,
        category="nonjudge",
    )

    assert entry["expected_answer"] == ["42", "42.0"]
    assert entry["task_type"] == "MathQA"
    assert entry["spoken_mqa_protocol"] == "arithmetic"
    assert entry["source_provenance"]["dataset_revision"] == SPOKEN_MQA_SOURCE_REVISION
    assert entry["messages"][1]["content"] == "What is the answer?"

    leaf = make_spoken_mqa_manifest_entry(entry)
    expected_path = "../../nonjudge/audio/spoken-mqa_short_digit/sample.wav"
    assert leaf["audio_path"] == [expected_path]
    assert leaf["messages"][1]["audio"]["path"] == expected_path
    assert leaf["messages"][1]["audios"][0]["path"] == expected_path


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        ("The intermediate values are 12 and 3.", "3"),
        ("The answer is: 1,200.", "1200"),
        ("The answer is 1/2", "0.5"),
        ("The answer is: 4\n\nThe answer is: 9", "4"),
        ("No numeric answer", ""),
    ],
)
def test_prediction_cleaning_matches_official_gsm8k_extraction(prediction, expected):
    assert clean_spoken_mqa_prediction(prediction) == expected


def test_arithmetic_and_reasoning_reference_normalization_match_official_split():
    arithmetic = evaluate_spoken_mqa(["1200.0"], "The answer is 1200", "arithmetic")
    reasoning = evaluate_spoken_mqa(["work #### 1,200"], "The answer is 1200", "reasoning")

    # The pinned upstream scorer normalizes reasoning references but leaves the
    # digit-task references untouched.
    assert arithmetic["is_correct"] is False
    assert reasoning["is_correct"] is True
    assert reasoning["spoken_mqa_references"] == ["1200"]
    assert reasoning["requires_judge"] is False


def test_audio_evaluator_scores_structured_mathqa_without_a_judge():
    result = evaluate_sample(
        {
            "task_type": "MathQA",
            "spoken_mqa_protocol": "reasoning",
            "expected_answer": ["reasoning #### 21"],
            "generation": "The answer is: 21.",
        },
        AudioEvaluatorConfig(),
    )

    assert result["is_correct"] is True
    assert result["predicted_answer"] == "21"
    assert result["requires_judge"] is False


def test_audio_metrics_emit_official_accuracy_name_and_compatibility_alias():
    metrics = AudioMetrics(max_k=1)
    metrics.update(
        [
            {
                "task_type": "MathQA",
                "category": "closed",
                "generation": "21",
                "is_correct": True,
                "num_generated_tokens": 1,
            }
        ]
    )

    pass_at_1 = metrics.get_metrics()["pass@1"]
    assert pass_at_1["accuracy"] == 100.0
    assert pass_at_1["success_rate"] == 100.0
    assert "accuracy" in metrics.metrics_to_print()


def test_group_and_all_four_leaf_modules_are_discoverable():
    group = import_module("nemo_skills.dataset.audiobench.spoken-mqa")
    leaves = {
        "audiobench.spoken-mqa.short_digit",
        "audiobench.spoken-mqa.long_digit",
        "audiobench.spoken-mqa.single_step_reasoning",
        "audiobench.spoken-mqa.multi_step_reasoning",
    }
    assert set(group.BENCHMARKS) == leaves
    for leaf in leaves:
        module = import_module(f"nemo_skills.dataset.{leaf}")
        assert module.METRICS_TYPE == "audio"
        assert module.EVAL_SPLIT == "test"
        assert "enable_audio=true" in module.GENERATION_ARGS


def test_group_score_is_explicitly_entry_weighted():
    scorer = import_module("nemo_skills.dataset.audiobench.spoken-mqa.spoken_mqa_score")
    combined = {
        "audiobench.spoken-mqa.short_digit": {"pass@1": {"accuracy": 100.0, "success_rate": 100.0, "num_entries": 1}},
        "audiobench.spoken-mqa.long_digit": {"pass@1": {"accuracy": 0.0, "success_rate": 0.0, "num_entries": 3}},
    }

    score = scorer.compute_score(combined)["pass@1"]
    assert score["accuracy"] == 25.0
    assert score["num_entries"] == 4
