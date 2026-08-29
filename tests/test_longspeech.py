# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

import copy
import importlib

import pytest

from nemo_skills.evaluation.evaluator.longspeech import (
    evaluate_longspeech_sample,
    evaluate_summary_sample,
    parse_language_answer,
    parse_numeric_answer,
)
from nemo_skills.evaluation.metrics.longspeech_metrics import LongSpeechMetrics

prepare = importlib.import_module("nemo_skills.dataset.longspeech.prepare")
longspeech = importlib.import_module("nemo_skills.dataset.longspeech")


EXPECTED_REVISION = "bb50d8c11fe9aa9fc5e899a0d941e3b40acbc740"
EXPECTED_LEAVES = {
    "longspeech.asr",
    "longspeech.translation",
    "longspeech.summary",
    "longspeech.content-separation",
    "longspeech.speaker-count",
    "longspeech.language-detection",
}


def _prediction(task, reference, generation, language="en"):
    source = {
        "longspeech_task": task,
        "expected_answer": reference,
        "generation": generation,
        "language": language,
    }
    return {**source, **evaluate_longspeech_sample(source)}


def _metrics(*predictions):
    calculator = LongSpeechMetrics()
    for prediction in predictions:
        calculator.update([prediction])
    return calculator.get_metrics()["pass@1"]


def test_public_protocol_is_pinned_and_exposes_only_reproducible_tasks():
    assert prepare.REPO_ID == "ATH-MaaS/Marco_Longspeech"
    assert prepare.REVISION == EXPECTED_REVISION
    assert set(longspeech.BENCHMARKS) == EXPECTED_LEAVES
    assert set(prepare.SUPPORTED_TASKS) == {leaf.removeprefix("longspeech.") for leaf in EXPECTED_LEAVES}
    assert "temporal-qa" in prepare.UNSUPPORTED_TASKS
    assert "GPT-4-Turbo" in prepare.UNSUPPORTED_TASKS["temporal-qa"]
    assert "emotion" in prepare.UNSUPPORTED_TASKS
    assert "Joyful" in prepare.UNSUPPORTED_TASKS["emotion"]


@pytest.mark.parametrize("task", ["temporal-qa", "emotion"])
def test_incompletely_specified_tasks_fail_explicitly(task):
    with pytest.raises(ValueError, match="Unsupported LongSpeech"):
        prepare._parse_tasks(task)
    with pytest.raises(ValueError, match="intentionally unavailable"):
        evaluate_longspeech_sample({"longspeech_task": task, "expected_answer": "x", "generation": "x"})


def test_prepare_preserves_exact_stored_messages_but_sends_audio_only_user_turn():
    source = {
        "language": "en",
        "task": "summary",
        "messages": [
            {
                "role": "user",
                "audio": "LongSpeech_p1/wavs/part_00/example.wav",
                "content": "Please analyze the audio and produce a summary.",
            },
            {"role": "assistant", "content": "The exact stored reference."},
        ],
    }
    untouched = copy.deepcopy(source)

    row = prepare._format_row(
        source,
        benchmark_task="summary",
        source_task="summary",
        row_index=7,
        audio_prefix="/data/longspeech/audio",
    )

    assert source == untouched
    assert row["source_messages"] == source["messages"]
    assert row["messages"] == [
        {
            "role": "user",
            "audio": {"path": "/data/longspeech/audio/LongSpeech_p1/wavs/part_00/example.wav"},
            "content": "Please analyze the audio and produce a summary.",
        }
    ]
    assert row["expected_answer"] == "The exact stored reference."
    assert row["source_revision"] == EXPECTED_REVISION
    assert "text" not in row


@pytest.mark.parametrize(
    "leaf",
    sorted(EXPECTED_LEAVES),
)
def test_each_supported_leaf_uses_path_registered_evaluator_and_metrics(leaf):
    module = importlib.import_module(f"nemo_skills.dataset.{leaf}")
    assert module.REQUIRES_DATA_DIR is True
    assert module.METRICS_TYPE.endswith("longspeech_metrics::LongSpeechMetrics")
    assert "longspeech::LongSpeechEvaluator" in module.EVAL_ARGS
    assert "++enable_audio=true" in module.GENERATION_ARGS


def test_asr_reports_official_non_cjk_wer_cjk_cer_and_overall_cer():
    english = _prediction("asr", "hello world", "hello brave world", language="en")
    chinese = _prediction("asr", "你好", "你", language="zh-CN")

    metrics = _metrics(english, chinese)

    assert metrics["non_cjk_wer"] == 50.0
    assert metrics["cjk_cer"] == 50.0
    assert metrics["overall_cer"] == 50.0
    assert metrics["num_non_cjk"] == 1
    assert metrics["num_cjk"] == 1


def test_translation_uses_case_insensitive_corpus_bleu4():
    reference = "This is a sufficiently long reference sentence."
    prediction = _prediction("translation", reference, reference.upper())

    assert _metrics(prediction)["bleu"] == 100.0


def test_summary_reports_rouge_1_2_and_l_f1():
    reference = "one two three four"
    scores = evaluate_summary_sample(reference, reference.upper())
    assert scores == {"rouge1_f1": 1.0, "rouge2_f1": 1.0, "rougeL_f1": 1.0}

    metrics = _metrics(_prediction("summary", reference, reference.upper()))
    assert metrics["rouge1_f1"] == 100.0
    assert metrics["rouge2_f1"] == 100.0
    assert metrics["rougeL_f1"] == 100.0


def test_numeric_parser_requires_one_unambiguous_value():
    assert parse_numeric_answer("There are two distinct speakers.") == 2
    assert parse_numeric_answer("2 speakers") == 2
    assert parse_numeric_answer("one or two") is None
    assert parse_numeric_answer("several") is None


@pytest.mark.parametrize("task", ["content-separation", "speaker-count"])
def test_fixed_answer_tasks_report_all_four_official_metrics(task):
    metrics = _metrics(
        _prediction(task, 2, "There are two."),
        _prediction(task, 2, "3"),
        _prediction(task, 2, "several"),
    )

    assert metrics["numeric_accuracy"] == 33.33
    assert metrics["parsability_rate"] == 66.67
    assert metrics["post_parsing_precision"] == 50.0
    assert metrics["misunderstanding_rate"] == 33.33


def test_language_detection_partitions_correct_misunderstanding_and_error():
    assert parse_language_answer("The language is English.") == "English"
    assert parse_language_answer("en") == "English"
    assert parse_language_answer("English or German") is None

    metrics = _metrics(
        _prediction("language-detection", "English", "The language is English."),
        _prediction("language-detection", "English", "German"),
        _prediction("language-detection", "English", "I cannot tell"),
    )

    assert metrics["detection_accuracy"] == 33.33
    assert metrics["misunderstanding_rate"] == 33.33
    assert metrics["detection_error_rate"] == 33.33
    assert metrics["detection_errors"] == 1
    assert sum(
        metrics[key]
        for key in (
            "detection_accuracy",
            "misunderstanding_rate",
            "detection_error_rate",
        )
    ) == pytest.approx(100.0, abs=0.02)
