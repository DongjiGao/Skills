# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

import copy
import hashlib
import importlib
import json

import pytest
import soundfile as sf

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

    assert {task: contract["rows"] for task, contract in prepare.SOURCE_CONTRACTS.items()} == {
        "asr": 15274,
        "translation": 6309,
        "summary": 937,
        "content-separation": 1263,
        "speaker-count": 1263,
        "language-detection": 3170,
    }
    assert {
        task: contract.get("prepared_rows", contract["rows"]) for task, contract in prepare.SOURCE_CONTRACTS.items()
    } == {
        "asr": 15273,
        "translation": 6309,
        "summary": 937,
        "content-separation": 1263,
        "speaker-count": 1263,
        "language-detection": 3170,
    }
    assert sum(contract["rows"] for contract in prepare.SOURCE_CONTRACTS.values()) == 28216
    assert (
        sum(contract.get("prepared_rows", contract["rows"]) for contract in prepare.SOURCE_CONTRACTS.values()) == 28215
    )
    assert prepare.SOURCE_MISSING_AUDIO == {"asr": {"LongSpeech_p1/wavs/part_02/020934.wav": (12375,)}}
    assert all(len(contract["sha256"]) == 64 for contract in prepare.SOURCE_CONTRACTS.values())


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
        audio_duration=12.5,
    )

    assert source == untouched
    assert row["source_messages"] == source["messages"]
    assert row["messages"] == [
        {
            "role": "user",
            "audio": {
                "path": "/data/longspeech/audio/LongSpeech_p1/wavs/part_00/example.wav",
                "duration": 12.5,
            },
            "content": "Please analyze the audio and produce a summary.",
        }
    ]
    assert row["expected_answer"] == "The exact stored reference."
    assert row["source_revision"] == EXPECTED_REVISION
    assert "text" not in row


@pytest.mark.parametrize("audio", ["/absolute.wav", "../escape.wav", "dir/../../escape.wav", r"dir\\audio.wav"])
def test_prepare_rejects_nonportable_or_escaping_audio_paths(audio):
    source = {
        "messages": [
            {"role": "user", "audio": audio, "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]
    }
    with pytest.raises(ValueError, match="audio path"):
        prepare._validate_source_row(source, "summary")


def test_prepare_reads_audio_duration_and_rejects_subframe_audio(tmp_path):
    audio_dir = tmp_path / "audio"
    valid_relpath = "LongSpeech_p1/wavs/valid.wav"
    short_relpath = "LongSpeech_p1/wavs/short.wav"
    valid_path = audio_dir / valid_relpath
    short_path = audio_dir / short_relpath
    valid_path.parent.mkdir(parents=True)
    sf.write(valid_path, [0.0] * 320, 16000)
    sf.write(short_path, [0.0] * 160, 16000)

    durations = prepare._read_audio_durations([valid_relpath], audio_dir)
    assert durations == {valid_relpath: pytest.approx(0.02)}

    with pytest.raises(ValueError, match="shorter than the 0.020s feature minimum"):
        prepare._read_audio_durations([short_relpath], audio_dir)
    with pytest.raises(ValueError, match="missing or empty"):
        prepare._read_audio_durations(["LongSpeech_p1/wavs/missing.wav"], audio_dir)


def test_source_contract_rejects_hash_and_row_drift(tmp_path, monkeypatch):
    path = tmp_path / "test.jsonl"
    rows = [{"messages": []}]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setitem(
        prepare.SOURCE_CONTRACTS,
        "summary",
        {"rows": 1, "unique_audio": 0, "sha256": digest},
    )
    assert prepare._validate_source_contract("summary", path, rows) == digest
    prepare.SOURCE_CONTRACTS["summary"]["rows"] = 2
    with pytest.raises(ValueError, match="row-count drift"):
        prepare._validate_source_contract("summary", path, rows)
    prepare.SOURCE_CONTRACTS["summary"].update(rows=1, sha256="0" * 64)
    with pytest.raises(ValueError, match="source hash drift"):
        prepare._validate_source_contract("summary", path, rows)


def test_audio_complete_selection_skips_only_pinned_missing_object(monkeypatch):
    rows = [
        {
            "messages": [
                {"role": "user", "audio": "present.wav", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ]
        },
        {
            "messages": [
                {"role": "user", "audio": "missing.wav", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ]
        },
    ]
    monkeypatch.setitem(
        prepare.SOURCE_CONTRACTS,
        "summary",
        {"rows": 2, "prepared_rows": 1, "unique_audio": 1, "sha256": "0" * 64},
    )
    monkeypatch.setitem(prepare.SOURCE_MISSING_AUDIO, "summary", {"missing.wav": (1,)})

    selected = prepare._select_audio_complete_rows("summary", "summary", rows)

    assert selected == [(0, rows[0])]
    with pytest.raises(ValueError, match="missing-audio contract drift"):
        prepare._select_audio_complete_rows("summary", "summary", list(reversed(rows)))


def test_audio_complete_selection_rejects_unexpected_prepared_count(monkeypatch):
    rows = [
        {
            "messages": [
                {"role": "user", "audio": "present.wav", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ]
        }
    ]
    monkeypatch.setitem(
        prepare.SOURCE_CONTRACTS,
        "summary",
        {"rows": 1, "prepared_rows": 2, "unique_audio": 1, "sha256": "0" * 64},
    )
    with pytest.raises(ValueError, match="prepared row-count drift"):
        prepare._select_audio_complete_rows("summary", "summary", rows)


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
