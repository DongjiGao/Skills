# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

import hashlib
import importlib
import json

import pytest

from nemo_skills.evaluation.evaluator import is_evaluator_registered
from nemo_skills.evaluation.evaluator.air_bench_foundation import (
    eval_air_bench_foundation,
    official_foundation_ground_truth,
    official_foundation_prediction,
)
from nemo_skills.evaluation.metrics.air_bench_foundation_metrics import AIRBenchFoundationMetrics
from nemo_skills.evaluation.metrics.compute_metrics import ComputeMetrics
from nemo_skills.evaluation.metrics.map_metrics import get_metrics

prepare = importlib.import_module("nemo_skills.dataset.air-bench-foundation-speech.prepare")
profile = importlib.import_module("nemo_skills.dataset.air-bench-foundation-speech")


def _row(
    uniq_id: int = 1,
    *,
    task_name: str = "Speech_Grounding",
    dataset_name: str = "librispeech",
    path: str = "sample.flac",
) -> dict:
    return {
        "path": path,
        "question": "Which answer is correct?",
        "choice_a": "first",
        "choice_b": "second",
        "choice_c": None,
        "choice_d": None,
        "answer_gt": "first",
        "task_name": task_name,
        "dataset_name": dataset_name,
        "uniq_id": uniq_id,
    }


def _scored(leaf: str, *, valid: bool, correct: bool) -> dict:
    return {
        "generation": "A" if valid else "unparseable",
        "official_parse_valid": valid,
        "symbolic_correct": correct,
        "air_bench_foundation_leaf": leaf,
    }


def test_public_slice_has_exact_pins_hashes_and_census():
    assert prepare.HF_REPO_ID == "qyang1021/AIR-Bench-Dataset"
    assert prepare.HF_REVISION == "8fa643aefd1d7248fd55fcf3e7ad22c81bbac1fa"
    assert prepare.FOUNDATION_METADATA_SHA256 == ("e59da5c972e30f03797517a009105428060ae34e3b6c112635db8f0bebc312fa")
    assert prepare.OFFICIAL_REPO_REVISION == "983247666e2a25519b5c122810c2bcd91f26fd8d"
    assert prepare.OFFICIAL_INFERENCE_SHA256 == ("607ece9a9a78b9994905824b9e3f74397f0ef575518d160ecfd0222a15247448")
    assert prepare.OFFICIAL_SCORER_SHA256 == ("1b6380648d0a716a2650cf258b0d76bdf25290cd52493891f163b07a0d18272d")
    assert len(prepare.SPEECH_TASKS) == 9
    assert len(prepare.EXPECTED_LEAF_COUNTS) == 11
    assert sum(prepare.EXPECTED_LEAF_COUNTS.values()) == prepare.EXPECTED_TOTAL_ROWS == 10973
    assert prepare.EXPECTED_LEAF_COUNTS["Speech_Grounding_librispeech"] == 981
    assert prepare.EXPECTED_LEAF_COUNTS["Speaker_Gender_Recognition_common_voice_13_en"] == 992
    assert "Speaker_Emotion_Recontion" in prepare.SPEECH_TASKS
    assert "Speech_Entity_Reconition" in prepare.SPEECH_TASKS
    assert profile.METRICS_TYPE == "air_bench_foundation"


def test_census_validation_is_leaf_exact_and_checks_inventory(monkeypatch):
    rows = [_row()]
    expected_leaf = "Speech_Grounding_librispeech"
    monkeypatch.setattr(prepare, "EXPECTED_LEAF_COUNTS", {expected_leaf: 1})
    monkeypatch.setattr(prepare, "EXPECTED_TOTAL_ROWS", 1)
    monkeypatch.setattr(prepare, "EXPECTED_AUDIO_INVENTORY_SHA256", prepare._inventory_sha256(rows))
    prepare.validate_pinned_speech_census(rows)

    monkeypatch.setattr(prepare, "EXPECTED_AUDIO_INVENTORY_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="audio-inventory drift"):
        prepare.validate_pinned_speech_census(rows)


def test_exact_official_prompt_including_two_choice_none_placeholders():
    row = _row()
    assert prepare.official_prompt(row) == (
        "Choose the most suitable answer from options A, B, C, and D to respond the question in next line, "
        "you may only choose A or B or C or D.\n"
        "Which answer is correct?\n"
        "A. first\nB. second\nC. None\nD. None"
    )
    entry = prepare.format_entry(row, "/data/air-bench-foundation-speech")
    assert entry["messages"][0]["content"] == prepare.official_prompt(row)
    assert entry["messages"][0]["audio"]["path"].endswith("/Foundation/Speech_Grounding_librispeech/sample.flac")
    assert entry["subset_for_metrics"] == "Speech_Grounding_librispeech"


@pytest.mark.parametrize(
    "response,expected",
    [
        ("A.", "A"),
        ("  B because it is correct", "B"),
        ("The answer is C.", "C"),
        ("The answer is D.\n", "D"),
        ("A text ending in D.", "A"),
        ("The answer is B", None),
        ("None", None),
        ("", None),
        ("unparseable", None),
    ],
)
def test_official_parser_semantics(response, expected):
    assert official_foundation_prediction(response) == expected


def test_ground_truth_uses_official_first_matching_choice_precedence():
    row = _row()
    row["choice_b"] = row["choice_a"]
    assert official_foundation_ground_truth(row) == "A"
    row["answer_gt"] = "missing"
    with pytest.raises(ValueError, match="does not match"):
        official_foundation_ground_truth(row)


def test_batch_evaluator_preserves_official_response_and_parse_failure(tmp_path):
    output_path = tmp_path / "results.jsonl"
    valid = {**_row(1), "generation": "The answer is A."}
    invalid = {**_row(2), "generation": "The answer is A"}
    output_path.write_text("\n".join(json.dumps(row) for row in (valid, invalid)) + "\n", encoding="utf-8")

    eval_air_bench_foundation({"input_file": str(output_path)})
    results = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert results[0]["response"] == valid["generation"]
    assert results[0]["predicted_answer"] == "A"
    assert results[0]["symbolic_correct"] is True
    assert results[1]["predicted_answer"] is None
    assert results[1]["official_parse_valid"] is False
    assert results[1]["match"] == 0


def test_official_leaf_accuracy_excludes_parse_failures():
    metrics = AIRBenchFoundationMetrics(max_k=1)
    leaf = "Speech_Grounding_librispeech"
    for prediction in (
        _scored(leaf, valid=True, correct=True),
        _scored(leaf, valid=True, correct=False),
        _scored(leaf, valid=False, correct=False),
    ):
        metrics.update([prediction])
    output = metrics.get_metrics()["pass@1"]
    assert output["num_entries"] == 3
    assert output["total_sum"] == 2
    assert output["correct_num"] == 1
    assert output["fail_num"] == 1
    assert output["accuracy"] == 50.0


def test_mixed_all_calculator_does_not_invent_an_aggregate_accuracy():
    metrics = AIRBenchFoundationMetrics(max_k=1)
    metrics.update([_scored("Speech_Grounding_librispeech", valid=True, correct=True)])
    metrics.update([_scored("Speaker_Emotion_Recontion_meld", valid=True, correct=False)])
    output = metrics.get_metrics()["pass@1"]
    assert output["num_leaves"] == 2
    assert output["total_sum"] == 2
    assert "accuracy" not in output


def test_compute_metrics_preserves_all_eleven_official_leaf_accuracies(tmp_path):
    result_path = tmp_path / "results.jsonl"
    rows = [
        {
            "generation": "A",
            "official_parse_valid": True,
            "symbolic_correct": True,
            "air_bench_foundation_leaf": leaf,
            "subset_for_metrics": leaf,
        }
        for leaf in prepare.EXPECTED_LEAF_COUNTS
    ]
    result_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    output = ComputeMetrics(
        benchmark="air-bench-foundation-speech", metric_type="air_bench_foundation"
    ).compute_metrics([str(result_path)])
    assert set(output) == {"_all_", *prepare.EXPECTED_LEAF_COUNTS}
    assert output["_all_"]["pass@1"]["num_leaves"] == 11
    assert "accuracy" not in output["_all_"]["pass@1"]
    for leaf in prepare.EXPECTED_LEAF_COUNTS:
        assert output[leaf]["pass@1"]["accuracy"] == 100.0


def test_prepare_publishes_atomic_manifest_and_provenance(tmp_path, monkeypatch):
    row = _row()
    audio_path = tmp_path / prepare._audio_repo_path(row)
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"not-empty")
    monkeypatch.setattr(prepare, "load_pinned_speech_rows", lambda _data_dir: [row])

    output_path = prepare.prepare_air_bench_foundation_speech(tmp_path, no_audio=True)
    manifest_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    receipt = json.loads((tmp_path / prepare.PROVENANCE_FILENAME).read_text(encoding="utf-8"))
    assert len(manifest_rows) == receipt["selected_rows"] == 1
    assert receipt["audio_mode"] == "validated-existing"
    assert receipt["manifest_sha256"] == hashlib.sha256(output_path.read_bytes()).hexdigest()
    assert receipt["scoring"].endswith("no aggregate accuracy")
    assert not list(tmp_path.glob(".*.tmp"))


def test_no_audio_fails_closed_on_missing_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "load_pinned_speech_rows", lambda _data_dir: [_row()])
    with pytest.raises(FileNotFoundError, match="missing=1"):
        prepare.prepare_air_bench_foundation_speech(tmp_path, no_audio=True)
    assert not (tmp_path / "test.jsonl").exists()


def test_evaluator_and_metrics_registration():
    assert is_evaluator_registered("air-bench-foundation")
    assert isinstance(get_metrics("air_bench_foundation", max_k=1), AIRBenchFoundationMetrics)
