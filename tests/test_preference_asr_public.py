# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import importlib
import json

import pytest

from nemo_skills.evaluation.evaluator import get_evaluator_class
from nemo_skills.evaluation.metrics.map_metrics import get_metrics

dataset = importlib.import_module("nemo_skills.dataset.preference-asr-public")
default = importlib.import_module("nemo_skills.dataset.preference-asr-public.default")
instructed = importlib.import_module("nemo_skills.dataset.preference-asr-public.instructed")
normalizer = importlib.import_module("nemo_skills.dataset.preference-asr-public.normalizer")
prepare = importlib.import_module("nemo_skills.dataset.preference-asr-public.prepare")
scorer = importlib.import_module("nemo_skills.dataset.preference-asr-public.scorer")
score = importlib.import_module("nemo_skills.dataset.preference-asr-public.score")


def _row(category="normalization", *, direction="itn", sub_category=None):
    if sub_category is None:
        sub_category = ["numbers"]
    row = {
        "audio_filepath": "data/ami/clip.flac",
        "duration": 1.0,
        "text": "I have twenty two apples.",
        "preference_text": "I have 22 apples.",
        "instruction": "Transcribe and write numbers as digits.",
        "preference_type": category,
        "sub_category": sub_category,
        "dataset": "ami",
    }
    if direction is not None:
        row["direction"] = direction
    return row


def _small_contract(rows):
    return prepare.ManifestContract(
        rows=len(rows),
        unique_audio=len({row["audio_filepath"] for row in rows}),
        row_duration_seconds=sum(row["duration"] for row in rows),
        empty_preference_text=sum(not row["preference_text"].strip() for row in rows),
        corpus_rows=tuple(sorted(__import__("collections").Counter(row["dataset"] for row in rows).items())),
        category_rows=tuple(sorted(__import__("collections").Counter(row["preference_type"] for row in rows).items())),
        direction_rows=tuple(
            sorted(
                __import__("collections").Counter(row.get("direction") for row in rows).items(),
                key=lambda x: str(x[0]),
            )
        ),
    )


def test_release_contract_pins_actual_public_artifact_not_earlier_paper_census():
    contract = prepare.MANIFEST_CONTRACT
    assert prepare.GITHUB_REVISION == "efa5cae9aecb56ce28aeea6105425ded654acbdd"
    assert prepare.MANIFEST_SHA256 == "134214055ef2b6f4ff88b5526072036be17a59e6a238472fd1f71d9b0c81c2d9"
    assert prepare.HF_RELEASE_REVISION == "038bd44722b25a17abc9ad53265c6a96be8195ab"
    assert prepare.ESB_REVISION == "b6bdcd0beb34f8975dc659796176d88f43aff502"
    assert prepare.VOX_REVISION == "07adf4a242e2e5dd955a230909c177573c978ee3"
    assert contract.rows == 2822
    assert contract.unique_audio == 1997
    assert contract.empty_preference_text == 5
    assert dict(contract.corpus_rows) == {"ami": 632, "common_voice": 477, "librispeech": 749, "voxpopuli": 964}
    assert dict(contract.category_rows) == {
        "case": 1113,
        "disfluencies": 516,
        "entities": 906,
        "normalization": 228,
        "standard": 59,
    }


def test_manifest_validation_fails_closed_on_counts_paths_and_reused_duration():
    rows = [_row(), {**_row(), "audio_filepath": "data/ami/other.flac"}]
    contract = _small_contract(rows)
    assert prepare._validate_manifest(rows, contract) == rows
    with pytest.raises(ValueError, match="row-count drift"):
        prepare._validate_manifest(rows[:1], contract)
    with pytest.raises(ValueError, match="path/corpus mismatch"):
        prepare._validate_manifest([{**rows[0], "audio_filepath": "data/voxpopuli/clip.flac"}, rows[1]], contract)
    duplicate = [rows[0], {**rows[0], "duration": 2.0}]
    with pytest.raises(ValueError, match="conflicting durations"):
        prepare._validate_manifest(duplicate, _small_contract(duplicate))


def test_load_manifest_requires_exact_released_sha(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps(_row()) + "\n")
    with pytest.raises(ValueError, match="manifest SHA drift"):
        prepare.load_pinned_manifest(path)


def test_default_and_instructed_records_use_request_scoped_prompts(monkeypatch):
    monkeypatch.setenv("NEMO_SKILLS_AUDIO_ROOT", "/data")
    row = _row()
    default_record = prepare._build_record(row, 1, "default")
    instructed_record = prepare._build_record(row, 1, "instructed")
    assert default_record["messages"][0]["content"] == "Transcribe the English audio into text"
    assert instructed_record["messages"][0]["content"] == row["instruction"]
    assert default_record["expected_answer"] == row["preference_text"]
    assert instructed_record["expected_answer"] == row["preference_text"]
    assert default_record["messages"][0]["audio"] == {
        "path": "/data/preference-asr-public/audio/ami/clip.wav",
        "duration": 1.0,
    }
    assert default_record["preference_prompt_variant"] == "default"
    assert instructed_record["preference_prompt_variant"] == "instructed"


def test_case_target_is_recovered_from_released_subcategory():
    row = _row(
        "case",
        direction=None,
        sub_category=["proper_punctuated_capitalized_format"],
    )
    row.update(text="Hello, World!", preference_text="Hello, World!", instruction="Use correct case and punctuation")
    record = prepare._build_record(row, 1, "instructed")
    assert record["target_case"] == "proper_punctuated_capitalized_format"
    norm = normalizer.PreferenceAwareNormalizer()
    assert norm.normalize_entry("Hello, World!", record) == "Hello, World!"
    assert norm.normalize_entry("hello world", record) == "hello world"
    with pytest.raises(ValueError, match="ambiguous target"):
        prepare._build_record({**row, "sub_category": []}, 1, "instructed")


def test_released_normalizer_delta_and_preference_number_behavior():
    standard = normalizer.standard_normalizer()
    preference = normalizer.preference_normalizer()
    entry = _row()
    assert standard("I have 22 apples") == standard("I have twenty two apples")
    assert preference.normalize_entry("I have 22 apples", entry) == "i have 22 apples"
    assert preference.normalize_entry("I have twenty two apples", entry) == "i have twenty two apples"
    assert standard("archaeological travelogue") == "archeological travelog"


def test_disfluency_keep_preserves_fillers_while_standard_removes_them():
    entry = _row("disfluencies", direction="keep", sub_category=["filler_words"])
    preference = normalizer.preference_normalizer()
    assert preference.normalize_entry("Uh I um agree", entry) == "um i um agree"
    assert normalizer.standard_normalizer()("Uh I um agree") == "i agree"


def _prediction(category, source, variant, reference, generation, **extra):
    return scorer.evaluate_preference_sample(
        {
            "expected_answer": reference,
            "generation": generation,
            "preference_type": category,
            "preference_source_dataset": source,
            "preference_prompt_variant": variant,
            "sub_category": extra.get("sub_category", []),
            "direction": extra.get("direction", ""),
            "target_case": extra.get("target_case", ""),
        }
    )


def test_scorer_reports_both_standard_and_preference_aware_counts():
    result = _prediction(
        "normalization",
        "ami",
        "instructed",
        "I have 22 apples",
        "I have twenty two apples",
        sub_category=["numbers"],
        direction="itn",
    )
    assert result["standard_errors"] == 0
    assert result["preference_errors"] == 2
    assert result["preference_ref_words"] == 4
    assert result["missing_prediction"] is False


def test_released_empty_remove_disfluency_reference_is_valid_and_counts_insertions():
    empty = _prediction(
        "disfluencies",
        "ami",
        "instructed",
        "",
        "",
        sub_category=["filler_words"],
        direction="remove",
    )
    hallucinated = _prediction(
        "disfluencies",
        "ami",
        "instructed",
        "",
        "hello",
        sub_category=["filler_words"],
        direction="remove",
    )
    assert empty["preference_ref_words"] == 0
    assert empty["preference_errors"] == 0
    assert hallucinated["preference_insertions"] == 1
    assert hallucinated["preference_errors"] == 1


def test_metrics_emit_official_table_dimensions_and_raw_counts():
    metrics = scorer.PreferenceASRMetrics(max_k=1)
    metrics.update([_prediction("entities", "ami", "instructed", "hello world", "hello world")])
    metrics.update([_prediction("standard", "librispeech", "instructed", "one two", "one")])
    values = metrics.get_metrics()["pass@1"]
    assert values["prompt_variant"] == "instructed"
    assert values["scored_entries"] == 2
    assert values["standard_errors"] == 1
    assert values["standard_ref_words"] == 3
    assert values["standard_wer"] == 33.33
    assert values["preference_wer"] == 33.33
    assert values["standard_wer_entities"] == 0.0
    assert values["preference_wer_standard"] == 100.0
    assert values["entries_entities"] == 1
    assert values["entries_standard"] == 1
    assert values["standard_wer_ami"] == 0.0
    assert values["standard_wer_librispeech"] == 100.0


def test_metrics_reject_mixed_prompt_conditions():
    metrics = scorer.PreferenceASRMetrics(max_k=1)
    metrics.update([_prediction("entities", "ami", "default", "hello", "hello")])
    with pytest.raises(ValueError, match="summarized separately"):
        metrics.update([_prediction("entities", "ami", "instructed", "hello", "hello")])


def test_group_profiles_evaluator_and_metrics_resolution():
    assert dataset.BENCHMARKS == {
        "preference-asr-public.default": {},
        "preference-asr-public.instructed": {},
    }
    for module in (default, instructed):
        assert module.METRICS_TYPE.endswith("preference-asr-public.scorer::PreferenceASRMetrics")
        assert "PreferenceASREvaluator" in module.EVAL_ARGS
        assert isinstance(get_metrics(module.METRICS_TYPE, max_k=1), scorer.PreferenceASRMetrics)
        assert isinstance(
            get_evaluator_class("nemo_skills.dataset.preference-asr-public.scorer::PreferenceASREvaluator", {}),
            scorer.PreferenceASREvaluator,
        )


def test_group_score_keeps_conditions_separate():
    result = score.compute_score(
        {
            "preference-asr-public.default": {"pass@1": {"preference_wer": 10.0}},
            "preference-asr-public.instructed": {"pass@1": {"preference_wer": 8.0}},
        }
    )
    assert result == {
        "pass@1": {
            "default": {"preference_wer": 10.0},
            "instructed": {"preference_wer": 8.0},
        }
    }


def test_instructed_opening_repair_preserves_release_instruction_and_census():
    row = _row()
    row["instruction"] = "Transcribe the speech in the input audio. Write numbers as digits."
    record = prepare._build_record(row, 1, "instructed")
    assert record["messages"][0]["content"] == "Transcribe this recording. Write numbers as digits."
    assert record["source_instruction"] == row["instruction"]
    assert record["prompt_policy"] == "preference-asr-recording-prefix-v1"
    assert prepare.MANIFEST_CONTRACT.rows == 2822
