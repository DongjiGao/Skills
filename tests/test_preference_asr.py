# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import importlib
from pathlib import Path

import pytest

prepare = importlib.import_module("nemo_skills.dataset.preference-asr.prepare")
scorer = importlib.import_module("nemo_skills.dataset.preference-asr.scorer")


def row(**kwargs):
    return dict(preference_type="standard", sub_category=[], subset_for_metrics="gigaspeech", **kwargs)


@pytest.mark.parametrize(
    "category,sub,ref,hyp",
    [
        ("case", ["lower_case"], "hello", "Hello"),
        ("case", ["proper_punctuated_capitalized_format"], "Hello.", "Hello"),
        ("normalization", ["numbers"], "twenty two", "22"),
        ("normalization", ["website_links"], "example.com", "example com"),
        ("disfluencies", ["filler_words"], "um hello", "hello"),
    ],
)
def test_preference_distinctions_survive(category, sub, ref, hyp):
    sample = {
        "preference_type": category,
        "sub_category": sub,
        "direction": "keep",
        "expected_answer": ref,
        "generation": hyp,
    }
    assert scorer.evaluate_sample(sample)["wer_errors"] > 0
    sample["generation"] = ref
    assert scorer.evaluate_sample(sample)["wer_errors"] == 0


def test_empty_removal_reference_is_retained():
    source = {
        "audio_filepath": "clip.wav",
        "preference_text": "",
        "preference_type": "disfluencies",
        "direction": "remove",
    }
    records = prepare._prepare_row(
        row=source,
        benchmark="preference-asr",
        output_dir=Path("/unused"),
        audio_root="/data",
        audio_base=None,
        corpus_name="test",
        idx=0,
        copy_audio=False,
    )
    assert records is not None
    assert all(r["expected_answer"] == "" for r in records)


def test_target_case_survives_preparation():
    source = {
        "audio_filepath": "clip.wav",
        "preference_text": "Hello.",
        "preference_type": "case",
        "sub_category": ["proper_punctuated_capitalized_format"],
        "target_case": "proper_punctuated_capitalized_format",
    }
    records = prepare._prepare_row(
        row=source,
        benchmark="preference-asr",
        output_dir=Path("/unused"),
        audio_root="/data",
        audio_base=None,
        corpus_name="test",
        idx=0,
        copy_audio=False,
    )
    assert all(r["target_case"] == source["target_case"] for r in records)


def test_empty_reference_insertions_count_in_corpus_wer():
    metrics = scorer.PreferenceASRMetrics()
    for sample in [
        row(expected_answer="hello world", generation="hello world"),
        {
            "preference_type": "disfluencies",
            "sub_category": [],
            "direction": "remove",
            "subset_for_metrics": "gigaspeech",
            "expected_answer": "",
            "generation": "unexpected",
        },
    ]:
        metrics.update([{**sample, **scorer.evaluate_sample(sample)}])
    result = metrics.get_metrics()["pass@1"]
    assert result["wer"] == 50.0
    assert result["wer_ref_words"] == 2
    assert result["num_entries"] == 2


def test_legacy_counts_fail_closed():
    with pytest.raises(ValueError, match="legacy ContextASR"):
        scorer.PreferenceASRMetrics().update([row(expected_answer="a", generation="a", wer_errors=0)])


@pytest.mark.parametrize(
    "original,expected",
    [
        (
            "Transcribe the speech in the input audio. Output in lowercase format",
            "Transcribe this recording. Output in lowercase format",
        ),
        (
            "Transcribe the speech in the input audio. Audio may contain entities: A, B.",
            "Transcribe this recording. Audio may contain entities: A, B.",
        ),
        ("Transcribe the speech in the input English audio", "Transcribe this English recording."),
        ("Transcribe the input speech. Keep filler words", "Transcribe the input speech. Keep filler words"),
        (
            "Please say: Transcribe the speech in the input audio.",
            "Please say: Transcribe the speech in the input audio.",
        ),
    ],
)
def test_prompt_repair_preserves_preference_suffix(original, expected):
    from nemo_skills.dataset.preference_asr_prompts import rewrite_instruction

    assert rewrite_instruction(original) == expected
    assert rewrite_instruction(expected) == expected


def test_native_prepare_rewrites_fine_only_and_keeps_original_instruction():
    source = {
        "audio_filepath": "clip.wav",
        "preference_text": "Hello.",
        "preference_type": "case",
        "sub_category": ["proper_punctuated_capitalized_format"],
        "instruction": "Transcribe the speech in the input audio. Output with correct casing and punctuation",
    }
    plain, fine = prepare._prepare_row(
        row=source,
        benchmark="preference-asr",
        output_dir=Path("/unused"),
        audio_root="/data",
        audio_base=None,
        corpus_name="test",
        idx=0,
        copy_audio=False,
    )
    assert plain["messages"][1]["content"] == "Transcribe this recording."
    assert fine["messages"][1]["content"] == "Transcribe this recording. Output with correct casing and punctuation"
    assert fine["source_instruction"] == source["instruction"]
    assert fine["id"] != plain["id"]


def test_group_keeps_profile_scores_separate():
    group = importlib.import_module("nemo_skills.dataset.preference-asr.score")
    result = group.compute_score(
        {"preference-asr.contextless": {"pass@1": {"wer": 10}}, "preference-asr.fine": {"pass@1": {"wer": 5}}}
    )
    assert result == {"pass@1": {"contextless": {"wer": 10}, "fine": {"wer": 5}}}


def test_prepare_dryrun_is_readonly_and_matches_written_manifests(tmp_path):
    import argparse
    import json

    import yaml

    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "clip.wav").write_bytes(b"fixture")
    manifest = tmp_path / "source.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "audio_filepath": "clip.wav",
                "preference_text": "",
                "text": "uh",
                "preference_type": "disfluencies",
                "direction": "remove",
                "sub_category": ["filler_words"],
            }
        )
        + "\n"
    )
    corpus = tmp_path / "corpus.yaml"
    corpus.write_text(
        yaml.safe_dump([{"corpus": "test", "manifest_filepath": str(manifest), "audio_filepaths": str(audio)}])
    )
    output = tmp_path / "prepared"
    args = argparse.Namespace(
        corpus_yaml=str(corpus),
        path_key="dfw",
        output_dir=str(output),
        max_samples=None,
        copy_audio=True,
        dry_run=True,
    )
    assert prepare.prepare(args) == 2
    assert not output.exists()
    args.dry_run = False
    assert prepare.prepare(args) == 2
    assert json.loads((output / "fine/test.jsonl").read_text())["expected_answer"] == ""
    with pytest.raises(FileExistsError, match="immutable"):
        prepare.prepare(args)
