# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

import importlib
import wave
import zipfile
from collections import Counter
from pathlib import Path

import pytest

profile = importlib.import_module("nemo_skills.dataset.mmau-pro-nonmusic")
prepare = importlib.import_module("nemo_skills.dataset.mmau-pro-nonmusic.prepare")
full_profile = importlib.import_module("nemo_skills.dataset.mmau-pro")


def _pinned_census_rows():
    rows = [
        {"id": f"{category}-{index}", "category": category, "audio_path": ["data/example.wav"]}
        for category, count in prepare.PINNED_CATEGORY_COUNTS.items()
        for index in range(count)
    ]
    for row in rows[-(prepare.PINNED_AUDIO_ARITY_COUNTS[2] + prepare.PINNED_AUDIO_ARITY_COUNTS[3]) :]:
        row["audio_path"].append("data/example-2.wav")
    for row in rows[-prepare.PINNED_AUDIO_ARITY_COUNTS[3] :]:
        row["audio_path"].append("data/example-3.wav")
    return rows


def _write_wav(path: Path, duration: float, sample_rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"\0\0" * int(duration * sample_rate))


def _source_row(audio_paths):
    return {
        "id": "ordered-three-audio",
        "audio_path": audio_paths,
        "question": "Which audio comes first?",
        "answer": "The first one.",
        "choices": ["The first one.", "The last one."],
        "length_type": "short",
        "perceptual_skills": [],
        "reasoning_skills": ["Temporal Reasoning"],
        "category": "multi",
        "transcription": None,
        "task_classification": None,
        "task_identifier": None,
        "kwargs": None,
        "sub-cat": None,
    }


def test_profile_name_is_separate_and_full_mmau_pro_is_unchanged():
    assert set(profile.BENCHMARKS) == {
        "mmau-pro-nonmusic.closed_form",
        "mmau-pro-nonmusic.open_ended",
        "mmau-pro-nonmusic.instruction_following",
    }
    assert set(full_profile.BENCHMARKS) == {
        "mmau-pro.closed_form",
        "mmau-pro.open_ended",
        "mmau-pro.instruction_following",
    }
    assert profile.SCORE_MODULE == full_profile.SCORE_MODULE


def test_open_ended_judge_uses_current_nvidia_inference_hub_contract():
    open_ended = importlib.import_module("nemo_skills.dataset.mmau-pro.open_ended")
    assert open_ended.JUDGE_PIPELINE_ARGS["server_address"] == ("https://inference-api.nvidia.com/v1")
    assert "++server.api_key_env_var=NVIDIA_INFERENCE_HUB_API_KEY" in open_ended.JUDGE_ARGS


@pytest.mark.parametrize(
    "leaf,attributes",
    [
        (
            "closed_form",
            ["METRICS_TYPE", "SCORE_MODULE", "GENERATION_ARGS", "JUDGE_PIPELINE_ARGS"],
        ),
        (
            "open_ended",
            ["METRICS_TYPE", "SCORE_MODULE", "GENERATION_ARGS", "JUDGE_PIPELINE_ARGS", "JUDGE_ARGS"],
        ),
        (
            "instruction_following",
            ["METRICS_TYPE", "SCORE_MODULE", "EVAL_ARGS", "GENERATION_ARGS"],
        ),
    ],
)
def test_each_leaf_reuses_full_profile_scorer_contract(leaf, attributes):
    nonmusic = importlib.import_module(f"nemo_skills.dataset.mmau-pro-nonmusic.{leaf}")
    full = importlib.import_module(f"nemo_skills.dataset.mmau-pro.{leaf}")
    for attribute in attributes:
        assert getattr(nonmusic, attribute) == getattr(full, attribute)


def test_dataset_and_official_scorer_are_immutably_pinned():
    assert prepare.HF_REPO_ID == "gamma-lab-umd/MMAU-Pro"
    assert prepare.HF_REVISION == "b8516c217464d53be36d6dc334fda3cd578ed103"
    assert prepare.PARQUET_SHA256 == "6a0d3d5ded7ea0ced0aa73365a75804a9eeec19db00216890b2349975d097c97"
    assert prepare.OFFICIAL_SCORER_REPO == "sonalkum/MMAUPro"
    assert prepare.OFFICIAL_SCORER_REVISION == "960e93a415e2267e987ae242516e21030a7d45dc"


def test_exact_pinned_category_cardinality_and_both_explicit_policies():
    rows = _pinned_census_rows()
    assert len(rows) == prepare.PINNED_TOTAL_ROWS == 5305
    assert Counter(row["category"] for row in rows) == Counter(
        {
            "spatial_audio": 325,
            "multi": 430,
            "voice_chat": 290,
            "music_speech": 46,
            "instruction following": 87,
            "speech": 891,
            "open": 625,
            "sound_music": 50,
            "music": 1418,
            "sound": 1048,
            "sound_music_speech": 7,
            "sound_speech": 88,
        }
    )
    assert Counter(len(row["audio_path"]) for row in rows) == Counter({1: 4849, 2: 430, 3: 26})
    assert prepare.PINNED_CATEGORY_MUSIC_ROWS == 1418
    prepare.validate_pinned_census(rows)

    literal = prepare.select_rows(rows)
    strict = prepare.select_rows(rows, prepare.EXCLUDE_ANY_MUSIC_POLICY)

    assert prepare.DEFAULT_MUSIC_POLICY == prepare.CATEGORY_ONLY_POLICY
    assert len(literal) == prepare.PINNED_CATEGORY_ONLY_ROWS == 3887
    assert len(strict) == prepare.PINNED_EXCLUDE_ANY_MUSIC_ROWS == 3784
    assert all(row["category"] != "music" for row in literal)
    assert any("music" in row["category"].split("_") for row in literal)
    assert all("music" not in row["category"].split("_") for row in strict)


def test_filter_preserves_source_order_and_rejects_ambiguous_policy():
    rows = [
        {"id": "speech", "category": "speech"},
        {"id": "music", "category": "music"},
        {"id": "mixed", "category": "sound_music"},
        {"id": "sound", "category": "sound"},
    ]
    assert [row["id"] for row in prepare.select_rows(rows)] == ["speech", "mixed", "sound"]
    assert [row["id"] for row in prepare.select_rows(rows, prepare.EXCLUDE_ANY_MUSIC_POLICY)] == ["speech", "sound"]
    with pytest.raises(ValueError, match="Unknown music policy"):
        prepare.select_rows(rows, "maybe-no-music")


@pytest.mark.parametrize("audio_count", [1, 2, 3])
def test_format_entry_measures_real_durations_and_preserves_one_to_three_audio_order(tmp_path, audio_count):
    audio_paths = [f"data/audio-{index}.wav" for index in range(audio_count)]
    for audio_path, duration in zip(audio_paths, [0.25, 0.5, 0.75][:audio_count]):
        _write_wav(tmp_path / audio_path, duration)

    formatted = prepare.format_entry(
        _source_row(audio_paths),
        local_audio_root=tmp_path,
        audio_prefix="/data/mmau-pro-nonmusic/audio",
        music_policy=prepare.CATEGORY_ONLY_POLICY,
    )

    expected_paths = [f"/data/mmau-pro-nonmusic/audio/{path}" for path in audio_paths]
    assert formatted["source_audio_path"] == audio_paths
    if audio_count == 1:
        ordered_audio = [formatted["messages"][0]["audio"]]
        assert formatted["audio_path"] == expected_paths[0]
    else:
        ordered_audio = formatted["messages"][0]["audios"]
        assert formatted["audio_path"] == expected_paths
    assert [audio["path"] for audio in ordered_audio] == expected_paths
    assert [audio["duration"] for audio in ordered_audio] == [0.25, 0.5, 0.75][:audio_count]
    assert formatted["duration"] == sum([0.25, 0.5, 0.75][:audio_count])
    assert formatted["dataset_revision"] == prepare.HF_REVISION
    assert formatted["official_scorer_revision"] == prepare.OFFICIAL_SCORER_REVISION
    assert formatted["music_policy"] == prepare.CATEGORY_ONLY_POLICY


def test_safe_extractor_streams_selected_file_and_ignores_unselected(tmp_path):
    archive_path = tmp_path / "data.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("data/selected.wav", b"selected")
        archive.writestr("data/unselected.wav", b"unselected")

    output = tmp_path / "audio"
    prepare.extract_selected_audio(archive_path, ["data/selected.wav"], output)

    assert (output / "data/selected.wav").read_bytes() == b"selected"
    assert not (output / "data/unselected.wav").exists()


def test_safe_extractor_rejects_path_traversal(tmp_path):
    archive_path = tmp_path / "malicious.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.wav", b"escape")

    with pytest.raises(ValueError, match="Unsafe MMAU-Pro archive path"):
        prepare.extract_selected_audio(archive_path, ["../escape.wav"], tmp_path / "audio")
    assert not (tmp_path / "escape.wav").exists()
