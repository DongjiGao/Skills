# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import importlib
import json

import pytest

prepare = importlib.import_module("nemo_skills.dataset.audiobench.prepare")


def test_pinned_earnings_sources_and_row_contracts():
    assert prepare.EARNINGS_LONGFORM_DATASETS == {
        "earnings21_test": {
            "leaf": "earnings21",
            "repo": "AudioLLMs/earnings21_test",
            "revision": "1edcc0d56dc6cdb6d21420f3155936e48485928d",
            "rows": 44,
            "paper": "arxiv:2104.11348",
        },
        "earnings22_test": {
            "leaf": "earnings22",
            "repo": "AudioLLMs/earnings22_test",
            "revision": "19da542f94ac74d603b9cb5eed0e13792517ccf4",
            "rows": 125,
            "paper": "arxiv:2203.15591",
        },
    }
    prepare._validate_pinned_row_contract("earnings21_test", 44)
    with pytest.raises(ValueError, match="expected 125, found 124"):
        prepare._validate_pinned_row_contract("earnings22_test", 124)


def test_earnings_leaf_preserves_audio_only_prompt_and_provenance():
    entry = prepare.create_manifest_entry(
        sample={
            "instruction": "Transcribe the audio into text.",
            "answer": "Revenue increased by twenty percent.",
        },
        audio_filename="earnings21_test_000000.wav",
        duration=3600.0,
        dataset_name="earnings21_test",
        sample_id=0,
        category="nonjudge",
    )
    assert entry["task_type"] == "ASR"
    assert entry["messages"][1]["content"] == "Transcribe the spoken words into written form."
    assert entry["messages"][1]["content"] in prepare.AUDIOBENCH_ASR_INSTRUCTIONS
    assert "Revenue" not in entry["messages"][1]["content"]
    assert (
        entry["source_provenance"]["dataset_revision"]
        == prepare.EARNINGS_LONGFORM_DATASETS["earnings21_test"]["revision"]
    )
    assert entry["source_provenance"]["normalization"] == "audiobench"
    assert entry["source_provenance"]["prompt_seed"] == 42
    leaf = prepare.make_earnings_manifest_entry(entry)
    expected_path = "../nonjudge/audio/earnings21_test/earnings21_test_000000.wav"
    assert leaf["audio_path"] == [expected_path]
    assert leaf["messages"][1]["audio"]["path"] == expected_path
    assert leaf["messages"][1]["audios"][0]["path"] == expected_path


def test_earnings_group_and_leaf_configs_match_audiobench_protocol():
    group = importlib.import_module("nemo_skills.dataset.audiobench.earnings")
    assert group.BENCHMARKS == {
        "audiobench.earnings21": {},
        "audiobench.earnings22": {},
    }
    for name in group.BENCHMARKS:
        module = importlib.import_module(f"nemo_skills.dataset.{name}")
        assert module.METRICS_TYPE == "audio"
        assert "normalization_mode=audiobench" in module.EVAL_ARGS
        assert module.EVAL_SPLIT == "test"


def test_earnings_provenance_discloses_protocol_boundary(tmp_path):
    prepare.write_earnings_provenance(tmp_path)
    provenance = json.loads((tmp_path / "earnings" / "provenance.json").read_text())
    assert provenance["benchmark_revision"] == prepare.AUDIOBENCH_OFFICIAL_REVISION
    assert provenance["leaves"] == prepare.EARNINGS_LONGFORM_DATASETS
    assert provenance["prompt_list"] == list(prepare.AUDIOBENCH_ASR_INSTRUCTIONS)
    assert provenance["prompt_seed"] == 42
    assert "Earnings-21 and Earnings-22" in provenance["comparability_note"]
    assert "FSTAlign" in provenance["comparability_note"]
