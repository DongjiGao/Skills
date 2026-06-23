# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0

import importlib
import json
from pathlib import Path

from nemo_skills.dataset.utils import get_dataset_module

prepare = importlib.import_module("nemo_skills.dataset.multispeaker-sot.prepare")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_group_registration():
    group_module, _ = get_dataset_module("multispeaker-sot")
    assert group_module.IS_BENCHMARK_GROUP is True
    assert "multispeaker-sot.under20s-ami-ihm-dev" in group_module.BENCHMARKS
    assert "multispeaker-sot.over20s-mixer6-ch04-test" in group_module.BENCHMARKS

    sub_module, _ = get_dataset_module("multispeaker-sot.under20s-ami-ihm-dev")
    assert sub_module.METRICS_TYPE == "audio"
    assert "++eval_config.speaker_token='<spk:*>'" in sub_module.EVAL_ARGS
    assert "++eval_config.cpwer_normalization=chime8" in sub_module.EVAL_ARGS


def test_under20_manifest_conversion_uses_spk_tags_and_audio_prefix(tmp_path):
    manifest_root = tmp_path / "manifests"
    audio = tmp_path / "audio" / "session.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"placeholder")
    _write_jsonl(
        manifest_root / "under20s_eval_set/ami_ihm_dev_10-20s.json",
        [
            {
                "audio_filepath": str(audio),
                "rttm_filepath": "/ref/session.rttm",
                "offset": 1.25,
                "duration": 3.5,
                "text": "[s0] Hello [s1] yes",
                "num_speakers": 2,
                "num_changes": 1,
            }
        ],
    )

    count = prepare.prepare_manifest_set(
        "under20s-ami-ihm-dev",
        manifest_root=manifest_root,
        output_dir=tmp_path / "bench",
        data_dir=tmp_path / "data",
        audio_prefix=Path("/data/multispeaker-sot/audio"),
        no_audio=True,
        max_samples=None,
        speaker_tag_style="spk",
    )

    assert count == 1
    record = json.loads((tmp_path / "bench/under20s-ami-ihm-dev/test.jsonl").read_text().splitlines()[0])
    assert record["text"] == "<spk:0> Hello <spk:1> yes"
    assert record["audio_filepath"].startswith("/data/multispeaker-sot/audio/under20s-ami-ihm-dev/")
    assert record["offset"] == 1.25
    assert record["messages"][0]["audio"]["path"] == record["audio_filepath"]


def test_over20_manifest_conversion_uses_reference_seglst(tmp_path):
    manifest_root = tmp_path / "manifests"
    audio = tmp_path / "audio" / "session.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"placeholder")
    seglst = manifest_root / "over20s_eval_set/ref_session.seglst.json"
    seglst.parent.mkdir(parents=True, exist_ok=True)
    seglst.write_text(
        json.dumps(
            [
                {"session_id": "session", "start_time": 0.0, "end_time": 1.0, "words": "hello", "speaker": "A"},
                {"session_id": "session", "start_time": 1.0, "end_time": 2.0, "words": "yes", "speaker": "B"},
            ]
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        manifest_root / "over20s_eval_set/ami_ihm_test.json",
        [
            {
                "audio_filepath": str(audio),
                "duration": 2.0,
                "text": "-",
                "seglst_filepath": "ref_session.seglst.json",
                "label": "session",
            }
        ],
    )

    count = prepare.prepare_manifest_set(
        "over20s-ami-ihm-test",
        manifest_root=manifest_root,
        output_dir=tmp_path / "bench",
        data_dir=tmp_path / "data",
        audio_prefix=Path("/data/multispeaker-sot/audio"),
        no_audio=True,
        max_samples=None,
        speaker_tag_style="spk",
    )

    assert count == 1
    record = json.loads((tmp_path / "bench/over20s-ami-ihm-test/test.jsonl").read_text().splitlines()[0])
    assert record["text"] == "<spk:0> hello <spk:1> yes"
    assert record["speaker_tag_map"] == {"A": "s0", "B": "s1"}
    assert record["reference_seglst_filepath"].startswith("/data/multispeaker-sot/seglst/")



def test_over20_manifest_conversion_uses_companion_reference_seglst(tmp_path):
    manifest_root = tmp_path / "manifests"
    audio = tmp_path / "audio" / "EN2002a.Mix-Headset.wav"
    audio_b = tmp_path / "audio" / "EN2002b.Mix-Headset.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"placeholder")
    audio_b.write_bytes(b"placeholder")
    seglst = manifest_root / "over20s_eval_set/ref_ami_ihm_test.seglst.json"
    seglst.parent.mkdir(parents=True, exist_ok=True)
    seglst.write_text(
        json.dumps(
            [
                {"session_id": "EN2002a", "start_time": 0.0, "end_time": 1.0, "words": "hello", "speaker": "A"},
                {"session_id": "EN2002b", "start_time": 0.0, "end_time": 1.0, "words": "ignored", "speaker": "C"},
                {"session_id": "EN2002a", "start_time": 1.0, "end_time": 2.0, "words": "yes", "speaker": "B"},
            ]
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        manifest_root / "over20s_eval_set/ami_ihm_test.json",
        [
            {
                "audio_filepath": str(audio),
                "duration": 2.0,
                "text": "-",
                "seglst_filepath": None,
                "label": "infer",
            },
            {
                "audio_filepath": str(audio_b),
                "duration": 1.0,
                "text": "-",
                "seglst_filepath": None,
                "label": "infer",
            },
        ],
    )

    count = prepare.prepare_manifest_set(
        "over20s-ami-ihm-test",
        manifest_root=manifest_root,
        output_dir=tmp_path / "bench",
        data_dir=tmp_path / "data",
        audio_prefix=Path("/data/multispeaker-sot/audio"),
        no_audio=True,
        max_samples=None,
        speaker_tag_style="spk",
    )

    assert count == 2
    records = [json.loads(line) for line in (tmp_path / "bench/over20s-ami-ihm-test/test.jsonl").read_text().splitlines()]
    assert len({record["sample_id"] for record in records}) == 2
    assert len({record["audio_filepath"] for record in records}) == 2
    assert records[0]["sample_id"].startswith("EN2002a.Mix-Headset_")
    assert records[0]["text"] == "<spk:0> hello <spk:1> yes"
    assert records[1]["sample_id"].startswith("EN2002b.Mix-Headset_")
    assert records[1]["text"] == "<spk:0> ignored"
    ref_path = Path(str(records[0]["reference_seglst_filepath"]).replace("/data/multispeaker-sot", str(tmp_path / "data")))
    ref_entries = json.loads(ref_path.read_text())
    assert {entry["session_id"] for entry in ref_entries} == {"EN2002a"}


def test_group_score_reports_duration_collapsed_dataset_and_mix_cpwer():
    group_module = importlib.import_module("nemo_skills.dataset.multispeaker-sot.multispeaker_sot_score")
    combined = {
        "multispeaker-sot.under20s-ami-ihm-test": {
            "pass@1": {"cpwer_errors": 3, "cpwer_ref_words": 10, "num_entries": 1}
        },
        "multispeaker-sot.over20s-ami-ihm-test": {
            "pass@1": {"cpwer_errors": 7, "cpwer_ref_words": 10, "num_entries": 1}
        },
        "multispeaker-sot.under20s-lsm-test-clean-1mix": {
            "pass@1": {"cpwer_errors": 1, "cpwer_ref_words": 10, "num_entries": 1}
        },
        "multispeaker-sot.under20s-lsm-test-clean-2mix": {
            "pass@1": {"cpwer_errors": 2, "cpwer_ref_words": 10, "num_entries": 1}
        },
        "multispeaker-sot.under20s-lsm-test-clean-3mix": {
            "pass@1": {"cpwer_errors": 3, "cpwer_ref_words": 10, "num_entries": 1}
        },
    }

    output = group_module.compute_score(combined)["pass@1"]

    assert output["ami-ihm-test"]["cpwer"] == 50.0
    assert output["lsm-test-clean"]["cpwer"] == 20.0
    assert output["1mix"]["cpwer"] == 10.0
    assert output["2mix"]["cpwer"] == 20.0
    assert output["3mix"]["cpwer"] == 30.0
    assert output["all"]["cpwer"] == 32.0
    assert "4mix" not in output
