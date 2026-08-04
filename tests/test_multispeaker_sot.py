# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0

import importlib
import json
import struct
import wave
from pathlib import Path

from nemo_skills.dataset.utils import get_dataset_module

prepare = importlib.import_module("nemo_skills.dataset.multispeaker-sot.prepare")
build_group_manifests = importlib.import_module("nemo_skills.dataset.multispeaker-sot.build_group_manifests")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_wav(path: Path, seconds: float, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fout:
        fout.setnchannels(1)
        fout.setsampwidth(2)
        fout.setframerate(sample_rate)
        fout.writeframes(struct.pack("<h", 0) * int(sample_rate * seconds))


def test_group_registration():
    group_module, _ = get_dataset_module("multispeaker-sot")
    assert group_module.IS_BENCHMARK_GROUP is True
    # one benchmark from each of the registered eval sets
    assert "multispeaker-sot.under20s-ami-ihm-dev" in group_module.BENCHMARKS
    assert "multispeaker-sot.60-90s-ami-ihm-test" in group_module.BENCHMARKS
    assert "multispeaker-sot.120-180s-mixer6-ch04-test" in group_module.BENCHMARKS
    assert "multispeaker-sot.full-chaes-ch109-test" in group_module.BENCHMARKS
    assert "multispeaker-sot.lsm-test-clean-1mix" in group_module.BENCHMARKS

    sub_module, _ = get_dataset_module("multispeaker-sot.60-90s-ami-ihm-test")
    assert sub_module.METRICS_TYPE == "audio"
    assert "++eval_config.speaker_token='<spk:*>'" in sub_module.EVAL_ARGS
    assert "++eval_config.cpwer_normalization=chime8" in sub_module.EVAL_ARGS


def test_full_group_is_a_single_benchmark_covering_every_full_length_set():
    group_module, _ = get_dataset_module("multispeaker-sot.full")

    # Merged single benchmark (one generation + one summarize job), not a group.
    assert getattr(group_module, "IS_BENCHMARK_GROUP", False) is False
    assert group_module.METRICS_TYPE == "audio"
    assert group_module.EVAL_SPLIT == "test"
    assert "++eval_config.cpwer_normalization=chime8" in group_module.EVAL_ARGS
    # The group runs through a generated shell script, where '<spk:*>' is parsed as
    # input redirection + glob and kills the job; the bare token must be used.
    assert "++eval_config.speaker_token=spk" in group_module.EVAL_ARGS
    assert "<spk:*>" not in group_module.EVAL_ARGS

    # SUBSETS must cover every full-length set registered by the benchmark group.
    expected = sorted(name for name in prepare.BENCHMARK_MANIFESTS if name.startswith("full-"))
    assert sorted(group_module.SUBSETS) == expected
    assert all(f"multispeaker-sot.{name}" in get_dataset_module("multispeaker-sot")[0].BENCHMARKS for name in expected)


def test_build_group_manifests_finds_full_group_when_data_dir_has_no_group_modules(tmp_path, monkeypatch):
    # prepare.py writes only <set>/test.jsonl into the data dir, so group SUBSETS
    # must be read from the dataset package instead.
    data_root = tmp_path / "data"
    bench_root = data_root / "multispeaker-sot"
    subsets = get_dataset_module("multispeaker-sot.full")[0].SUBSETS
    for name in subsets:
        _write_jsonl(
            bench_root / name / "test.jsonl",
            [{"subset_for_metrics": name, "audio_filepath": f"/data/multispeaker-sot/audio/{name}/s.wav"}],
        )
    assert not list(bench_root.glob("*/__init__.py"))

    monkeypatch.setattr(
        "sys.argv", ["build_group_manifests.py", "--data-dir", str(data_root), "--groups", "full"]
    )
    build_group_manifests.main()

    rows = [json.loads(line) for line in (bench_root / "full" / "test.jsonl").read_text().splitlines()]
    assert [row["subset_for_metrics"] for row in rows] == list(subsets)
    # The prep-time /data mount prefix is rewritten to the real data dir.
    assert all(row["audio_filepath"].startswith(f"{data_root}/multispeaker-sot/audio/") for row in rows)


def test_full_length_row_without_duration_uses_probed_audio_length(tmp_path, monkeypatch):
    monkeypatch.setitem(prepare.BENCHMARK_MANIFESTS, "full-ami-ihm-test", "full_length_eval_set/ami_ihm_test_full.json")
    manifest_root = tmp_path / "manifests"
    audio = tmp_path / "audio" / "EN2002a.Mix-Headset.wav"
    _write_wav(audio, 4.0)
    # Full-session rows transcribe the whole file and routinely omit "duration".
    _write_jsonl(
        manifest_root / "full_length_eval_set/ami_ihm_test_full.json",
        [{"audio_filepath": str(audio), "text": "[s0] hello [s1] yes", "num_speakers": 2}],
    )

    count = prepare.prepare_manifest_set(
        "full-ami-ihm-test",
        manifest_root=manifest_root,
        output_dir=tmp_path / "bench",
        data_dir=tmp_path / "data",
        audio_prefix=Path("/data/multispeaker-sot/audio"),
        no_audio=False,
        max_samples=None,
        speaker_tag_style="spk",
    )

    assert count == 1
    record = json.loads((tmp_path / "bench/full-ami-ihm-test/test.jsonl").read_text().splitlines()[0])
    assert record["duration"] == 4.0
    assert record["messages"][0]["audio"]["duration"] == 4.0
    assert record["subset_for_metrics"] == "full-ami-ihm-test"


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


def test_seglst_manifest_conversion_uses_row_reference_seglst(tmp_path, monkeypatch):
    # Register an ad-hoc benchmark that resolves its SOT reference from a per-row
    # seglst_filepath (the current eval sets embed text inline, but the seglst path
    # remains supported for manifests that opt in).
    monkeypatch.setitem(prepare.BENCHMARK_MANIFESTS, "seglst-ami-ihm-test", "longform_eval_set/ami_ihm_test.json")
    manifest_root = tmp_path / "manifests"
    audio = tmp_path / "audio" / "session.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"placeholder")
    seglst = manifest_root / "longform_eval_set/ref_session.seglst.json"
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
        manifest_root / "longform_eval_set/ami_ihm_test.json",
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
        "seglst-ami-ihm-test",
        manifest_root=manifest_root,
        output_dir=tmp_path / "bench",
        data_dir=tmp_path / "data",
        audio_prefix=Path("/data/multispeaker-sot/audio"),
        no_audio=True,
        max_samples=None,
        speaker_tag_style="spk",
    )

    assert count == 1
    record = json.loads((tmp_path / "bench/seglst-ami-ihm-test/test.jsonl").read_text().splitlines()[0])
    assert record["text"] == "<spk:0> hello <spk:1> yes"
    assert record["speaker_tag_map"] == {"A": "s0", "B": "s1"}
    assert record["reference_seglst_filepath"].startswith("/data/multispeaker-sot/seglst/")



def test_seglst_manifest_conversion_uses_companion_reference_seglst(tmp_path, monkeypatch):
    # Register an ad-hoc benchmark whose SOT reference is resolved from a companion
    # SegLST file (looked up via REFERENCE_SEGLST_FILENAMES).
    monkeypatch.setitem(prepare.BENCHMARK_MANIFESTS, "seglst-ami-ihm-test", "longform_eval_set/ami_ihm_test.json")
    monkeypatch.setitem(prepare.REFERENCE_SEGLST_FILENAMES, "seglst-ami-ihm-test", "ref_ami_ihm_test.seglst.json")
    manifest_root = tmp_path / "manifests"
    audio = tmp_path / "audio" / "EN2002a.Mix-Headset.wav"
    audio_b = tmp_path / "audio" / "EN2002b.Mix-Headset.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"placeholder")
    audio_b.write_bytes(b"placeholder")
    seglst = manifest_root / "longform_eval_set/ref_ami_ihm_test.seglst.json"
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
        manifest_root / "longform_eval_set/ami_ihm_test.json",
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
        "seglst-ami-ihm-test",
        manifest_root=manifest_root,
        output_dir=tmp_path / "bench",
        data_dir=tmp_path / "data",
        audio_prefix=Path("/data/multispeaker-sot/audio"),
        no_audio=True,
        max_samples=None,
        speaker_tag_style="spk",
    )

    assert count == 2
    records = [json.loads(line) for line in (tmp_path / "bench/seglst-ami-ihm-test/test.jsonl").read_text().splitlines()]
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
        "multispeaker-sot.full-ami-ihm-test": {
            "pass@1": {"cpwer_errors": 7, "cpwer_ref_words": 10, "num_entries": 1}
        },
        "multispeaker-sot.lsm-test-clean-1mix": {
            "pass@1": {"cpwer_errors": 1, "cpwer_ref_words": 10, "num_entries": 1}
        },
        "multispeaker-sot.lsm-test-clean-2mix": {
            "pass@1": {"cpwer_errors": 2, "cpwer_ref_words": 10, "num_entries": 1}
        },
        "multispeaker-sot.lsm-test-clean-3mix": {
            "pass@1": {"cpwer_errors": 3, "cpwer_ref_words": 10, "num_entries": 1}
        },
    }

    output = group_module.compute_score(combined)["pass@1"]

    # per-corpus collapse across duration splits (under20s + full)
    assert output["ami-ihm-test"]["cpwer"] == 50.0
    # duration buckets
    assert output["under20s"]["cpwer"] == 30.0
    assert output["full"]["cpwer"] == 70.0
    # LibriSpeechMix collapse + per-mix
    assert output["lsm-test-clean"]["cpwer"] == 20.0
    assert output["1mix"]["cpwer"] == 10.0
    assert output["2mix"]["cpwer"] == 20.0
    assert output["3mix"]["cpwer"] == 30.0
    assert output["all"]["cpwer"] == 32.0
    assert "4mix" not in output
