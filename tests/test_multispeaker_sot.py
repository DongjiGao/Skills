# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0

import hashlib
import importlib
import json
import struct
import wave
from pathlib import Path

import pytest

from nemo_skills.dataset.utils import get_dataset_module

prepare = importlib.import_module("nemo_skills.dataset.multispeaker-sot.prepare")
build_group_manifests = importlib.import_module("nemo_skills.dataset.multispeaker-sot.build_group_manifests")
nsv_converter = importlib.import_module(
    "nemo_skills.dataset.multispeaker-sot.turing-diarization-v1.convert_nsv_tokens"
)


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


def test_dfw_manifest_root_requires_an_explicit_verified_mirror(tmp_path, monkeypatch):
    monkeypatch.delenv("MULTISPEAKER_SOT_MANIFEST_ROOT", raising=False)
    with pytest.raises(ValueError, match="DFW has no canonical"):
        prepare._resolve_manifest_root("dfw", None)

    assert prepare._resolve_manifest_root("dfw", str(tmp_path)) == tmp_path.resolve()
    monkeypatch.setenv("MULTISPEAKER_SOT_MANIFEST_ROOT", str(tmp_path))
    assert prepare._resolve_manifest_root("dfw", None) == tmp_path.resolve()
    assert prepare._resolve_manifest_root("iad", None) == tmp_path.resolve()


def test_under20_ami_source_contract_is_exactly_pinned():
    assert prepare.PINNED_SOURCE_MANIFESTS["under20s-ami-ihm-dev"] == (
        "e8ebe7f8e5c285f6b3e3ef3961c3ecf9c8766bb32c294c86bead1c0ee74250c9",
        476,
    )


def test_group_registration():
    group_module, _ = get_dataset_module("multispeaker-sot")
    assert group_module.IS_BENCHMARK_GROUP is True
    expected_groups = (
        "under20s",
        "60-90s",
        "120-180s",
        "full",
        "lsm",
        "turing-diarization-v1",
        "mlc-slm-2-dev-13L",
        "notsofar1-eval",
    )
    assert group_module.EVAL_GROUPS == expected_groups
    assert tuple(group_module.BENCHMARKS) == tuple(
        f"multispeaker-sot.{name}" for name in expected_groups
    )
    assert "multispeaker-sot.full" in group_module.BENCHMARKS

    # Individual leaf benchmarks remain directly addressable.
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
    assert all(
        get_dataset_module(f"multispeaker-sot.{name}")[0].METRICS_TYPE == "audio"
        for name in expected
    )


def test_notsofar1_eval_aliases_use_the_full_evaluation_contract():
    full_module, _ = get_dataset_module("multispeaker-sot.full")
    for name in ("notsofar1-eval-ihm", "notsofar1-eval-sc"):
        module, _ = get_dataset_module(f"multispeaker-sot.{name}")
        assert module.METRICS_TYPE == full_module.METRICS_TYPE
        assert module.EVAL_ARGS == full_module.EVAL_ARGS
        assert module.GENERATION_ARGS == full_module.GENERATION_ARGS
        assert module.GENERATION_MODULE == full_module.GENERATION_MODULE
        assert module.EVAL_SPLIT == full_module.EVAL_SPLIT


def test_new_parent_groups_expose_exact_language_locale_and_microphone_leaves():
    expected = {
        "turing-diarization-v1": prepare.TURING_SUBSETS,
        "mlc-slm-2-dev-13L": prepare.MLC_SLM_2_SUBSETS,
        "notsofar1-eval": prepare.NOTSOFAR1_BENCHMARKS,
    }
    assert len(prepare.TURING_SUBSETS) == 10
    assert len(prepare.MLC_SLM_2_SUBSETS) == 13
    assert len(prepare.NOTSOFAR1_BENCHMARKS) == 2
    for parent, leaves in expected.items():
        parent_module, _ = get_dataset_module(f"multispeaker-sot.{parent}")
        assert tuple(parent_module.SUBSETS) == tuple(leaves)
        for leaf in leaves:
            leaf_module, _ = get_dataset_module(f"multispeaker-sot.{leaf}")
            assert leaf_module.METRICS_TYPE == "audio"
            assert leaf_module.EVAL_ARGS == parent_module.EVAL_ARGS


def test_allsets_group_contains_every_regular_eval_group():
    group_module, _ = get_dataset_module("multispeaker-sot.allsets")
    expected_groups = [
        "under20s",
        "60-90s",
        "120-180s",
        "full",
        "lsm",
        "turing-diarization-v1",
        "mlc-slm-2-dev-13L",
        "notsofar1-eval",
    ]

    assert group_module.SUBSETS == expected_groups

    legacy_groups = expected_groups[:5]
    flattened = [
        subset
        for group_name in legacy_groups
        for subset in get_dataset_module(f"multispeaker-sot.{group_name}")[0].SUBSETS
    ]
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == set(prepare.BENCHMARK_MANIFESTS)


def test_new_dataset_defaults_use_shared_dfw_roots_only():
    assert prepare.DEFAULT_MLC_SLM_2_OPUS_DEV_ROOT == (
        "/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr/"
        "data/multi_speaker_asr/MultiSpeakerASR/MLC_SLM_2_opus/dev"
    )
    assert prepare.DEFAULT_NOTSOFAR1_ROOT == (
        "/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr/"
        "data/multi_speaker_asr/diarization_e2e/above4spk_finetune_datasets/notsofar1"
    )
    assert prepare.DEFAULT_TURING_ROOT == (
        "/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr/"
        "data/multi_speaker_asr/diarization_e2e/above4spk_finetune_datasets/"
        "Turing_Batch_Deliveries"
    )
    for source_root in (
        prepare.DEFAULT_MLC_SLM_2_OPUS_DEV_ROOT,
        prepare.DEFAULT_NOTSOFAR1_ROOT,
        prepare.DEFAULT_TURING_ROOT,
    ):
        assert "home" not in Path(source_root).parts
        assert "pool" not in Path(source_root).parts


def test_prepare_turing_uses_opus_mixdown_and_stg_v4_2(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    batch = source_root / "Delivery_Batch1_01012026"
    session = batch / "NV-GR-SS02-CONVO01"
    session.mkdir(parents=True)
    mixed_opus = session / "NV-GR-SS02-CONVO01_mixed.opus"
    mixed_opus.write_bytes(b"delivered-opus")
    (session / "SPK01.seglst.json").write_text(
        json.dumps(
            [
                {
                    "session_id": session.name,
                    "start_time": "2.0",
                    "end_time": "3.0",
                    "words": "later",
                    "speaker": "SPK01",
                }
            ]
        ),
        encoding="utf-8",
    )
    (session / "SPK02.seglst.json").write_text(
        json.dumps(
            [
                {
                    "session_id": session.name,
                    "start_time": "1.0",
                    "end_time": "1.5",
                    "words": "earlier",
                    "speaker": "SPK02",
                }
            ]
        ),
        encoding="utf-8",
    )
    missing_mix = batch / "NV-EN-SS02-CONVO02"
    missing_mix.mkdir()
    (missing_mix / "SPK01.seglst.json").write_text("[]", encoding="utf-8")

    monkeypatch.setattr(prepare, "_audio_duration_seconds", lambda path: 3.0)

    def fake_stg(input_manifest, output_manifest, **kwargs):
        inputs = list(prepare.iter_jsonl(input_manifest))
        assert kwargs["dataset_id"] == "turing-diarization-v1"
        assert kwargs["max_speakers"] == 8
        assert len(inputs) == 1
        combined = json.loads(Path(inputs[0]["seglst_filepath"]).read_text(encoding="utf-8"))
        assert {entry["speaker"] for entry in combined} == {"SPK01", "SPK02"}
        return [
            {
                **inputs[0],
                "offset": 1.0,
                "duration": 2.0,
                "num_speakers": 2,
                "num_changes": 1,
                "global_speaker_ids": ["SPK02", "SPK01"],
                "text": "<spk:0> earlier [inhale] <spk:1> later",
            }
        ]

    monkeypatch.setattr(prepare, "_run_stg_v4_2", fake_stg)
    count = prepare.prepare_turing_diarization_v1(
        source_root=source_root,
        output_dir=tmp_path / "output",
        data_dir=tmp_path / "data",
        audio_prefix=Path("/data/multispeaker-sot/audio"),
        no_audio=False,
        max_samples=None,
        speaker_tag_style="spk",
    )

    assert count == 1
    record = json.loads((tmp_path / "output/turing-diarization-v1/test.jsonl").read_text())
    leaf_record = json.loads((tmp_path / "output/turing-diarization-v1-el/test.jsonl").read_text())
    assert leaf_record == record
    assert record["audio_filepath"].endswith("/turing-diarization-v1-el/NV-GR-SS02-CONVO01.opus")
    assert record["text"] == "<spk:0> earlier <ev:breath> <spk:1> later"
    assert record["speaker_tag_map"] == {"SPK02": "s0", "SPK01": "s1"}
    assert record["source_lang"] == "el"
    assert record["language"] == "el"
    assert record["dataset_id"] == "turing-diarization-v1-el"
    assert record["subset_for_metrics"] == "turing-diarization-v1-el"
    assert record["num_speakers"] == 2
    assert record["num_changes"] == 1
    assert (tmp_path / "data/audio/turing-diarization-v1-el/NV-GR-SS02-CONVO01.opus").read_bytes() == b"delivered-opus"


def test_prepare_mlc_slm_2_uses_processed_opus_delivery_and_stg_v4_2(tmp_path, monkeypatch):
    opus_dev_root = tmp_path / "MLC_SLM_2_opus/dev"
    locale = "es-MX"
    recording_id = "sample_phone"
    locale_root = opus_dev_root / "data" / locale
    (locale_root / "opus").mkdir(parents=True)
    (locale_root / "seglsts").mkdir()
    (locale_root / "rttms").mkdir()
    (locale_root / "seglsts" / f"{recording_id}.seglst.json").write_text(
        json.dumps(
            [
                {
                    "session_id": recording_id,
                    "start_time": "2.0",
                    "end_time": "3.0",
                    "words": "México",
                    "speaker": "O1",
                },
                {
                    "session_id": recording_id,
                    "start_time": "1.0",
                    "end_time": "1.5",
                    "words": "primero",
                    "speaker": "O2",
                },
            ]
        ),
        encoding="utf-8",
    )
    (locale_root / "rttms" / f"{recording_id}.rttm").write_text(
        "SPEAKER sample 1 1.0 0.5 <NA> <NA> O2 <NA>\n"
    )
    (locale_root / "opus" / f"{recording_id}.opus").write_bytes(b"opus")

    monkeypatch.setattr(prepare, "_audio_duration_seconds", lambda path: 3.0)

    def fake_stg(input_manifest, output_manifest, **kwargs):
        inputs = list(prepare.iter_jsonl(input_manifest))
        assert kwargs["dataset_id"] == "mlc-slm-2-dev-13L"
        assert kwargs["max_speakers"] == 4
        assert inputs[0]["source_lang"] == "es"
        return [
            {
                **inputs[0],
                "offset": 1.0,
                "duration": 2.0,
                "num_speakers": 2,
                "num_changes": 1,
                "global_speaker_ids": ["O2", "O1"],
                "text": "<spk:0> primero <spk:1> México",
            }
        ]

    monkeypatch.setattr(prepare, "_run_stg_v4_2", fake_stg)
    count = prepare.prepare_mlc_slm_2_dev_13l(
        opus_dev_root=opus_dev_root,
        output_dir=tmp_path / "output",
        data_dir=tmp_path / "data",
        audio_prefix=Path("/data/multispeaker-sot/audio"),
        no_audio=False,
        max_samples=None,
        speaker_tag_style="spk",
        locales=(locale,),
    )

    assert count == 1
    record = json.loads((tmp_path / "output/mlc-slm-2-dev-13L/test.jsonl").read_text())
    leaf_record = json.loads((tmp_path / "output/mlc-slm-2-dev-13L-es-MX/test.jsonl").read_text())
    assert leaf_record == record
    assert record["source_lang"] == "es"
    assert record["locale"] == "es-MX"
    assert record["sample_id"] == "es-MX-sample_phone"
    assert record["text"] == "<spk:0> primero <spk:1> México"
    assert record["dataset_id"] == "mlc-slm-2-dev-13L-es-MX"
    assert record["subset_for_metrics"] == "mlc-slm-2-dev-13L-es-MX"
    assert record["audio_filepath"].endswith("/mlc-slm-2-dev-13L-es-MX/sample_phone.opus")
    assert record["reference_seglst_filepath"].endswith(
        "/seglst/mlc-slm-2-dev-13L-es-MX/sample_phone_ref.seglst.json"
    )
    assert record["rttm_filepath"].endswith("/rttm/mlc-slm-2-dev-13L-es-MX/sample_phone.rttm")
    assert "México" in (tmp_path / "output/mlc-slm-2-dev-13L/test.jsonl").read_text(encoding="utf-8")
    assert "\\u00e9" not in (tmp_path / "output/mlc-slm-2-dev-13L/test.jsonl").read_text(encoding="utf-8")


def test_stg_v4_2_command_uses_full_session_overlap_contract(tmp_path, monkeypatch):
    script = tmp_path / "serialized_transcription_generator_v4_2.py"
    script.write_text("# fixture\n", encoding="utf-8")
    input_manifest = tmp_path / "input.jsonl"
    output_manifest = tmp_path / "output.jsonl"
    _write_jsonl(input_manifest, [])
    captured = {}

    def fake_run(command, check):
        captured["command"] = command
        assert check is True
        _write_jsonl(output_manifest, [])

    monkeypatch.setattr(prepare.subprocess, "run", fake_run)
    assert prepare._run_stg_v4_2(
        input_manifest,
        output_manifest,
        dataset_id="fixture",
        max_speakers=8,
        speaker_tag_style="spk",
        stg_script=script,
        python_executable="python-fixture",
    ) == []
    command = captured["command"]
    assert "--full_session" in command
    assert "--use_word_ts_sim" in command
    assert command[command.index("--min_overlap_turn_sec") + 1] == "1.2"
    assert command[command.index("--speaker_token_position") + 1] == "segment"
    assert command[command.index("--speaker_token_style") + 1] == "<spk:*>"


def test_copy_file_atomic_replaces_same_size_stale_label(tmp_path):
    source = tmp_path / "source.rttm"
    destination = tmp_path / "destination.rttm"
    source.write_text("new\n", encoding="utf-8")
    destination.write_text("old\n", encoding="utf-8")

    prepare._copy_file_atomic(source, destination)

    assert destination.read_text(encoding="utf-8") == "new\n"


def test_turing_nsv_converter_maps_event_families_and_preserves_speech_annotations():
    source = (
        "<spk:0> [inhale] hello [chuckle] [lip-smack] [typing] [groan] "
        "[hum-tune] [whoop] [cough] [cry] [unintelligible] [um] [oh] [hm] [uh]"
    )

    converted = nsv_converter.convert_text(source)

    assert converted == (
        "<spk:0> <ev:breath> hello <ev:laughter> <ev:mouth_noise> "
        "<ev:gesture_noise> <ev:effort_strain> <ev:music_vocal> "
        "<ev:shout_exclaim> <ev:throat_cough> <ev:crying> [unintelligible] um oh hm uh"
    )


def test_turing_nsv_converter_rejects_unknown_bracketed_annotation():
    with pytest.raises(ValueError, match="Unknown bracketed annotation"):
        nsv_converter.convert_text("<spk:0> hello [new-event]")


def test_turing_nsv_manifest_conversion_is_utf8_atomic_and_updates_answer(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    stats_path = tmp_path / "stats.json"
    _write_jsonl(
        input_path,
        [
            {
                "sample_id": "ko-example",
                "text": "<spk:0> 안녕[inhale] [laugh] [Umm]",
                "expected_answer": "<spk:0> 안녕[inhale] [laugh] [Umm]",
            }
        ],
    )

    report = nsv_converter.convert_manifest(input_path, output_path, stats_path)
    raw_output = output_path.read_text(encoding="utf-8")
    row = json.loads(raw_output)

    assert row["text"] == "<spk:0> 안녕 <ev:breath> <ev:laughter> um"
    assert row["expected_answer"] == row["text"]
    assert report["rows"] == 1
    assert report["changed_rows"] == 1
    assert report["source_nsv_total"] == 2
    assert report["source_filler_total"] == 1
    assert report["source_filler_counts"] == {"[Umm]": 1}
    assert report["target_nsv_counts"]["<ev:breath>"] == 1
    assert report["target_nsv_counts"]["<ev:laughter>"] == 1
    assert "안녕" in raw_output
    assert "\\uc548" not in raw_output
    assert json.loads(stats_path.read_text(encoding="utf-8"))["unknown_bracketed_annotations"] == []


def test_prepare_notsofar1_eval_uses_full_tn_raw_nopnc_opus(tmp_path):
    manifest_root = tmp_path / "manifests"
    manifest_root.mkdir()
    source_audio = tmp_path / "source" / "S32050101.opus"
    source_audio.parent.mkdir()
    source_audio.write_bytes(b"opus")
    source_manifest = manifest_root / "nemoSOT_NOTSOFAR1_ihm_eval_full_tn_raw_noPnC.jsonl"
    _write_jsonl(
        source_manifest,
        [
            {
                "audio_filepath": str(source_audio),
                "offset": 0.0,
                "duration": 360.12345,
                "text": "<spk:0> hello um <spk:1> yes",
                "source_lang": "en",
                "target_lang": "en",
                "taskname": "asr",
                "num_speakers": 2,
                "num_changes": 1,
                "dataset_id": "NOTSOFAR1",
                "global_speaker_ids": ["notsofar1_P1", "notsofar1_P2"],
            }
        ],
    )

    count = prepare.prepare_notsofar1_eval(
        "notsofar1-eval-ihm",
        manifest_root=manifest_root,
        output_dir=tmp_path / "output",
        data_dir=tmp_path / "data",
        audio_prefix=Path("/data/multispeaker-sot/audio"),
        no_audio=False,
        max_samples=None,
        speaker_tag_style="spk",
    )

    assert count == 1
    record = json.loads((tmp_path / "output/notsofar1-eval-ihm/test.jsonl").read_text())
    assert record["audio_filepath"] == "/data/multispeaker-sot/audio/notsofar1-eval-ihm/S32050101.opus"
    assert record["duration"] == 360.1234
    assert record["text"] == "<spk:0> hello um <spk:1> yes"
    assert record["expected_answer"] == record["text"]
    assert record["pnc"] == "no"
    assert record["subset_for_metrics"] == "notsofar1-eval-ihm"
    assert record["dataset_id"] == "notsofar1-eval-ihm"
    assert record["speaker_tag_map"] == {"s0": "s0", "s1": "s1"}
    assert record["global_speaker_ids"] == ["notsofar1_P1", "notsofar1_P2"]
    assert (tmp_path / "data/audio/notsofar1-eval-ihm/S32050101.opus").read_bytes() == b"opus"


def test_build_allsets_manifest_merges_every_regular_eval_group(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    bench_root = data_root / "multispeaker-sot"
    groups = get_dataset_module("multispeaker-sot.allsets")[0].SUBSETS
    for group_name in groups:
        _write_jsonl(
            bench_root / group_name / "test.jsonl",
            [{"subset_for_metrics": group_name, "audio_filepath": f"/data/multispeaker-sot/{group_name}.wav"}],
        )
    # A data directory may retain a copied pre-update module. The canonical
    # package definition must win so a normal rebuild cannot drop `full` again.
    (bench_root / "allsets" / "__init__.py").parent.mkdir(parents=True, exist_ok=True)
    (bench_root / "allsets" / "__init__.py").write_text(
        'SUBSETS = ["under20s", "60-90s", "120-180s", "lsm"]\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv", ["build_group_manifests.py", "--data-dir", str(data_root), "--groups", "allsets"]
    )
    build_group_manifests.main()

    rows = [json.loads(line) for line in (bench_root / "allsets" / "test.jsonl").read_text().splitlines()]
    assert [row["subset_for_metrics"] for row in rows] == groups


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


def test_under20_manifest_conversion_uses_spk_tags_and_audio_prefix(tmp_path, monkeypatch):
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
    manifest_path = manifest_root / "under20s_eval_set/ami_ihm_dev_10-20s.json"
    monkeypatch.setitem(
        prepare.PINNED_SOURCE_MANIFESTS,
        "under20s-ami-ihm-dev",
        (hashlib.sha256(manifest_path.read_bytes()).hexdigest(), 1),
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


def test_under20_pinned_source_manifest_rejects_hash_drift(tmp_path, monkeypatch):
    manifest = tmp_path / "under20s_eval_set/ami_ihm_dev_10-20s.json"
    _write_jsonl(manifest, [{"text": "row"}])
    monkeypatch.setitem(prepare.PINNED_SOURCE_MANIFESTS, "under20s-ami-ihm-dev", ("0" * 64, 1))

    with pytest.raises(ValueError, match="source manifest SHA-256 drift"):
        prepare._validate_pinned_source_manifest("under20s-ami-ihm-dev", manifest)


def test_under20_pinned_prepared_closure_rejects_skipped_rows(tmp_path, monkeypatch):
    manifest_root = tmp_path / "manifests"
    manifest = manifest_root / "under20s_eval_set/ami_ihm_dev_10-20s.json"
    _write_jsonl(manifest, [{"audio_filepath": "/missing.wav", "text": "[s0] row"}])
    monkeypatch.setitem(
        prepare.PINNED_SOURCE_MANIFESTS,
        "under20s-ami-ihm-dev",
        (hashlib.sha256(manifest.read_bytes()).hexdigest(), 1),
    )

    def invalid_record(*_args, **_kwargs):
        raise prepare.InvalidAudioSegmentError("invalid")

    monkeypatch.setattr(prepare, "build_record", invalid_record)
    output_dir = tmp_path / "bench"
    with pytest.raises(ValueError, match="prepared row closure drift"):
        prepare.prepare_manifest_set(
            "under20s-ami-ihm-dev",
            manifest_root=manifest_root,
            output_dir=output_dir,
            data_dir=tmp_path / "data",
            audio_prefix=Path("/data/multispeaker-sot/audio"),
            no_audio=False,
            max_samples=None,
            speaker_tag_style="spk",
            invalid_audio_policy="skip",
        )
    assert not (output_dir / "under20s-ami-ihm-dev/test.jsonl").exists()


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


def test_group_score_aggregates_the_five_regular_merged_benchmarks():
    group_module = importlib.import_module("nemo_skills.dataset.multispeaker-sot.multispeaker_sot_score")
    group_errors = {
        "under20s": 1,
        "60-90s": 2,
        "120-180s": 3,
        "full": 4,
        "lsm": 5,
    }
    combined = {
        f"multispeaker-sot.{name}": {
            "pass@1": {
                "cpwer_errors": errors,
                "cpwer_ref_words": 10,
                "num_entries": 1,
            }
        }
        for name, errors in group_errors.items()
    }

    output = group_module.compute_score(combined)["pass@1"]

    assert output["all"]["cpwer"] == 30.0
    for name, errors in group_errors.items():
        assert output[name]["cpwer"] == errors * 10.0


def test_group_score_nests_notsofar1_microphone_views_under_parent():
    group_module = importlib.import_module("nemo_skills.dataset.multispeaker-sot.multispeaker_sot_score")
    combined = {
        "multispeaker-sot.notsofar1-eval-ihm": {
            "pass@1": {"cpwer_errors": 2, "cpwer_ref_words": 10, "num_entries": 1}
        },
        "multispeaker-sot.notsofar1-eval-sc": {
            "pass@1": {"cpwer_errors": 4, "cpwer_ref_words": 10, "num_entries": 1}
        },
    }

    output = group_module.compute_score(combined)["pass@1"]

    assert output["notsofar1-eval"]["cpwer"] == 30.0
    assert output["notsofar1-eval-ihm"]["cpwer"] == 20.0
    assert output["notsofar1-eval-sc"]["cpwer"] == 40.0
    assert output["all"]["cpwer"] == 30.0


def test_group_score_recovers_leaf_counts_from_merged_parent_metrics():
    group_module = importlib.import_module("nemo_skills.dataset.multispeaker-sot.multispeaker_sot_score")
    leaf = "turing-diarization-v1-ko"
    combined = {
        "multispeaker-sot.turing-diarization-v1": {
            "pass@1": {
                "cpwer_errors": 3,
                "cpwer_substitutions": 2,
                "cpwer_insertions": 1,
                "cpwer_deletions": 0,
                "cpwer_ref_words": 20,
                "num_entries": 2,
                f"cpwer::{leaf}": 15.0,
                f"cpwer::{leaf}_errors": 3,
                f"cpwer::{leaf}_substitutions": 2,
                f"cpwer::{leaf}_insertions": 1,
                f"cpwer::{leaf}_deletions": 0,
                f"cpwer::{leaf}_ref_words": 20,
                f"cpwer::{leaf}_num_entries": 2,
            }
        }
    }

    output = group_module.compute_score(combined)["pass@1"]

    assert output["turing-diarization-v1"]["cpwer"] == 15.0
    assert output[leaf] == {
        "num_entries": 2,
        "cpwer": 15.0,
        "cpwer_errors": 3,
        "cpwer_substitutions": 2,
        "cpwer_insertions": 1,
        "cpwer_deletions": 0,
        "cpwer_ref_words": 20,
    }
