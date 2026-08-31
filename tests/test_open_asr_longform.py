# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

import hashlib
import importlib
import io
import json

import numpy as np
import pytest
import soundfile as sf

from nemo_skills.pipeline.summarize_results import (
    ASR_LEADERBOARD_LONGFORM_CORAAL_SUBSETS,
    ASR_LEADERBOARD_LONGFORM_PRIMARY_SUBSETS,
    _set_asr_leaderboard_longform_macro_wer,
)

prepare = importlib.import_module("nemo_skills.dataset.asr-leaderboard-longform.prepare")
profile = importlib.import_module("nemo_skills.dataset.asr-leaderboard-longform")
audiobench_earnings = importlib.import_module("nemo_skills.dataset.audiobench.earnings21")


class FakeDataset(list):
    def __init__(self, rows, column_names=None):
        super().__init__(rows)
        self._column_names = list(column_names or (rows[0] if rows else []))
        self.cast = None

    @property
    def column_names(self):
        return self._column_names

    def cast_column(self, name, feature):
        self.cast = (name, feature)
        return self


def _spec(leaf="earnings22", *, rows=1):
    return prepare.SourceSpec(
        leaf=leaf,
        repo="source/repo",
        revision="1" * 40,
        config=leaf,
        expected_rows=rows,
        exact_columns=frozenset({"audio", "text"}),
        license="CC-BY-4.0",
    )


def _source_row(text="Hello world"):
    return {
        "audio": {"array": np.zeros(320, dtype=np.float32), "sampling_rate": 16000},
        "text": text,
    }


def test_exact_public_protocol_sources_and_census_are_pinned():
    assert prepare.OFFICIAL_PROTOCOL_REVISION == "8f37837f988e3d641e726b61087b9c761c060013"
    assert prepare.LONGFORM_REPO == "hf-audio/asr-leaderboard-longform"
    assert prepare.LONGFORM_REVISION == "4b158b35bb19ed58280bd09da84cdbbcbfa2be8e"
    assert prepare.CORAAL_REPO == "bezzam/coraal"
    assert prepare.CORAAL_REVISION == "76417d96197aa8bfa419eb5cc7637dab2f75cba0"
    assert prepare.CORAAL_ALLOWED_MISSING_SUBSET_ROWS == frozenset({("VLD", 13, "VLD_se0_ag2_f_02_1")})
    assert prepare.CORAAL_VLD_FINAL_SHARD_SHA256 == (
        "a60fafa448bf5ef6a98dafad3fcae11f50ebbfec0889fddd1389098766967d24"
    )
    assert prepare.PRIMARY_SUBSETS == ("earnings21", "earnings22", "tedlium")
    assert prepare.CORAAL_CONFIGS == ("ATL", "DCA", "DCB", "DTA", "LES", "PRV", "ROC", "VLD")
    assert prepare.EXPECTED_ROWS == {
        "earnings21": 44,
        "earnings22": 125,
        "tedlium": 11,
        "coraal_ATL": 14,
        "coraal_DCA": 74,
        "coraal_DCB": 63,
        "coraal_DTA": 40,
        "coraal_LES": 15,
        "coraal_PRV": 32,
        "coraal_ROC": 19,
        "coraal_VLD": 14,
    }
    assert prepare.EXPECTED_TOTAL_ROWS == 451
    assert prepare.OFFICIAL_NEMO_RUNNER_SHA256 == ("dcd7f91d13ef194f997d9e11b444689001ad925527172132b7797cc76df71c79")
    assert prepare.OFFICIAL_NORMALIZER_SHA256 == ("5cfcaab0e772ac331e9904495ad7c1e07e3eace072f7399d68369303d6927cc4")


def test_load_uses_exact_revision_split_and_raw_audio_cast(monkeypatch):
    spec = _spec()
    dataset = FakeDataset([_source_row()])
    observed = {}

    def fake_load_dataset(repo, config, **kwargs):
        observed.update(repo=repo, config=config, **kwargs)
        return dataset

    monkeypatch.setattr(prepare, "load_dataset", fake_load_dataset)
    assert prepare.load_pinned_source(spec) is dataset
    assert observed == {
        "repo": spec.repo,
        "config": spec.config,
        "split": "test",
        "revision": spec.revision,
    }
    name, feature = dataset.cast
    assert name == "audio"
    assert feature.decode is False


def test_raw_audio_bytes_decode_without_torchcodec():
    source = io.BytesIO()
    sf.write(source, np.zeros(320, dtype=np.float32), 16000, format="WAV", subtype="PCM_16")

    array, sample_rate = prepare._decode_audio({"bytes": source.getvalue(), "path": None})

    assert sample_rate == 16000
    assert array.shape == (320,)
    assert array.dtype == np.float32


def test_raw_audio_contract_rejects_non_mono_and_resamples_non_16khz():
    stereo = io.BytesIO()
    sf.write(stereo, np.zeros((320, 2), dtype=np.float32), 16000, format="WAV", subtype="PCM_16")
    with pytest.raises(ValueError, match="2 channels"):
        prepare._decode_audio({"bytes": stereo.getvalue(), "path": None})

    wrong_rate = io.BytesIO()
    source_rate = 44100
    samples = 0.1 * np.sin(2 * np.pi * 1000 * np.arange(source_rate) / source_rate)
    sf.write(wrong_rate, samples.astype(np.float32), source_rate, format="WAV", subtype="PCM_16")
    first, sample_rate = prepare._decode_audio({"bytes": wrong_rate.getvalue(), "path": None})
    second, second_sample_rate = prepare._decode_audio({"bytes": wrong_rate.getvalue(), "path": None})
    assert sample_rate == second_sample_rate == 16000
    assert first.shape == second.shape == (16000,)
    assert first.dtype == second.dtype == np.float32
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_raw_audio_contract_rejects_invalid_sample_rate():
    with pytest.raises(ValueError, match="invalid sample rate"):
        prepare._decode_audio({"array": np.zeros(320, dtype=np.float32), "sampling_rate": 0})


def test_source_contract_fails_closed_on_row_or_schema_drift():
    spec = _spec()
    prepare._validate_source_dataset(FakeDataset([_source_row()]), spec)
    with pytest.raises(ValueError, match="row contract"):
        prepare._validate_source_dataset(FakeDataset([]), spec)
    with pytest.raises(ValueError, match="column contract"):
        prepare._validate_source_dataset(FakeDataset([_source_row()], ["audio", "text", "new"]), spec)


def test_coraal_vld_final_row_accepts_only_authenticated_missing_subset():
    spec = next(spec for spec in prepare.SOURCE_SPECS if spec.leaf == "coraal_VLD")
    row = {
        **_source_row(),
        "file_id": "VLD_se0_ag2_f_02_1",
        "CORAAL.Sub": None,
    }

    assert prepare._validate_source_row(row, spec, 13) == (
        "VLD_se0_ag2_f_02_1",
        "Hello world",
    )


@pytest.mark.parametrize(
    ("index", "file_id", "subset"),
    [
        (12, "VLD_se0_ag2_f_02_1", None),
        (13, "VLD_unexpected", None),
        (13, "VLD_se0_ag2_f_02_1", "ROC"),
    ],
)
def test_coraal_missing_subset_exception_remains_fail_closed(index, file_id, subset):
    spec = next(spec for spec in prepare.SOURCE_SPECS if spec.leaf == "coraal_VLD")
    row = {
        **_source_row(),
        "file_id": file_id,
        "CORAAL.Sub": subset,
    }
    with pytest.raises(ValueError, match="source subset drift"):
        prepare._validate_source_row(row, spec, index)


def test_prepare_source_materializes_validated_audio_and_deterministic_prompt(tmp_path, monkeypatch):
    spec = _spec()
    monkeypatch.setattr(prepare, "load_pinned_source", lambda _spec: FakeDataset([_source_row()]))
    rows = prepare.prepare_source(spec, tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["subset_for_metrics"] == "earnings22"
    assert row["messages"][1]["content"] == "Transcribe the following audio."
    assert row["messages"][1]["audio"]["duration"] == pytest.approx(0.02)
    assert row["source_revision"] == spec.revision
    assert row["extra_fields"]["official_aggregation"].startswith("macro(earnings21")
    audio_path = tmp_path / "data" / "earnings22" / "0000-earnings22-0000.flac"
    info = sf.info(audio_path)
    assert (info.format, info.subtype, info.samplerate, info.channels, info.frames) == (
        "FLAC",
        "PCM_16",
        16000,
        1,
        320,
    )


def test_no_audio_mode_requires_complete_matching_inventory(tmp_path, monkeypatch):
    spec = _spec()
    monkeypatch.setattr(prepare, "load_pinned_source", lambda _spec: FakeDataset([_source_row()]))
    with pytest.raises(FileNotFoundError, match="Missing/empty prepared audio"):
        prepare.prepare_source(spec, tmp_path, no_audio=True)


def test_full_publication_writes_atomic_manifests_and_provenance(tmp_path, monkeypatch):
    specs = (_spec("one"), _spec("two"))
    monkeypatch.setattr(prepare, "SOURCE_SPECS", specs)
    monkeypatch.setattr(prepare, "OFFICIAL_SUBSETS", ("one", "two"))
    monkeypatch.setattr(prepare, "EXPECTED_ROWS", {"one": 1, "two": 1})
    monkeypatch.setattr(prepare, "EXPECTED_TOTAL_ROWS", 2)

    def fake_prepare_source(spec, _output_dir, no_audio=False):
        return [
            {
                "id": spec.leaf,
                "subset_for_metrics": spec.leaf,
                "expected_answer": "hello",
                "messages": [],
            }
        ]

    monkeypatch.setattr(prepare, "prepare_source", fake_prepare_source)
    output_path = prepare.prepare_open_asr_longform(tmp_path)
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    receipt = json.loads((tmp_path / prepare.PROVENANCE_FILENAME).read_text(encoding="utf-8"))
    assert [row["subset_for_metrics"] for row in rows] == ["one", "two"]
    assert receipt["total_rows"] == 2
    assert receipt["manifests_sha256"]["test.jsonl"] == hashlib.sha256(output_path.read_bytes()).hexdigest()
    assert receipt["prompt_contract"]["randomized"] is False
    assert receipt["prompt_contract"]["audio_bench_earnings_compatible"] is False
    assert receipt["audio_contract"]["resampling"] == {
        "backend": "scipy.signal.resample_poly",
        "scipy_version": prepare.scipy.__version__,
        "window": ["kaiser", 5.0],
        "padtype": "constant",
    }
    assert receipt["known_source_exceptions"] == [
        {
            "repo": prepare.CORAAL_REPO,
            "revision": prepare.CORAAL_REVISION,
            "config": "VLD",
            "source_row_index": 13,
            "file_id": "VLD_se0_ag2_f_02_1",
            "missing_field": "CORAAL.Sub",
            "source_shard": prepare.CORAAL_VLD_FINAL_SHARD,
            "source_shard_sha256": prepare.CORAAL_VLD_FINAL_SHARD_SHA256,
        }
    ]
    assert not list(tmp_path.glob(".*.tmp"))


def _leaf_metric(wer):
    return {
        "pass@1": {
            "wer": wer,
            "num_entries": 1,
            "avg_tokens": 10,
            "success_rate": 100.0,
            "no_answer": 0.0,
            "substitutions": 1,
            "insertions": 0,
            "deletions": 0,
            "ref_words": 10,
        }
    }


def test_official_hierarchical_macro_and_without_coraal_diagnostic():
    metrics = {"_all_": {"pass@1": {"wer": 999.0}}}
    for subset, wer in zip(ASR_LEADERBOARD_LONGFORM_PRIMARY_SUBSETS, (6.0, 10.0, 2.0), strict=True):
        metrics[subset] = _leaf_metric(wer)
    for subset, wer in zip(ASR_LEADERBOARD_LONGFORM_CORAAL_SUBSETS, range(10, 18), strict=True):
        metrics[subset] = _leaf_metric(float(wer))

    _set_asr_leaderboard_longform_macro_wer(metrics)
    assert metrics["coraal_avg"]["pass@1"]["wer"] == 13.5
    assert metrics["coraal_avg"]["pass@1"]["num_entries"] == 8
    assert metrics["_all_"]["pass@1"]["coraal_wer"] == 13.5
    assert metrics["_all_"]["pass@1"]["wer_without_coraal"] == 6.0
    assert metrics["_all_"]["pass@1"]["wer"] == 7.875


def test_official_macro_refuses_partial_coraal_claim():
    metrics = {"_all_": {"pass@1": {"wer": 0.0}}}
    for subset in (*ASR_LEADERBOARD_LONGFORM_PRIMARY_SUBSETS, *ASR_LEADERBOARD_LONGFORM_CORAAL_SUBSETS[:-1]):
        metrics[subset] = _leaf_metric(1.0)
    with pytest.raises(ValueError, match="leaf coverage drift"):
        _set_asr_leaderboard_longform_macro_wer(metrics)


def test_profile_uses_open_asr_scoring_not_audiobench_earnings():
    assert profile.METRICS_TYPE == "open_asr_leaderboard"
    assert "normalization_mode=hf_leaderboard" in profile.EVAL_ARGS
    assert "normalization_mode=audiobench_earnings" in audiobench_earnings.EVAL_ARGS
    assert profile.EVAL_ARGS != audiobench_earnings.EVAL_ARGS
