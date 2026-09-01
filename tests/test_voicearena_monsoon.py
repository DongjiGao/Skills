# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import importlib
import io
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from nemo_skills.evaluation.evaluator import get_evaluator_class
from nemo_skills.evaluation.metrics.map_metrics import get_metrics

prepare = importlib.import_module("nemo_skills.dataset.voicearena-monsoon.prepare")
scorer = importlib.import_module("nemo_skills.dataset.voicearena-monsoon.scorer")
score = importlib.import_module("nemo_skills.dataset.voicearena-monsoon.score")
dataset = importlib.import_module("nemo_skills.dataset.voicearena-monsoon")
en_dataset = importlib.import_module("nemo_skills.dataset.voicearena-monsoon.en-in.public")
hi_dataset = importlib.import_module("nemo_skills.dataset.voicearena-monsoon.hi-in.public")


def _contract(profile: str, rows: int = 2, speakers: int = 2):
    return replace(
        prepare.SOURCE_CONTRACTS[profile],
        expected_rows=rows,
        expected_speakers=speakers,
    )


def _english_row(source_id="clip-1", speaker_id=1):
    return {
        "id": source_id,
        "text": "Hello, world!",
        "speaker_id": speaker_id,
        "language": "en_in",
        "audio_length_s": 1.0,
    }


def _hindi_row(source_id="clip-1", speaker_id=1):
    return {
        "id": source_id,
        "lattice": [["नमस्ते"], ["दुनिया", "विश्व"]],
        "speaker_id": speaker_id,
        "language": "hi_in",
        "audio_length_s": 1.0,
    }


def test_public_source_contracts_are_exact_and_separate_from_private_leaderboard():
    en = prepare.SOURCE_CONTRACTS["en-in.public"]
    hi = prepare.SOURCE_CONTRACTS["hi-in.public"]
    assert en.repo == "VoiceArena/MonsoonASR-Open-ASR-leaderboard-en-IN"
    assert en.revision == "bc1da7b42ef6e2853123c97bf6d22067e4802d11"
    assert en.expected_rows == 2102
    assert en.expected_speakers == 1444
    assert en.official_metric == "corpus_micro_wer"
    assert hi.repo == "VoiceArena/MonsoonASR-Open-ASR-leaderboard-hi-IN"
    assert hi.revision == "a4b7375e9adea16c16100295976cdb0aabd9fb6d"
    assert hi.expected_rows == 753
    assert hi.expected_speakers == 468
    assert hi.official_metric == "corpus_micro_oiwer"
    assert prepare.SOURCE_LICENSE == "CC-BY-4.0"
    assert scorer.OIWER_VERSION == "0.1.4"
    requirements = (Path(__file__).parents[1] / "requirements/audio.txt").read_text()
    assert "voi-oiwer==0.1.4" in requirements
    assert "indic-nlp-library==0.92" in requirements


def test_metadata_validation_fails_closed_on_census_identity_and_reference_drift():
    contract = _contract("en-in.public")
    rows = [_english_row("clip-1", 1), _english_row("clip-2", 2)]
    assert prepare._validate_metadata(rows, contract) == rows

    with pytest.raises(ValueError, match="row-count drift"):
        prepare._validate_metadata(rows[:1], contract)
    with pytest.raises(ValueError, match="duplicate id"):
        prepare._validate_metadata([rows[0], {**rows[1], "id": "clip-1"}], contract)
    with pytest.raises(ValueError, match="speaker-count drift"):
        prepare._validate_metadata([rows[0], {**rows[1], "speaker_id": 1}], contract)
    with pytest.raises(ValueError, match="unsafe id"):
        prepare._validate_metadata([{**rows[0], "id": "../escape"}, rows[1]], contract)
    with pytest.raises(ValueError, match="empty reference"):
        prepare._validate_metadata([{**rows[0], "text": ""}, rows[1]], contract)


def test_hindi_lattice_contract_rejects_missing_or_empty_variants():
    contract = _contract("hi-in.public")
    rows = [_hindi_row("clip-1", 1), _hindi_row("clip-2", 2)]
    assert prepare._validate_metadata(rows, contract) == rows

    with pytest.raises(ValueError, match="invalid lattice"):
        prepare._validate_metadata([{**rows[0], "lattice": []}, rows[1]], contract)
    with pytest.raises(ValueError, match="empty/non-list lattice slot"):
        prepare._validate_metadata([{**rows[0], "lattice": [[]]}, rows[1]], contract)
    with pytest.raises(ValueError, match="invalid lattice variant"):
        prepare._validate_metadata([{**rows[0], "lattice": [[""]]}, rows[1]], contract)


def test_formatted_records_pin_provenance_and_preserve_hindi_lattice(monkeypatch):
    monkeypatch.setenv("NEMO_SKILLS_AUDIO_ROOT", "/data")
    en_contract = prepare.SOURCE_CONTRACTS["en-in.public"]
    hi_contract = prepare.SOURCE_CONTRACTS["hi-in.public"]
    en = prepare._build_record(_english_row(), en_contract)
    hi = prepare._build_record(_hindi_row(), hi_contract)

    assert en["expected_answer"] == "Hello, world!"
    assert en["messages"][0]["audio"] == {
        "path": "/data/voicearena-monsoon/en-in/public/audio/clip-1.wav",
        "duration": 1.0,
    }
    assert en["extra_fields"]["source_revision"] == en_contract.revision
    assert "reference_lattice" not in en
    assert hi["reference_lattice"] == [["नमस्ते"], ["दुनिया", "विश्व"]]
    assert hi["expected_answer"] == "नमस्ते दुनिया"
    assert hi["extra_fields"]["source_revision"] == hi_contract.revision


def test_pcm16_audio_contract_requires_16k_mono_and_minimum_samples(tmp_path):
    path = tmp_path / "clip.wav"
    sf.write(path, np.zeros(16000, dtype=np.float32), 16000, format="WAV", subtype="PCM_16")
    prepare._validate_wav(path, 1.0)

    sf.write(path, np.zeros(319, dtype=np.float32), 16000, format="WAV", subtype="PCM_16")
    with pytest.raises(ValueError, match="audio contract failed"):
        prepare._validate_wav(path, 319 / 16000)


def _encoded_wav(*, sample_rate=16000, channels=1, samples=16000):
    payload = io.BytesIO()
    shape = (samples,) if channels == 1 else (samples, channels)
    sf.write(payload, np.zeros(shape, dtype=np.float32), sample_rate, format="WAV", subtype="PCM_16")
    return payload.getvalue()


def test_raw_audio_decode_uses_libsndfile_for_bytes_and_path(tmp_path):
    encoded = _encoded_wav()
    samples, sample_rate = prepare._decode_audio({"bytes": encoded, "path": None})
    assert samples.shape == (16000,)
    assert samples.dtype == np.float32
    assert sample_rate == 16000

    source = tmp_path / "source.wav"
    source.write_bytes(encoded)
    from_path, path_rate = prepare._decode_audio({"bytes": None, "path": str(source)})
    np.testing.assert_array_equal(from_path, samples)
    assert path_rate == sample_rate


def test_raw_audio_decode_resamples_44k1_source_to_canonical_output():
    samples, sample_rate = prepare._decode_audio(
        {"bytes": _encoded_wav(sample_rate=44100, samples=44100), "path": None}
    )
    assert samples.shape == (16000,)
    assert samples.dtype == np.float32
    assert sample_rate == 16000


def test_raw_audio_decode_fails_closed_on_missing_or_stereo():
    with pytest.raises(ValueError, match="neither bytes nor path"):
        prepare._decode_audio({"bytes": None, "path": None})
    with pytest.raises(ValueError, match="must be mono"):
        prepare._decode_audio({"bytes": _encoded_wav(channels=2), "path": None})


def test_materialize_audio_revalidates_and_skips_complete_wav(tmp_path):
    contract = _contract("en-in.public", rows=1, speakers=1)
    row = _english_row()
    path = tmp_path / "en-in" / "public" / "audio" / "clip-1.wav"
    path.parent.mkdir(parents=True)
    path.write_bytes(_encoded_wav())

    class ExistingOnlyDataset:
        def __len__(self):
            return 1

        def __getitem__(self, _index):
            raise AssertionError("completed audio must be skipped without re-decoding")

    prepare._materialize_audio(ExistingOnlyDataset(), [row], contract, tmp_path, no_audio=False)
    prepare._validate_wav(path, 1.0)


def test_materialize_audio_writes_resampled_pcm16_output(tmp_path):
    contract = _contract("en-in.public", rows=1, speakers=1)
    row = _english_row()

    class Raw44k1Dataset:
        def __len__(self):
            return 1

        def __getitem__(self, _index):
            return {"audio": {"bytes": _encoded_wav(sample_rate=44100, samples=44100), "path": None}}

    prepare._materialize_audio(Raw44k1Dataset(), [row], contract, tmp_path, no_audio=False)
    path = tmp_path / "en-in" / "public" / "audio" / "clip-1.wav"
    with sf.SoundFile(path) as stream:
        assert (stream.samplerate, stream.channels, stream.subtype, stream.frames) == (16000, 1, "PCM_16", 16000)
    prepare._validate_wav(path, 1.0)


def test_prepare_profile_forces_raw_hf_audio_before_row_access(tmp_path, monkeypatch):
    rows = [_english_row("clip-1", 1), _english_row("clip-2", 2)]

    class RawDataset:
        def __init__(self):
            self.raw_audio = False
            self.column_names = ["audio", "id", "text", "speaker_id", "language", "audio_length_s"]

        def cast_column(self, name, feature):
            assert name == "audio"
            assert feature.decode is False
            self.raw_audio = True
            return self

        def remove_columns(self, name):
            assert name == "audio"
            assert self.raw_audio
            return rows

    source = RawDataset()
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: source)
    monkeypatch.setattr(prepare, "_manifest_dir", lambda _contract: tmp_path / "manifest")

    observed = []

    def fake_materialize(dataset, validated_rows, contract, data_root, no_audio):
        assert dataset is source
        assert dataset.raw_audio
        assert validated_rows == rows
        observed.append((contract.profile, data_root, no_audio))

    monkeypatch.setattr(prepare, "_materialize_audio", fake_materialize)
    output = prepare.prepare_profile(_contract("en-in.public"), tmp_path / "data")
    assert output == tmp_path / "manifest" / "test.jsonl"
    assert observed == [("en-in.public", tmp_path / "data", False)]
    assert output.read_text().count("\n") == 2
    receipt = json.loads((tmp_path / "manifest" / prepare.PROVENANCE_FILENAME).read_text())
    assert receipt["audio_contract"]["hf_audio_decode"] is False
    assert receipt["audio_contract"]["decode_backend"] == "libsndfile via soundfile"
    assert receipt["audio_contract"]["resampling"] == {
        "backend": "scipy.signal.resample_poly",
        "padtype": "constant",
        "scipy_version": prepare.scipy.__version__,
        "window": ["kaiser", 5.0],
    }


def test_english_scorer_uses_open_asr_normalizer_and_exact_counts():
    result = scorer.evaluate_monsoon_sample(
        {
            "monsoon_language": "en-in",
            "expected_answer": "Hello world",
            "generation": "hello brave world",
        }
    )
    assert result["monsoon_metric"] == "wer"
    assert result["monsoon_errors"] == 1
    assert result["monsoon_insertions"] == 1
    assert result["monsoon_ref_words"] == 2
    assert result["missing_prediction"] is False


def test_hindi_scorer_calls_official_oiwer_and_preserves_count_invariant(monkeypatch):
    calls = []

    def fake_oiwer(hypothesis, lattice, language):
        calls.append((hypothesis, lattice, language))
        return (50.0, [], [], [], (1, 0, 0), 2, [], lattice)

    monkeypatch.setattr(scorer, "_official_oiwer", lambda: fake_oiwer)
    lattice = [["नमस्ते"], ["दुनिया", "विश्व"]]
    result = scorer.evaluate_monsoon_sample(
        {
            "monsoon_language": "hi-in",
            "expected_answer": "नमस्ते दुनिया",
            "reference_lattice": lattice,
            "generation": "नमस्ते",
        }
    )
    assert calls == [("नमस्ते", lattice, "hindi")]
    assert result["monsoon_metric"] == "oiwer"
    assert result["monsoon_errors"] == 1
    assert result["monsoon_insertions"] == 1
    assert result["monsoon_ref_words"] == 2


def test_oiwer_distribution_version_guard_fails_before_import(monkeypatch):
    scorer._official_oiwer.cache_clear()
    monkeypatch.setattr(scorer, "version", lambda _name: "0.1.3")
    with pytest.raises(RuntimeError, match="requires voi-oiwer==0.1.4"):
        scorer._official_oiwer()
    scorer._official_oiwer.cache_clear()


def _prediction(metric, errors, ref_words, *, missing=False):
    return {
        "monsoon_metric": metric,
        "monsoon_errors": errors,
        "monsoon_substitutions": errors,
        "monsoon_insertions": 0,
        "monsoon_deletions": 0,
        "monsoon_ref_words": ref_words,
        "missing_prediction": missing,
    }


@pytest.mark.parametrize(("metric", "expected_key"), [("wer", "wer"), ("oiwer", "oiwer")])
def test_metrics_report_official_corpus_micro_aggregate(metric, expected_key):
    metrics = scorer.MonsoonASRMetrics(max_k=1)
    metrics.update([_prediction(metric, 1, 1)])
    metrics.update([_prediction(metric, 1, 9)])
    values = metrics.get_metrics()["pass@1"]
    assert values[expected_key] == 20.0
    assert values[expected_key] != pytest.approx((100.0 + 100.0 / 9.0) / 2.0)
    assert values["errors"] == 2
    assert values["ref_words"] == 10
    assert values["coverage"] == 100.0


def test_metrics_reject_mixed_wer_and_oiwer_rows():
    metrics = scorer.MonsoonASRMetrics(max_k=1)
    metrics.update([_prediction("wer", 0, 1)])
    with pytest.raises(ValueError, match="separate profiles"):
        metrics.update([_prediction("oiwer", 0, 1)])


def test_group_score_keeps_languages_separate_without_invented_average():
    combined = {
        "voicearena-monsoon.en-in.public": {"pass@1": {"wer": 4.0, "num_entries": 2102}},
        "voicearena-monsoon.hi-in.public": {"pass@1": {"oiwer": 9.0, "num_entries": 753}},
    }
    assert score.compute_score(combined) == {
        "pass@1": {
            "en-in.public": {"wer": 4.0, "num_entries": 2102},
            "hi-in.public": {"oiwer": 9.0, "num_entries": 753},
        }
    }


def test_parent_and_leaf_modules_register_custom_scorer():
    assert dataset.REQUIRES_DATA_DIR is True
    assert dataset.IS_BENCHMARK_GROUP is True
    assert set(dataset.BENCHMARKS) == {
        "voicearena-monsoon.en-in.public",
        "voicearena-monsoon.hi-in.public",
    }
    for module in (en_dataset, hi_dataset):
        assert module.REQUIRES_DATA_DIR is True
        assert module.METRICS_TYPE.endswith("voicearena-monsoon.scorer::MonsoonASRMetrics")
        assert "MonsoonASREvaluator" in module.EVAL_ARGS
        assert isinstance(get_metrics(module.METRICS_TYPE, max_k=1), scorer.MonsoonASRMetrics)
        assert isinstance(
            get_evaluator_class(
                "nemo_skills.dataset.voicearena-monsoon.scorer::MonsoonASREvaluator",
                {},
            ),
            scorer.MonsoonASREvaluator,
        )
