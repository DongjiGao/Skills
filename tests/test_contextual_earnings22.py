import importlib
import io
import json
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

prepare = importlib.import_module("nemo_skills.dataset.contextual-earnings22.prepare")


def _sample() -> dict:
    return {
        "file_id": "call-1",
        "audio_filename": "call-1__000000.00__0.wav",
        "audio_start": 0.0,
        "audio_end": 0.02,
        "duration": 0.02,
        "source_duration": 0.02,
        "duration_policy": prepare.DURATION_POLICY,
        "text": "NVIDIA Corporation",
        "norm_text": "nvidia corporation",
        "transcript_tokens": ["NVIDIA", "Corporation"],
        "local_keywords": ["NVIDIA"],
        "global_keywords": ["NVIDIA", "Corporation"],
        "is_flagged": False,
        "source_row_id": "call-1:0.00:0",
        "source_repo": prepare.HF_REPO_ID,
        "source_revision": prepare.HF_REVISION,
        "source_split": prepare.SPLIT,
        "expected_rows": prepare.EXPECTED_ROWS,
        "preparation_schema_version": prepare.PREPARATION_SCHEMA_VERSION,
    }


def _write_wav(path: Path, frames: int = 320) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * frames)


def test_contextual_earnings22_source_contract_is_immutable():
    assert prepare.HF_REPO_ID == "argmaxinc/contextual-earnings22"
    assert prepare.HF_REVISION == "59d7f94771cc197984d8038e494c83abeff87282"
    assert prepare.SPLIT == "test"
    assert prepare.EXPECTED_ROWS == 772


def test_contextual_earnings22_default_data_dir_honors_prepare_env(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_SKILLS_DATA_DIR", str(tmp_path))
    assert prepare._default_data_dir() == tmp_path / prepare.BENCHMARK_NAME


def _encoded_wav(*, sample_rate=16000, channels=1, samples=16000):
    payload = io.BytesIO()
    shape = (samples,) if channels == 1 else (samples, channels)
    sf.write(payload, np.zeros(shape, dtype=np.float32), sample_rate, format="WAV", subtype="PCM_16")
    return payload.getvalue()


def test_contextual_earnings22_raw_audio_decode_avoids_torchcodec(tmp_path):
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

    with pytest.raises(ValueError, match="must be mono"):
        prepare._decode_audio({"bytes": _encoded_wav(channels=2), "path": None})
    with pytest.raises(ValueError, match="sample-rate drift"):
        prepare._decode_audio({"bytes": _encoded_wav(sample_rate=8000), "path": None})


def test_contextual_earnings22_download_forces_raw_audio_and_atomic_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "EXPECTED_ROWS", 1)
    raw_row = {
        "audio": {"bytes": _encoded_wav(), "path": None},
        "file_id": "call-1",
        "audio_start": 0.0,
        "audio_end": 1.0,
        "duration": 1.0,
        "text": "NVIDIA Corporation",
        "norm_text": "nvidia corporation",
        "transcript": ["NVIDIA", "Corporation"],
        "dictionary": ["NVIDIA"],
        "keywords": ["NVIDIA", "Corporation"],
        "is_flagged": False,
    }

    class RawDataset:
        def __init__(self):
            self.raw_audio = False

        def __len__(self):
            return 1

        def cast_column(self, name, feature):
            assert name == "audio"
            assert feature.decode is False
            self.raw_audio = True
            return self

        def __iter__(self):
            assert self.raw_audio
            return iter([raw_row])

    source = RawDataset()
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: source)
    assert prepare.download_and_extract_audio(tmp_path) == 1
    assert not list(tmp_path.rglob("*.partial"))
    metadata = [json.loads(line) for line in (tmp_path / "samples.jsonl").read_text().splitlines()]
    assert len(metadata) == 1
    prepare._validate_wav(tmp_path / "audio" / metadata[0]["audio_filename"])


def test_contextual_earnings22_publishes_decoded_duration_and_preserves_nominal_source_duration(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "EXPECTED_ROWS", 1)
    decoded_frames = 232_319
    raw_row = {
        "audio": {"bytes": _encoded_wav(samples=decoded_frames), "path": None},
        "file_id": "call-73",
        "audio_start": 0.0,
        "audio_end": 15.0,
        "duration": 15.0,
        "text": "NVIDIA Corporation",
        "norm_text": "nvidia corporation",
        "transcript": ["NVIDIA", "Corporation"],
        "dictionary": ["NVIDIA"],
        "keywords": ["NVIDIA", "Corporation"],
        "is_flagged": False,
    }

    class RawDataset:
        def __len__(self):
            return 1

        def cast_column(self, name, feature):
            assert name == "audio"
            assert feature.decode is False
            return self

        def __iter__(self):
            return iter([raw_row])

    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: RawDataset())
    assert prepare.download_and_extract_audio(tmp_path) == 1
    sample = json.loads((tmp_path / "samples.jsonl").read_text().strip())
    assert sample["duration"] == 14.5199375
    assert sample["source_duration"] == 15.0
    assert sample["duration_policy"] == "decoded_wav_frames_divided_by_16000"
    assert sample["preparation_schema_version"] == 2
    prepare._validate_samples([sample], tmp_path, require_audio=True)

    record = prepare.format_entry(sample, "contextless", "/data/ce22")
    assert record["messages"][0]["audio"]["duration"] == 14.5199375
    assert record["duration"] == 14.5199375
    assert record["source_duration"] == 15.0


def test_contextual_earnings22_validates_all_audio_and_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "EXPECTED_ROWS", 1)
    sample = _sample()
    sample["expected_rows"] = 1
    _write_wav(tmp_path / "audio" / sample["audio_filename"])

    prepare._validate_samples([sample], tmp_path, require_audio=True)


def test_contextual_earnings22_rejects_stale_cached_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "EXPECTED_ROWS", 1)
    sample = _sample()
    sample["expected_rows"] = 1
    sample["source_revision"] = "0" * 40

    with pytest.raises(ValueError, match="pinned source contract"):
        prepare._validate_samples([sample], tmp_path, require_audio=False)


def test_contextual_earnings22_rejects_incomplete_audio_closure(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "EXPECTED_ROWS", 1)
    sample = _sample()
    sample["expected_rows"] = 1

    with pytest.raises(FileNotFoundError, match="missing or empty"):
        prepare._validate_samples([sample], tmp_path, require_audio=True)


def test_contextual_earnings22_rejects_short_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "EXPECTED_ROWS", 1)
    sample = _sample()
    sample["expected_rows"] = 1
    _write_wav(tmp_path / "audio" / sample["audio_filename"], frames=319)

    with pytest.raises(ValueError, match="audio contract failed"):
        prepare._validate_samples([sample], tmp_path, require_audio=True)


def test_contextual_earnings22_records_unique_mode_id_and_source_provenance(
    monkeypatch,
):
    monkeypatch.setattr(prepare, "EXPECTED_ROWS", 1)
    sample = _sample()
    sample["expected_rows"] = 1

    row = prepare.format_entry(sample, "local", "/data/ce22")

    assert row["id"] == "contextual-earnings22-local-call-1:0.00:0"
    assert row["source_provenance"] == {
        "repo_id": "argmaxinc/contextual-earnings22",
        "revision": "59d7f94771cc197984d8038e494c83abeff87282",
        "split": "test",
        "expected_rows": 1,
        "preparation_schema_version": 2,
        "duration_policy": "decoded_wav_frames_divided_by_16000",
    }
    assert row["messages"][0]["audio"]["path"].startswith("/data/ce22/audio/")


def test_contextual_earnings22_publishes_atomically_to_data_root_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "EXPECTED_ROWS", 1)
    sample = _sample()
    sample["expected_rows"] = 1

    outputs = prepare.publish_manifests([sample], tmp_path, "/data/contextual-earnings22")
    assert outputs == {
        "contextless": tmp_path / "contextless" / "test.jsonl",
        "local": tmp_path / "local" / "test.jsonl",
        "global": tmp_path / "global" / "test.jsonl",
    }
    receipt_path = tmp_path / prepare.PROVENANCE_FILENAME
    receipt_payload = receipt_path.read_bytes()
    receipt = json.loads(receipt_payload)
    assert receipt["status"] == "ready"
    assert receipt["expected_rows_per_mode"] == 1
    assert [item["mode"] for item in receipt["manifests"]] == ["contextless", "local", "global"]
    assert all(item["rows"] == 1 for item in receipt["manifests"])
    assert not list(tmp_path.rglob("*.partial"))

    first_payloads = {mode: path.read_bytes() for mode, path in outputs.items()}
    resumed = prepare.publish_manifests([sample], tmp_path, "/data/contextual-earnings22")
    assert {mode: path.read_bytes() for mode, path in resumed.items()} == first_payloads
    assert receipt_path.read_bytes() == receipt_payload
    assert not list(tmp_path.rglob("*.partial"))


def test_contextual_earnings22_resume_rejects_stale_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "EXPECTED_ROWS", 1)
    sample = _sample()
    sample["expected_rows"] = 1
    outputs = prepare.publish_manifests([sample], tmp_path, "/data/contextual-earnings22")
    stale_path = outputs["local"]
    stale_path.write_text("stale\n")

    with pytest.raises(ValueError, match="existing artifact does not match pinned output"):
        prepare.publish_manifests([sample], tmp_path, "/data/contextual-earnings22")
    assert stale_path.read_text() == "stale\n"
    assert not list(tmp_path.rglob("*.partial"))


def test_contextual_earnings22_publication_rejects_row_count_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "EXPECTED_ROWS", 2)
    with pytest.raises(ValueError, match="publication row-count drift"):
        prepare.publish_manifests([_sample()], tmp_path, "/data/contextual-earnings22")
    assert not (tmp_path / prepare.PROVENANCE_FILENAME).exists()
