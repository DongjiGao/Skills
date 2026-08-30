import importlib
import wave
from pathlib import Path

import pytest

prepare = importlib.import_module("nemo_skills.dataset.contextual-earnings22.prepare")


def _sample() -> dict:
    return {
        "file_id": "call-1",
        "audio_filename": "call-1__000000.00__0.wav",
        "audio_start": 0.0,
        "audio_end": 1.0,
        "duration": 1.0,
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
    }
    assert row["messages"][0]["audio"]["path"].startswith("/data/ce22/audio/")
