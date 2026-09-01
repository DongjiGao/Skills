# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import importlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from nemo_skills.dataset.fleurs.languages import FLEURS_LANGUAGES
from nemo_skills.evaluation.evaluator import is_evaluator_registered
from nemo_skills.evaluation.evaluator.fleurs_lid import evaluate_fleurs_lid_sample
from nemo_skills.evaluation.metrics.fleurs_lid_metrics import FleursLIDMetrics
from nemo_skills.evaluation.metrics.map_metrics import get_metrics

prepare = importlib.import_module("nemo_skills.dataset.fleurs-lid.prepare")


def _evaluated(locale: str, generation: str) -> dict:
    sample = {"expected_answer": locale, "generation": generation}
    return {**sample, **evaluate_fleurs_lid_sample(sample)}


def test_fleurs_lid_has_exact_official_label_space_and_pin():
    assert len(prepare.CANONICAL_LABELS) == 102
    assert tuple(sorted(FLEURS_LANGUAGES)) == prepare.CANONICAL_LABELS
    assert len({metadata["group"] for metadata in FLEURS_LANGUAGES.values()}) == 7
    assert prepare.HF_REVISION == "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
    assert set(prepare.EXPECTED_LOCALE_COUNTS) == set(prepare.CANONICAL_LABELS)
    assert sum(prepare.EXPECTED_LOCALE_COUNTS.values()) == prepare.EXPECTED_TOTAL_ROWS == 77810
    assert prepare.EXPECTED_LOCALE_COUNTS["en_us"] == 647
    assert prepare.EXPECTED_LOCALE_COUNTS["xh_za"] == 1041
    assert prepare.EXPECTED_WAV_INVENTORY_SHA256 == (
        "58ed6f8d74f07dc401a4844b24587576a2aced7fd00fbf741fe89eb278e8d27d"
    )
    assert "en_us" in prepare.PROMPT and "yue_hant_hk" in prepare.PROMPT


def _tsv_row(row_id: int, filename: str) -> str:
    return f"{row_id}\t{filename}\tRaw text\tnormalized text\tphones\t320\tFEMALE\n"


def _wav_bytes(*, sampling_rate: int = 16000, channels: int = 1, frames: int = 320) -> bytes:
    samples = np.zeros((frames, channels), dtype=np.float32) if channels > 1 else np.zeros(frames, dtype=np.float32)
    output = io.BytesIO()
    sf.write(output, samples, sampling_rate, format="WAV", subtype="PCM_16")
    return output.getvalue()


def _write_tar(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for filename, payload in members.items():
            info = tarfile.TarInfo(filename)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def _patch_downloads(monkeypatch, tsv_path: Path, tar_path: Path) -> None:
    def fake_download(*, filename, **_kwargs):
        return str(tar_path if filename.endswith(".tar.gz") else tsv_path)

    monkeypatch.setattr(prepare, "hf_hub_download", fake_download)


def _source_row(filename: str = "123.wav") -> dict:
    return {
        "id": 7,
        "raw_transcription": "Raw text",
        "transcription": "normalized text",
        "wav_filename": filename,
        "samples": np.zeros(320, dtype=np.float32),
        "sampling_rate": 16000,
    }


def _source_artifact(rows: int = 1, filenames: list[str] | None = None) -> dict:
    return {
        "rows": rows,
        "tsv_sha256": "1" * 64,
        "audio_tar_sha256": "2" * 64,
        "wav_inventory_sha256": hashlib.sha256(
            json.dumps(sorted(filenames or ["123.wav"]), separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _patch_mini_contract(monkeypatch, *, inventory_sha256: str | None = None) -> None:
    filenames = ["123.wav"]
    monkeypatch.setattr(prepare, "CANONICAL_LABELS", ("en_us",))
    monkeypatch.setattr(prepare, "FLEURS_LANGUAGES", {"en_us": FLEURS_LANGUAGES["en_us"]})
    monkeypatch.setattr(prepare, "EXPECTED_LOCALE_COUNTS", {"en_us": 1})
    monkeypatch.setattr(prepare, "EXPECTED_TOTAL_ROWS", 1)
    monkeypatch.setattr(
        prepare,
        "EXPECTED_WAV_INVENTORY_SHA256",
        inventory_sha256 or prepare._inventory_sha256({"en_us": filenames}),
    )
    monkeypatch.setattr(prepare, "_load_locale", lambda _locale, _cache: ([_source_row()], _source_artifact()))


def test_fleurs_lid_record_ids_use_unique_locale_and_wav_identity():
    first = prepare._build_record("en_us", _source_row("111.wav"))
    second = prepare._build_record("en_us", _source_row("222.wav"))
    assert first["extra_fields"]["source_id"] == second["extra_fields"]["source_id"] == 7
    assert first["id"] == "en_us:111.wav"
    assert second["id"] == "en_us:222.wav"
    assert first["id"] != second["id"]


def test_fleurs_lid_strict_tsv_rejects_duplicate_wavs(tmp_path):
    tsv_path = tmp_path / "test.tsv"
    tsv_path.write_text(_tsv_row(1, "same.wav") + _tsv_row(1, "same.wav"), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate WAV"):
        prepare._parse_tsv_strict(tsv_path, "en_us")


def test_fleurs_lid_load_requires_tsv_tar_equality_and_16khz_mono(tmp_path, monkeypatch):
    tsv_path = tmp_path / "test.tsv"
    tar_path = tmp_path / "test.tar.gz"
    tsv_path.write_text(_tsv_row(1, "one.wav"), encoding="utf-8")
    monkeypatch.setitem(prepare.EXPECTED_LOCALE_COUNTS, "en_us", 1)
    _patch_downloads(monkeypatch, tsv_path, tar_path)

    _write_tar(tar_path, {"one.wav": _wav_bytes()})
    rows, artifact = prepare._load_locale("en_us", tmp_path / "cache")
    assert len(rows) == artifact["rows"] == 1
    assert rows[0]["sampling_rate"] == 16000
    assert len(artifact["tsv_sha256"]) == len(artifact["audio_tar_sha256"]) == 64

    _write_tar(tar_path, {"other.wav": _wav_bytes()})
    with pytest.raises(ValueError, match="TSV/tar inventory mismatch"):
        prepare._load_locale("en_us", tmp_path / "cache")

    _write_tar(tar_path, {"one.wav": _wav_bytes(sampling_rate=8000)})
    with pytest.raises(ValueError, match="16000-Hz mono"):
        prepare._load_locale("en_us", tmp_path / "cache")

    _write_tar(tar_path, {"one.wav": _wav_bytes(channels=2)})
    with pytest.raises(ValueError, match="16000-Hz mono"):
        prepare._load_locale("en_us", tmp_path / "cache")


def test_fleurs_lid_prepare_publishes_atomic_manifest_and_provenance(tmp_path, monkeypatch):
    _patch_mini_contract(monkeypatch)
    output_path = prepare.prepare_fleurs_lid(tmp_path)
    receipt_path = tmp_path / prepare.PROVENANCE_FILENAME

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert [record["id"] for record in records] == ["en_us:123.wav"]
    assert receipt["total_rows"] == 1
    assert receipt["audio_mode"] == "materialized"
    assert receipt["manifest_sha256"] == hashlib.sha256(output_path.read_bytes()).hexdigest()
    assert receipt["source_artifacts"]["en_us"] == _source_artifact()
    info = sf.info(str(tmp_path / "audio" / "en_us" / "123.wav"))
    assert (info.samplerate, info.channels, info.frames) == (16000, 1, 320)
    assert not list(tmp_path.glob(".*.tmp"))


def test_fleurs_lid_resume_uses_validated_locale_shard_without_reloading_source(tmp_path, monkeypatch):
    _patch_mini_contract(monkeypatch)
    first_output = prepare.prepare_fleurs_lid(tmp_path, num_workers=1)
    first_payload = first_output.read_bytes()
    first_receipt = json.loads((tmp_path / prepare.PROVENANCE_FILENAME).read_text(encoding="utf-8"))
    assert first_receipt["resumable_locale_shards"] is True
    assert first_receipt["preparation_workers"] == 1

    first_output.unlink()
    (tmp_path / prepare.PROVENANCE_FILENAME).unlink()

    def fail_if_source_is_reloaded(*_args, **_kwargs):
        raise AssertionError("completed locale shard must be resumed without source reload")

    monkeypatch.setattr(prepare, "_load_locale", fail_if_source_is_reloaded)
    resumed_output = prepare.prepare_fleurs_lid(tmp_path, num_workers=1)
    assert resumed_output.read_bytes() == first_payload


def test_fleurs_lid_parallel_locale_shards_merge_in_canonical_order(tmp_path, monkeypatch):
    filenames = {"en_us": "english.wav", "fr_fr": "french.wav"}
    monkeypatch.setattr(prepare, "CANONICAL_LABELS", ("en_us", "fr_fr"))
    monkeypatch.setattr(
        prepare,
        "FLEURS_LANGUAGES",
        {locale: FLEURS_LANGUAGES[locale] for locale in prepare.CANONICAL_LABELS},
    )
    monkeypatch.setattr(prepare, "EXPECTED_LOCALE_COUNTS", {"en_us": 1, "fr_fr": 1})
    monkeypatch.setattr(prepare, "EXPECTED_TOTAL_ROWS", 2)
    monkeypatch.setattr(
        prepare,
        "EXPECTED_WAV_INVENTORY_SHA256",
        prepare._inventory_sha256({locale: [filename] for locale, filename in filenames.items()}),
    )

    def fake_load(locale, _cache):
        filename = filenames[locale]
        return [_source_row(filename)], _source_artifact(filenames=[filename])

    monkeypatch.setattr(prepare, "_load_locale", fake_load)
    output_path = prepare.prepare_fleurs_lid(tmp_path, num_workers=2)
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    receipt = json.loads((tmp_path / prepare.PROVENANCE_FILENAME).read_text(encoding="utf-8"))
    assert [record["id"] for record in records] == ["en_us:english.wav", "fr_fr:french.wav"]
    assert receipt["preparation_workers"] == 2
    assert list(receipt["locale_shards"]) == ["en_us", "fr_fr"]


def test_fleurs_lid_bootstraps_interrupted_audio_only_after_exact_source_validation(tmp_path, monkeypatch):
    _patch_mini_contract(monkeypatch)
    audio_path = tmp_path / "audio" / "en_us" / "123.wav"
    audio_path.parent.mkdir(parents=True)
    sf.write(str(audio_path), np.zeros(320, dtype=np.float32), 16000)

    def fail_if_existing_audio_is_rewritten(*_args, **_kwargs):
        raise AssertionError("validated interrupted audio must be reused")

    monkeypatch.setattr(prepare, "_write_audio_atomic", fail_if_existing_audio_is_rewritten)
    output_path = prepare.prepare_fleurs_lid(tmp_path, num_workers=1)
    assert output_path.is_file()


def test_fleurs_lid_refuses_mismatched_interrupted_audio(tmp_path, monkeypatch):
    _patch_mini_contract(monkeypatch)
    audio_path = tmp_path / "audio" / "en_us" / "123.wav"
    audio_path.parent.mkdir(parents=True)
    sf.write(str(audio_path), np.ones(320, dtype=np.float32), 16000)
    with pytest.raises(ValueError, match="differs from canonical pinned source"):
        prepare.prepare_fleurs_lid(tmp_path, num_workers=1)
    assert not (tmp_path / "test.jsonl").exists()


def test_fleurs_lid_failed_contract_does_not_replace_published_files(tmp_path, monkeypatch):
    output_path = tmp_path / "test.jsonl"
    receipt_path = tmp_path / prepare.PROVENANCE_FILENAME
    output_path.write_text("old manifest\n", encoding="utf-8")
    receipt_path.write_text("old receipt\n", encoding="utf-8")
    _patch_mini_contract(monkeypatch, inventory_sha256="0" * 64)

    with pytest.raises(ValueError, match="merged WAV inventory drift"):
        prepare.prepare_fleurs_lid(tmp_path)
    assert output_path.read_text(encoding="utf-8") == "old manifest\n"
    assert receipt_path.read_text(encoding="utf-8") == "old receipt\n"
    assert not list(tmp_path.glob(".*.tmp"))


def test_fleurs_lid_no_audio_requires_and_validates_existing_files(tmp_path, monkeypatch):
    _patch_mini_contract(monkeypatch)
    with pytest.raises(ValueError, match="--no-audio requires existing audio"):
        prepare.prepare_fleurs_lid(tmp_path, no_audio=True)
    assert not (tmp_path / "test.jsonl").exists()

    audio_path = tmp_path / "audio" / "en_us" / "123.wav"
    audio_path.parent.mkdir(parents=True)
    sf.write(str(audio_path), np.zeros(320, dtype=np.float32), 16000)
    output_path = prepare.prepare_fleurs_lid(tmp_path, no_audio=True)
    receipt = json.loads((tmp_path / prepare.PROVENANCE_FILENAME).read_text(encoding="utf-8"))
    assert output_path.is_file()
    assert receipt["audio_mode"] == "validated-existing"


def test_fleurs_lid_evaluator_requires_exact_canonical_output():
    assert evaluate_fleurs_lid_sample({"expected_answer": "en_us", "generation": " en_us\n"})["lid_correct"]
    invalid = evaluate_fleurs_lid_sample({"expected_answer": "en_us", "generation": "EN_US"})
    assert invalid["valid_label"] is False
    assert invalid["lid_correct"] is False


def test_fleurs_lid_micro_locale_macro_and_geographic_reporting():
    metrics = FleursLIDMetrics(max_k=1)
    for prediction in [
        _evaluated("en_us", "en_us"),
        _evaluated("en_us", "fr_fr"),
        _evaluated("fr_fr", "fr_fr"),
        _evaluated("ja_jp", "ko_kr"),
    ]:
        metrics.update([prediction])
    output = metrics.get_metrics()["pass@1"]
    assert output["accuracy_micro"] == 50.0
    assert output["accuracy_macro"] == 50.0
    assert output["accuracy::en_us"] == 50.0
    assert output["accuracy::fr_fr"] == 100.0
    assert output["accuracy::ja_jp"] == 0.0
    assert output["accuracy_group::western_european_we"] == 66.67
    assert output["accuracy_group_macro::western_european_we"] == 75.0
    assert output["accuracy_group::chinese_japanase_korean_cjk"] == 0.0


def test_fleurs_lid_registration():
    assert is_evaluator_registered("fleurs_lid")
    assert isinstance(get_metrics("fleurs_lid", max_k=1), FleursLIDMetrics)
