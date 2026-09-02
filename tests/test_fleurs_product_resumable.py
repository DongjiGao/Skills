# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import pytest

from nemo_skills.dataset.fleurs import prepare

LANGUAGES = ["en_us", "de_de"]


def _source_row(locale: str) -> dict:
    value = 0.0 if locale == "en_us" else 0.25
    return {
        "id": 1,
        "raw_transcription": f"raw-{locale}",
        "transcription": f"text-{locale}",
        "wav_filename": f"{locale}.wav",
        "samples": np.full(320, value, dtype=np.float32),
        "sampling_rate": 16000,
    }


def _source_artifact(locale: str) -> dict:
    return {
        "rows": 1,
        "tsv_sha256": hashlib.sha256(f"tsv-{locale}".encode()).hexdigest(),
        "audio_tar_sha256": hashlib.sha256(f"tar-{locale}".encode()).hexdigest(),
        "wav_inventory_sha256": prepare._inventory_sha256([f"{locale}.wav"]),
        "source_id_inventory_sha256": prepare._inventory_sha256([1]),
        "unique_source_ids": 1,
    }


def _wav_bytes() -> bytes:
    stream = io.BytesIO()
    prepare.sf.write(stream, np.zeros(320, dtype=np.float32), 16000, format="WAV", subtype="PCM_16")
    return stream.getvalue()


def test_fleurs_product_source_is_revision_pinned_and_inventory_closed(tmp_path, monkeypatch):
    tsv_path = tmp_path / "test.tsv"
    tar_path = tmp_path / "test.tar.gz"
    tsv_path.write_text("1\ten_us.wav\traw\ttext\tph\t320\tFEMALE\n", encoding="utf-8")
    payload = _wav_bytes()
    with tarfile.open(tar_path, "w:gz") as archive:
        member = tarfile.TarInfo("nested/en_us.wav")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        assert kwargs["repo_id"] == prepare.HF_REPO_ID
        assert kwargs["revision"] == prepare.HF_REVISION
        return tar_path if kwargs["filename"].endswith(".tar.gz") else tsv_path

    monkeypatch.setattr(prepare, "hf_hub_download", fake_download)
    monkeypatch.setattr(prepare, "EXPECTED_TEST_LOCALE_COUNTS", {"en_us": 1})
    rows, artifact = prepare._load_product_locale_source("en_us", "test", tmp_path / "cache")
    assert len(calls) == 2
    assert len(rows) == artifact["rows"] == 1
    assert rows[0]["wav_filename"] == "en_us.wav"
    assert artifact["wav_inventory_sha256"] == prepare._inventory_sha256(["en_us.wav"])
    assert artifact["source_id_inventory_sha256"] == prepare._inventory_sha256([1])
    assert artifact["unique_source_ids"] == 1


def test_fleurs_product_source_preserves_repeated_sentence_ids_and_every_wav(tmp_path):
    tsv_path = tmp_path / "test.tsv"
    tsv_path.write_text(
        "1715\tfirst.wav\traw\ttext\tph\t320\tFEMALE\n"
        "1715\tsecond.wav\traw\ttext\tph\t320\tMALE\n",
        encoding="utf-8",
    )
    rows = prepare._parse_tsv_rows(tsv_path, "bg_bg")
    assert [row["id"] for row in rows] == [1715, 1715]
    assert [row["wav_filename"] for row in rows] == ["first.wav", "second.wav"]
    assert prepare.index_by_id(rows)[1715] is rows[0]


def test_fleurs_product_source_rejects_duplicate_wav_identity(tmp_path):
    tsv_path = tmp_path / "test.tsv"
    tsv_path.write_text(
        "1\tduplicate.wav\traw-1\ttext-1\tph\t320\tFEMALE\n"
        "2\tduplicate.wav\traw-2\ttext-2\tph\t320\tMALE\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate WAV filename"):
        prepare._parse_tsv_rows(tsv_path, "en_us")


def test_fleurs_product_repeated_sentence_id_requires_identical_translation_text():
    rows = [
        {"id": 1715, "raw_transcription": "raw", "transcription": "text"},
        {"id": 1715, "raw_transcription": "different", "transcription": "text"},
    ]
    with pytest.raises(ValueError, match="conflicting translation text"):
        prepare.index_by_id(rows)


def _patch_mini_product(monkeypatch, *, expected_st_rows: int = 2) -> None:
    monkeypatch.setattr(prepare, "OFFICIAL_PRODUCT_LANGUAGES", tuple(LANGUAGES))
    monkeypatch.setattr(prepare, "EXPECTED_TEST_LOCALE_COUNTS", {locale: 1 for locale in LANGUAGES})
    monkeypatch.setattr(prepare, "EXPECTED_TEST_ASR_ROWS", 2)
    monkeypatch.setattr(prepare, "EXPECTED_TEST_ST_ROWS", expected_st_rows)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_fleurs_product_parallel_shards_publish_deterministically(tmp_path, monkeypatch):
    _patch_mini_product(monkeypatch)
    calls = []

    def fake_load(locale, split, cache_dir):
        assert split == "test"
        assert cache_dir == tmp_path / "hf-fleurs"
        calls.append(locale)
        if locale == "en_us":
            time.sleep(0.01)
        return [_source_row(locale)], _source_artifact(locale)

    monkeypatch.setattr(prepare, "_load_product_locale_source", fake_load)
    prepare.prepare_fleurs(tmp_path, "test", LANGUAGES, no_audio=False, num_workers=2)

    asr = _read_jsonl(tmp_path / "asr" / "test.jsonl")
    st = _read_jsonl(tmp_path / "st" / "test.jsonl")
    receipt = json.loads((tmp_path / prepare.PROVENANCE_FILENAME).read_text(encoding="utf-8"))
    assert sorted(calls) == sorted(LANGUAGES)
    assert [row["subset_for_metrics"] for row in asr] == LANGUAGES
    assert [row["subset_for_metrics"] for row in st] == ["de_de->en_us", "en_us->de_de"]
    assert receipt["official_product_contract"] is True
    assert receipt["language_count"] == 2
    assert receipt["asr_rows"] == receipt["st_rows"] == 2
    assert receipt["resumable_locale_shards"] is True
    assert receipt["preparation_workers"] == 2
    assert sorted(receipt["locale_shards"]) == sorted(LANGUAGES)
    assert receipt["asr_manifest_sha256"] == hashlib.sha256((tmp_path / "asr" / "test.jsonl").read_bytes()).hexdigest()
    assert receipt["st_manifest_sha256"] == hashlib.sha256((tmp_path / "st" / "test.jsonl").read_bytes()).hexdigest()


def test_fleurs_product_resume_revalidates_and_skips_completed_locales(tmp_path, monkeypatch):
    _patch_mini_product(monkeypatch)
    monkeypatch.setattr(
        prepare,
        "_load_product_locale_source",
        lambda locale, _split, _cache: ([_source_row(locale)], _source_artifact(locale)),
    )
    prepare.prepare_fleurs(tmp_path, "test", LANGUAGES, no_audio=False, num_workers=1)
    expected_asr = (tmp_path / "asr" / "test.jsonl").read_bytes()
    expected_st = (tmp_path / "st" / "test.jsonl").read_bytes()
    (tmp_path / "asr" / "test.jsonl").unlink()
    (tmp_path / "st" / "test.jsonl").unlink()
    (tmp_path / prepare.PROVENANCE_FILENAME).unlink()

    def fail_if_reloaded(*_args, **_kwargs):
        raise AssertionError("validated locale shards must resume without source reload")

    monkeypatch.setattr(prepare, "_load_product_locale_source", fail_if_reloaded)
    prepare.prepare_fleurs(tmp_path, "test", LANGUAGES, no_audio=False, num_workers=2)
    assert (tmp_path / "asr" / "test.jsonl").read_bytes() == expected_asr
    assert (tmp_path / "st" / "test.jsonl").read_bytes() == expected_st


def test_fleurs_product_corrupt_resume_receipt_fails_closed(tmp_path, monkeypatch):
    _patch_mini_product(monkeypatch)
    monkeypatch.setattr(
        prepare,
        "_load_product_locale_source",
        lambda locale, _split, _cache: ([_source_row(locale)], _source_artifact(locale)),
    )
    prepare.prepare_fleurs(tmp_path, "test", LANGUAGES, no_audio=False, num_workers=1)
    published_asr = (tmp_path / "asr" / "test.jsonl").read_bytes()
    published_st = (tmp_path / "st" / "test.jsonl").read_bytes()
    published_receipt = (tmp_path / prepare.PROVENANCE_FILENAME).read_bytes()

    _shard, locale_receipt = prepare._locale_shard_paths(tmp_path, "test", "en_us")
    receipt = json.loads(locale_receipt.read_text(encoding="utf-8"))
    receipt["source_revision"] = "0" * 40
    locale_receipt.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="source_revision"):
        prepare.prepare_fleurs(tmp_path, "test", LANGUAGES, no_audio=False, num_workers=2)

    assert (tmp_path / "asr" / "test.jsonl").read_bytes() == published_asr
    assert (tmp_path / "st" / "test.jsonl").read_bytes() == published_st
    assert (tmp_path / prepare.PROVENANCE_FILENAME).read_bytes() == published_receipt


def test_fleurs_product_closure_failure_preserves_published_manifests(tmp_path, monkeypatch):
    _patch_mini_product(monkeypatch, expected_st_rows=3)
    old_asr = b"old asr\n"
    old_st = b"old st\n"
    old_receipt = b"old receipt\n"
    (tmp_path / "asr").mkdir(parents=True)
    (tmp_path / "st").mkdir(parents=True)
    (tmp_path / "asr" / "test.jsonl").write_bytes(old_asr)
    (tmp_path / "st" / "test.jsonl").write_bytes(old_st)
    (tmp_path / prepare.PROVENANCE_FILENAME).write_bytes(old_receipt)
    monkeypatch.setattr(
        prepare,
        "_load_product_locale_source",
        lambda locale, _split, _cache: ([_source_row(locale)], _source_artifact(locale)),
    )

    with pytest.raises(ValueError, match="product closure drift"):
        prepare.prepare_fleurs(tmp_path, "test", LANGUAGES, no_audio=False, num_workers=2)
    assert (tmp_path / "asr" / "test.jsonl").read_bytes() == old_asr
    assert (tmp_path / "st" / "test.jsonl").read_bytes() == old_st
    assert (tmp_path / prepare.PROVENANCE_FILENAME).read_bytes() == old_receipt
    assert not list((tmp_path / "asr").glob(".*.tmp"))
    assert not list((tmp_path / "st").glob(".*.tmp"))


def test_fleurs_product_worker_bounds_and_environment(monkeypatch):
    monkeypatch.setenv("NEMO_SKILLS_FLEURS_NUM_WORKERS", "4")
    assert prepare._resolve_num_workers(None, 2) == 2
    with pytest.raises(ValueError, match="between 1 and 32"):
        prepare._resolve_num_workers(0, 2)
    with pytest.raises(ValueError, match="must be an integer"):
        monkeypatch.setenv("NEMO_SKILLS_FLEURS_NUM_WORKERS", "invalid")
        prepare._resolve_num_workers(None, 2)
