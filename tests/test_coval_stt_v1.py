# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import hashlib
import importlib
import io
import json
from decimal import Decimal

import numpy as np
import pytest
import soundfile as sf

from nemo_skills.evaluation.evaluator import get_evaluator_class
from nemo_skills.evaluation.metrics.map_metrics import get_metrics

prepare = importlib.import_module("nemo_skills.dataset.coval-stt-v1.prepare")
scorer = importlib.import_module("nemo_skills.dataset.coval-stt-v1.scorer")
dataset = importlib.import_module("nemo_skills.dataset.coval-stt-v1")


def _wav_bytes(*, sample_rate=16000, channels=1, frames=32000, subtype="PCM_16") -> bytes:
    shape = frames if channels == 1 else (frames, channels)
    samples = np.zeros(shape, dtype=np.float32)
    output = io.BytesIO()
    sf.write(output, samples, sample_rate, format="WAV", subtype=subtype)
    return output.getvalue()


def _item(payload: bytes, *, duration=2.0) -> dict:
    return {
        "path": "audio/0001.wav",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "transcript": "HELLO WORLD",
        "duration_sec": duration,
        "speaker_id": "1089",
        "chapter_id": "134686",
        "utterance_id": "1089-134686-0000",
        "speech_end_offset_ms": duration * 1000.0,
    }


def _manifest(item: dict) -> dict:
    return {
        "_license": prepare.SOURCE_CODE_LICENSE,
        "id": prepare.SOURCE_DATASET_ID,
        "version": prepare.SOURCE_DATASET_VERSION,
        "license": prepare.SOURCE_DATASET_LICENSE,
        "source": prepare.SOURCE_DATASET_DESCRIPTION,
        "items": [item],
    }


def _mini_contract(monkeypatch, item: dict) -> None:
    monkeypatch.setattr(prepare, "EXPECTED_ROWS", 1)
    monkeypatch.setattr(prepare, "EXPECTED_DURATION_SECONDS", Decimal(str(item["duration_sec"])))
    monkeypatch.setattr(prepare, "EXPECTED_SPEAKERS", 1)


def test_coval_stt_v1_exact_public_contract_pins():
    assert prepare.SOURCE_REPO == "https://github.com/coval-ai/benchmarks"
    assert prepare.SOURCE_REVISION == "db1e3d4c731b9051292fa36e73d9d2dc859372d3"
    assert prepare.SOURCE_MANIFEST_SHA256 == ("8c35e1d080e5482bb64eb8b9f39fd870ca334f9f5ca6d188680750715b6605e3")
    assert prepare.EXPECTED_ROWS == 50
    assert prepare.EXPECTED_DURATION_SECONDS == Decimal("393.265063")
    assert prepare.EXPECTED_SPEAKERS == 40
    assert prepare.SOURCE_DATASET_LICENSE == "CC-BY-4.0"
    assert prepare.SOURCE_OBJECT_ROOT == ("https://storage.googleapis.com/coval-benchmarks-datasets/stt-v1")
    assert prepare.NORMALIZER_VERSION == scorer.NORMALIZER_VERSION == "0.1.12"


def test_source_manifest_contract_is_strict(monkeypatch):
    payload = _wav_bytes()
    item = _item(payload)
    _mini_contract(monkeypatch, item)
    assert prepare._validate_source_manifest(_manifest(item)) == [item]

    exact_millisecond_item = _item(payload, duration=8.12)
    _mini_contract(monkeypatch, exact_millisecond_item)
    assert prepare._validate_source_manifest(_manifest(exact_millisecond_item)) == [exact_millisecond_item]

    bad_path = _manifest({**item, "path": "../0001.wav"})
    with pytest.raises(ValueError, match="unsafe/noncanonical path"):
        prepare._validate_source_manifest(bad_path)

    bad_hash = _manifest({**item, "sha256": "0" * 63})
    with pytest.raises(ValueError, match="invalid SHA-256"):
        prepare._validate_source_manifest(bad_hash)

    bad_identity = _manifest({**item, "utterance_id": "other"})
    with pytest.raises(ValueError, match="inconsistent LibriSpeech identity"):
        prepare._validate_source_manifest(bad_identity)


class _BytesResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_verified_download_is_atomic_and_preserves_previous_file(tmp_path, monkeypatch):
    destination = tmp_path / "artifact"
    destination.write_bytes(b"old")
    monkeypatch.setattr(prepare.urllib.request, "urlopen", lambda *_args, **_kwargs: _BytesResponse(b"bad"))
    with pytest.raises(ValueError, match="hash drift"):
        prepare._download_verified(
            "https://example.invalid/artifact", destination, hashlib.sha256(b"good").hexdigest()
        )
    assert destination.read_bytes() == b"old"
    assert not list(tmp_path.glob(".*.partial"))

    monkeypatch.setattr(prepare.urllib.request, "urlopen", lambda *_args, **_kwargs: _BytesResponse(b"good"))
    prepare._download_verified("https://example.invalid/artifact", destination, hashlib.sha256(b"good").hexdigest())
    assert destination.read_bytes() == b"good"


@pytest.mark.parametrize(
    ("sample_rate", "channels", "subtype"),
    [(8000, 1, "PCM_16"), (16000, 2, "PCM_16"), (16000, 1, "PCM_24")],
)
def test_wav_contract_rejects_non_16k_mono_pcm16(tmp_path, sample_rate, channels, subtype):
    payload = _wav_bytes(sample_rate=sample_rate, channels=channels, subtype=subtype)
    path = tmp_path / "bad.wav"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="audio format drift"):
        prepare._validate_wav(path, _item(payload))


def test_prepare_validates_every_hash_and_publishes_manifest_and_receipt_atomically(tmp_path, monkeypatch):
    payload = _wav_bytes()
    item = _item(payload)
    _mini_contract(monkeypatch, item)
    monkeypatch.setattr(prepare, "_load_source_manifest", lambda _data_dir: [item])
    audio_path = tmp_path / "audio" / "0001.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(payload)

    output_path = prepare.prepare_coval_stt_v1(tmp_path, no_audio=True)
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    receipt = json.loads((tmp_path / prepare.PROVENANCE_FILENAME).read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["id"] == item["utterance_id"]
    assert rows[0]["expected_answer"] == item["transcript"]
    assert rows[0]["messages"][0] == {"role": "system", "content": prepare.SYSTEM_PROMPT}
    assert rows[0]["messages"][1]["content"] == prepare.USER_PROMPT
    assert rows[0]["messages"][1]["audio"] == {
        "path": "/data/coval-stt-v1/audio/0001.wav",
        "duration": 2.0,
    }
    assert rows[0]["extra_fields"]["source_audio_sha256"] == item["sha256"]
    assert receipt["manifest_sha256"] == hashlib.sha256(output_path.read_bytes()).hexdigest()
    assert receipt["source_revision"] == prepare.SOURCE_REVISION
    assert receipt["source_manifest_sha256"] == prepare.SOURCE_MANIFEST_SHA256
    assert receipt["audio_contract"] == {
        "channels": 1,
        "format": "WAV",
        "per_item_sha256": True,
        "sample_rate": 16000,
        "subtype": "PCM_16",
    }
    assert receipt["quality_scoring_contract"]["official_metric"] == "utterance_macro_wer"
    assert receipt["quality_scoring_contract"]["diagnostic_metric"] == "corpus_micro_wer"
    assert not list(tmp_path.glob(".*.tmp"))


def test_failed_audio_validation_does_not_replace_published_outputs(tmp_path, monkeypatch):
    payload = _wav_bytes()
    item = _item(payload)
    _mini_contract(monkeypatch, item)
    monkeypatch.setattr(prepare, "_load_source_manifest", lambda _data_dir: [item])
    output_path = tmp_path / "test.jsonl"
    provenance_path = tmp_path / prepare.PROVENANCE_FILENAME
    output_path.write_text("old manifest\n", encoding="utf-8")
    provenance_path.write_text("old receipt\n", encoding="utf-8")
    audio_path = tmp_path / "audio" / "0001.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="audio hash mismatch"):
        prepare.prepare_coval_stt_v1(tmp_path, no_audio=True)
    assert output_path.read_text(encoding="utf-8") == "old manifest\n"
    assert provenance_path.read_text(encoding="utf-8") == "old receipt\n"


def _evaluated(reference: str, generation: str) -> dict:
    row = {"expected_answer": reference, "generation": generation}
    return {**row, **scorer.evaluate_coval_stt_v1_sample(row)}


def test_deterministic_dp_and_pinned_normalization():
    assert scorer.normalize_text("Mr. Smith paid twenty dollars.") == "mister smith paid $20"
    assert scorer.word_error_counts(["a", "b", "c"], ["a", "x", "c", "y"]) == {
        "errors": 2,
        "substitutions": 1,
        "insertions": 1,
        "deletions": 0,
        "ref_words": 3,
    }


def test_official_utterance_macro_wer_differs_from_diagnostic_corpus_micro_wer():
    short = _evaluated("alpha", "omega")
    long = _evaluated(
        "apple banana cherry delta echo foxtrot golf hotel india",
        "apple banana cherry delta echo foxtrot golf hotel juliet",
    )
    metrics = scorer.CovalSTTV1Metrics(max_k=1)
    metrics.update([short])
    metrics.update([long])
    values = metrics.get_metrics()["pass@1"]
    assert values["wer"] == 55.56
    assert values["corpus_wer_diagnostic"] == 20.0
    assert values["wer"] != values["corpus_wer_diagnostic"]
    assert values["coverage"] == 100.0
    assert values["missing_predictions"] == 0


def test_empty_generation_is_scored_as_deletions_and_reported_missing():
    metrics = scorer.CovalSTTV1Metrics(max_k=1)
    metrics.update([_evaluated("hello world", "")])
    values = metrics.get_metrics()["pass@1"]
    assert values["wer"] == 100.0
    assert values["corpus_wer_diagnostic"] == 100.0
    assert values["coverage"] == 0.0
    assert values["missing_predictions"] == 1
    assert values["wer_deletions"] == 2


def test_dataset_uses_path_registered_scorer_without_latency_claims():
    assert dataset.REQUIRES_DATA_DIR is True
    assert dataset.METRICS_TYPE.endswith("coval-stt-v1.scorer::CovalSTTV1Metrics")
    assert "coval-stt-v1.scorer::CovalSTTV1Evaluator" in dataset.EVAL_ARGS
    assert "latency" not in dataset.METRICS_TYPE.lower()
    assert isinstance(get_metrics(dataset.METRICS_TYPE, max_k=1), scorer.CovalSTTV1Metrics)
    assert isinstance(
        get_evaluator_class(
            "nemo_skills.dataset.coval-stt-v1.scorer::CovalSTTV1Evaluator",
            {},
        ),
        scorer.CovalSTTV1Evaluator,
    )
