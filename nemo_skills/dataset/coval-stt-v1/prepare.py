# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Prepare Coval's frozen, publicly redistributable STT v1 quality profile.

The source-of-truth manifest is fetched from an immutable Coval repository
commit and authenticated before it is parsed. Each public GCS object is then
downloaded atomically, checked against its per-item SHA-256, and validated as
16 kHz mono PCM16 WAV. The NeMo Skills manifest and provenance receipt are
published only after the complete 50-item contract passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.request
from decimal import Decimal
from pathlib import Path, PurePosixPath

import soundfile as sf
from tqdm import tqdm

BENCHMARK_NAME = "coval-stt-v1"
SOURCE_REPO = "https://github.com/coval-ai/benchmarks"
SOURCE_REVISION = "db1e3d4c731b9051292fa36e73d9d2dc859372d3"
SOURCE_MANIFEST_PATH = "runner/src/coval_bench/datasets/manifests/stt-v1.json"
SOURCE_MANIFEST_URL = f"https://raw.githubusercontent.com/coval-ai/benchmarks/{SOURCE_REVISION}/{SOURCE_MANIFEST_PATH}"
SOURCE_MANIFEST_SHA256 = "8c35e1d080e5482bb64eb8b9f39fd870ca334f9f5ca6d188680750715b6605e3"
SOURCE_OBJECT_ROOT = "https://storage.googleapis.com/coval-benchmarks-datasets/stt-v1"
SOURCE_DATASET_ID = "stt-v1"
SOURCE_DATASET_VERSION = "1.0.0"
SOURCE_DATASET_LICENSE = "CC-BY-4.0"
SOURCE_DATASET_DESCRIPTION = "LibriSpeech test-clean (OpenSLR-12)"
SOURCE_CODE_LICENSE = "Apache-2.0"
EXPECTED_ROWS = 50
EXPECTED_DURATION_SECONDS = Decimal("393.265063")
EXPECTED_SPEAKERS = 40
EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHANNELS = 1
EXPECTED_SUBTYPE = "PCM_16"
EXPECTED_FORMAT = "WAV"
PROVENANCE_FILENAME = "provenance.json"
NORMALIZER_DISTRIBUTION = "whisper-normalizer"
NORMALIZER_VERSION = "0.1.12"
SYSTEM_PROMPT = "You are a helpful assistant. /no_think"
USER_PROMPT = "Produce a verbatim transcript of the audio."
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_verified(url: str, destination: Path, expected_sha256: str) -> Path:
    """Fetch one immutable artifact without publishing partial or bad bytes."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256_file(destination) == expected_sha256:
        return destination

    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "nemo-skills/coval-stt-v1"})
    try:
        with urllib.request.urlopen(request, timeout=300) as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        actual_sha256 = sha256_file(temporary)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"Coval artifact hash drift for {url}: expected {expected_sha256}, got {actual_sha256}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _require_string(item: dict, field: str, row_number: int) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Coval manifest row {row_number} has invalid {field}: {value!r}")
    return value


def _validate_source_manifest(data: object) -> list[dict]:
    if not isinstance(data, dict):
        raise TypeError("Coval source manifest must be a JSON object")
    expected_top_level = {"_license", "id", "version", "license", "source", "items"}
    if set(data) != expected_top_level:
        raise ValueError(
            f"Coval source manifest keys drifted: expected {sorted(expected_top_level)}, got {sorted(data)}"
        )
    expected_metadata = {
        "_license": SOURCE_CODE_LICENSE,
        "id": SOURCE_DATASET_ID,
        "version": SOURCE_DATASET_VERSION,
        "license": SOURCE_DATASET_LICENSE,
        "source": SOURCE_DATASET_DESCRIPTION,
    }
    for field, expected in expected_metadata.items():
        if data[field] != expected:
            raise ValueError(f"Coval source manifest {field} drift: expected {expected!r}, got {data[field]!r}")

    items = data["items"]
    if not isinstance(items, list) or len(items) != EXPECTED_ROWS:
        actual_rows = len(items) if isinstance(items, list) else None
        raise ValueError(f"Coval source row-count drift: expected {EXPECTED_ROWS}, got {actual_rows}")

    seen_paths: set[str] = set()
    seen_utterances: set[str] = set()
    speakers: set[str] = set()
    total_duration = Decimal(0)
    required_fields = {
        "path",
        "sha256",
        "transcript",
        "duration_sec",
        "speaker_id",
        "chapter_id",
        "utterance_id",
        "speech_end_offset_ms",
    }
    for row_number, item in enumerate(items, start=1):
        if not isinstance(item, dict) or set(item) != required_fields:
            actual_fields = sorted(item) if isinstance(item, dict) else None
            raise ValueError(
                f"Coval manifest row {row_number} fields drifted: "
                f"expected {sorted(required_fields)}, got {actual_fields}"
            )

        relative_path = _require_string(item, "path", row_number)
        posix_path = PurePosixPath(relative_path)
        expected_path = PurePosixPath("audio") / f"{row_number:04d}.wav"
        if posix_path != expected_path or posix_path.is_absolute() or ".." in posix_path.parts:
            raise ValueError(
                f"Coval manifest row {row_number} has unsafe/noncanonical path: "
                f"expected {expected_path}, got {relative_path!r}"
            )
        if relative_path in seen_paths:
            raise ValueError(f"Coval manifest has duplicate path: {relative_path}")
        seen_paths.add(relative_path)

        item_sha256 = _require_string(item, "sha256", row_number)
        if _SHA256_RE.fullmatch(item_sha256) is None:
            raise ValueError(f"Coval manifest row {row_number} has invalid SHA-256: {item_sha256!r}")
        _require_string(item, "transcript", row_number)
        speaker_id = _require_string(item, "speaker_id", row_number)
        chapter_id = _require_string(item, "chapter_id", row_number)
        utterance_id = _require_string(item, "utterance_id", row_number)
        if not utterance_id.startswith(f"{speaker_id}-{chapter_id}-"):
            raise ValueError(
                f"Coval manifest row {row_number} has inconsistent LibriSpeech identity: {utterance_id!r}"
            )
        if utterance_id in seen_utterances:
            raise ValueError(f"Coval manifest has duplicate utterance_id: {utterance_id}")
        seen_utterances.add(utterance_id)
        speakers.add(speaker_id)

        duration = item["duration_sec"]
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not 2.0 <= duration <= 15.0:
            raise ValueError(f"Coval manifest row {row_number} has invalid duration_sec: {duration!r}")
        duration_decimal = Decimal(str(duration))
        total_duration += duration_decimal
        speech_end_ms = item["speech_end_offset_ms"]
        if (
            isinstance(speech_end_ms, bool)
            or not isinstance(speech_end_ms, (int, float))
            or not Decimal(0) <= Decimal(str(speech_end_ms)) <= duration_decimal * 1000
        ):
            raise ValueError(f"Coval manifest row {row_number} has invalid speech_end_offset_ms: {speech_end_ms!r}")

    if total_duration != EXPECTED_DURATION_SECONDS:
        raise ValueError(f"Coval duration drift: expected {EXPECTED_DURATION_SECONDS}, got {total_duration}")
    if len(speakers) != EXPECTED_SPEAKERS:
        raise ValueError(f"Coval speaker-count drift: expected {EXPECTED_SPEAKERS}, got {len(speakers)}")
    return items


def _load_source_manifest(data_dir: Path) -> list[dict]:
    manifest_path = _download_verified(
        SOURCE_MANIFEST_URL,
        data_dir / "source" / "stt-v1.json",
        SOURCE_MANIFEST_SHA256,
    )
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Coval source manifest is not valid UTF-8 JSON: {manifest_path}") from error
    return _validate_source_manifest(data)


def _validate_wav(path: Path, item: dict) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Coval audio is missing: {path}")
    expected_sha256 = item["sha256"]
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Coval audio hash mismatch for {path}: expected {expected_sha256}, got {actual_sha256}")
    try:
        with sf.SoundFile(str(path)) as stream:
            observed = (stream.format, stream.subtype, stream.samplerate, stream.channels)
            expected = (EXPECTED_FORMAT, EXPECTED_SUBTYPE, EXPECTED_SAMPLE_RATE, EXPECTED_CHANNELS)
            if observed != expected:
                raise ValueError(f"Coval audio format drift for {path}: expected {expected}, got {observed}")
            actual_duration = stream.frames / stream.samplerate
    except RuntimeError as error:
        raise ValueError(f"Coval audio has an unreadable WAV header: {path}") from error
    if abs(actual_duration - float(item["duration_sec"])) > 1.0 / EXPECTED_SAMPLE_RATE:
        raise ValueError(
            f"Coval audio duration drift for {path}: manifest={item['duration_sec']}, WAV={actual_duration}"
        )


def _materialize_audio(data_dir: Path, items: list[dict], *, no_audio: bool) -> None:
    for item in tqdm(items, desc="Coval STT v1 audio"):
        relative_path = PurePosixPath(item["path"])
        local_path = data_dir.joinpath(*relative_path.parts)
        if not no_audio:
            object_url = f"{SOURCE_OBJECT_ROOT}/{item['path']}"
            _download_verified(object_url, local_path, item["sha256"])
        _validate_wav(local_path, item)


def _build_record(item: dict, audio_prefix: str) -> dict:
    audio_path = f"{audio_prefix.rstrip('/')}/{item['path']}"
    duration = float(item["duration_sec"])
    audio = {"path": audio_path, "duration": duration}
    return {
        "id": item["utterance_id"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT, "audio": audio},
        ],
        "expected_answer": item["transcript"],
        "audio_filepath": audio_path,
        "audio_duration": duration,
        "subset_for_metrics": BENCHMARK_NAME,
        "speaker_id": item["speaker_id"],
        "task_type": "Coval-STT-v1-ASR",
        "extra_fields": {
            "source_repo": SOURCE_REPO,
            "source_revision": SOURCE_REVISION,
            "source_manifest_path": SOURCE_MANIFEST_PATH,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_dataset_id": SOURCE_DATASET_ID,
            "source_dataset_version": SOURCE_DATASET_VERSION,
            "source_license": SOURCE_DATASET_LICENSE,
            "source_object_path": f"{SOURCE_DATASET_ID}/{item['path']}",
            "source_audio_sha256": item["sha256"],
            "source_chapter_id": item["chapter_id"],
            "source_utterance_id": item["utterance_id"],
        },
    }


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as output:
            temporary_path = Path(output.name)
            json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def prepare_coval_stt_v1(
    data_dir: Path,
    *,
    no_audio: bool = False,
    audio_prefix: str = f"/data/{BENCHMARK_NAME}",
) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    items = _load_source_manifest(data_dir)
    _materialize_audio(data_dir, items, no_audio=no_audio)

    output_path = data_dir / "test.jsonl"
    provenance_path = data_dir / PROVENANCE_FILENAME
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=data_dir, prefix=".test.", suffix=".jsonl.tmp", delete=False
        ) as output:
            temporary_path = Path(output.name)
            record_ids: set[str] = set()
            for item in items:
                record = _build_record(item, audio_prefix)
                if record["id"] in record_ids:
                    raise ValueError(f"Coval generated duplicate record id: {record['id']}")
                record_ids.add(record["id"])
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())

        if len(record_ids) != EXPECTED_ROWS:
            raise ValueError(f"Coval generated row-count drift: expected {EXPECTED_ROWS}, got {len(record_ids)}")
        manifest_sha256 = sha256_file(temporary_path)
        receipt = {
            "schema_version": 1,
            "dataset": BENCHMARK_NAME,
            "manifest": output_path.name,
            "manifest_sha256": manifest_sha256,
            "source_repo": SOURCE_REPO,
            "source_revision": SOURCE_REVISION,
            "source_manifest_path": SOURCE_MANIFEST_PATH,
            "source_manifest_url": SOURCE_MANIFEST_URL,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_dataset_id": SOURCE_DATASET_ID,
            "source_dataset_version": SOURCE_DATASET_VERSION,
            "source_dataset_license": SOURCE_DATASET_LICENSE,
            "source_object_root": SOURCE_OBJECT_ROOT,
            "total_rows": EXPECTED_ROWS,
            "total_duration_seconds": float(EXPECTED_DURATION_SECONDS),
            "speaker_count": EXPECTED_SPEAKERS,
            "audio_mode": "validated-existing" if no_audio else "downloaded-and-validated",
            "audio_contract": {
                "format": EXPECTED_FORMAT,
                "subtype": EXPECTED_SUBTYPE,
                "sample_rate": EXPECTED_SAMPLE_RATE,
                "channels": EXPECTED_CHANNELS,
                "per_item_sha256": True,
            },
            "quality_scoring_contract": {
                "official_metric": "utterance_macro_wer",
                "diagnostic_metric": "corpus_micro_wer",
                "normalizer_distribution": NORMALIZER_DISTRIBUTION,
                "normalizer_version": NORMALIZER_VERSION,
            },
        }
        os.replace(temporary_path, output_path)
        temporary_path = None
        _write_json_atomic(provenance_path, receipt)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the pinned Coval STT v1 public quality profile")
    parser.add_argument("--data_dir", default=os.getenv("NEMO_SKILLS_DATA_DIR"))
    parser.add_argument("--audio-prefix", default=f"/data/{BENCHMARK_NAME}")
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Do not download; require every canonical WAV to exist and pass its hash/header contract.",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir) / BENCHMARK_NAME if args.data_dir else Path(__file__).parent
    output_path = prepare_coval_stt_v1(data_dir, no_audio=args.no_audio, audio_prefix=args.audio_prefix)
    print(f"Coval STT v1 prepared at {output_path}")


if __name__ == "__main__":
    main()
