# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a pinned public reconstruction of the NFA timestamp benchmark.

No paper-authored CTMs or alignment artifacts were published. This adapter
therefore downloads nothing and refuses to run without an explicitly hashed
local contract, manifest, source manifests, alignment config, and audio files.
See README.md for the artifact schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any

ARTIFACT_SCHEMA = "nfa-timestamp-local-artifact-v1"
PROTOCOL_VERSION = "nfa-timestamp-public-reconstruction-v1"
BENCHMARK_NAME = "nfa-timestamp-reconstruction"
TICK_SECONDS = Decimal("0.08")
TICK_MILLISECONDS = 80
STRICT_TOLERANCE_MILLISECONDS = 240
SYSTEM_PROMPT = ""
USER_PROMPT = "Transcribe the following with word-level timestamps:"
COMPARABILITY = "public-reconstruction-proxy"
COMPARABILITY_LIMIT = (
    "The authors did not publish their CTMs, alignment artifacts, or alignment/configuration pins; "
    "this result depends on the caller-pinned local reconstruction and is not paper-equivalent."
)
PROVENANCE_FILENAME = "provenance.json"

SOURCE_SPECS = {
    "librispeech-test-other": {
        "dataset": "LibriSpeech",
        "release": "OpenSLR-12",
        "split": "test-other",
        "language": "en",
    },
    "common-voice-16.1-en": {
        "dataset": "Mozilla Common Voice",
        "release": "16.1",
        "split": "test",
        "language": "en",
    },
    "common-voice-16.1-es": {
        "dataset": "Mozilla Common Voice",
        "release": "16.1",
        "split": "test",
        "language": "es",
    },
    "common-voice-16.1-fr": {
        "dataset": "Mozilla Common Voice",
        "release": "16.1",
        "split": "test",
        "language": "fr",
    },
    "common-voice-16.1-de": {
        "dataset": "Mozilla Common Voice",
        "release": "16.1",
        "split": "test",
        "language": "de",
    },
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAMPLE_ID_RE = re.compile(r"[A-Za-z0-9_.-]+")
_TIMESTAMP_TOKEN_RE = re.compile(r"<\|\d+\|>")
CONTRACT_FIELDS = {
    "schema_version",
    "protocol_version",
    "manifest",
    "manifest_sha256",
    "manifest_rows",
    "sources",
    "alignment",
}
SOURCE_FIELDS = {
    "dataset",
    "release",
    "split",
    "language",
    "rows",
    "source_manifest",
    "source_manifest_sha256",
}
ALIGNMENT_FIELDS = {"tool", "revision", "config", "config_sha256"}
MANIFEST_FIELDS = {
    "id",
    "source",
    "audio_filepath",
    "audio_sha256",
    "audio_bytes",
    "duration",
    "words",
}
WORD_FIELDS = {"word", "start", "end"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_exact_fields(value: object, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"Invalid {context} fields: expected={sorted(expected)}, actual={actual}")
    return value


def _safe_relative_path(value: object, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty relative path")
    pure_path = PurePosixPath(value)
    if pure_path.is_absolute() or "." in pure_path.parts or ".." in pure_path.parts:
        raise ValueError(f"Unsafe {context}: {value!r}")
    return Path(*pure_path.parts)


def _positive_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _load_contract(contract_path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    expected_sha256 = _require_sha256(expected_sha256, "--artifact-contract-sha256")
    actual_sha256 = _sha256(contract_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Artifact contract hash mismatch: expected={expected_sha256}, actual={actual_sha256}")
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read artifact contract {contract_path}: {error}") from error
    _require_exact_fields(contract, CONTRACT_FIELDS, "artifact contract")
    if contract["schema_version"] != ARTIFACT_SCHEMA:
        raise ValueError(f"Unsupported artifact schema: {contract['schema_version']!r}")
    if contract["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported protocol version: {contract['protocol_version']!r}")
    _safe_relative_path(contract["manifest"], "contract manifest")
    _require_sha256(contract["manifest_sha256"], "manifest_sha256")
    manifest_rows = _positive_int(contract["manifest_rows"], "manifest_rows")

    sources = contract["sources"]
    if not isinstance(sources, dict) or set(sources) != set(SOURCE_SPECS):
        actual = sorted(sources) if isinstance(sources, dict) else type(sources).__name__
        raise ValueError(
            f"Artifact must contain exactly the five public protocol sources: "
            f"expected={sorted(SOURCE_SPECS)}, actual={actual}"
        )
    total_source_rows = 0
    for source_key, expected_source in SOURCE_SPECS.items():
        source = _require_exact_fields(sources[source_key], SOURCE_FIELDS, f"source {source_key}")
        for field, expected_value in expected_source.items():
            if source[field] != expected_value:
                raise ValueError(
                    f"Source contract drift for {source_key}.{field}: "
                    f"expected={expected_value!r}, actual={source[field]!r}"
                )
        total_source_rows += _positive_int(source["rows"], f"sources.{source_key}.rows")
        source_manifest_sha256 = _require_sha256(
            source["source_manifest_sha256"],
            f"sources.{source_key}.source_manifest_sha256",
        )
        source_manifest = contract_path.parent / _safe_relative_path(
            source["source_manifest"],
            f"sources.{source_key}.source_manifest",
        )
        if not source_manifest.is_file():
            raise FileNotFoundError(f"Missing pinned source manifest for {source_key}: {source_manifest}")
        actual_source_manifest_sha256 = _sha256(source_manifest)
        if actual_source_manifest_sha256 != source_manifest_sha256:
            raise ValueError(
                f"Source manifest hash mismatch for {source_key}: "
                f"expected={source_manifest_sha256}, actual={actual_source_manifest_sha256}"
            )
    if total_source_rows != manifest_rows:
        raise ValueError(f"Source row total does not match manifest_rows: {total_source_rows} != {manifest_rows}")

    alignment = _require_exact_fields(contract["alignment"], ALIGNMENT_FIELDS, "alignment")
    for field in ("tool", "revision"):
        if not isinstance(alignment[field], str) or not alignment[field].strip():
            raise ValueError(f"alignment.{field} must be a non-empty string")
    config_sha256 = _require_sha256(alignment["config_sha256"], "alignment.config_sha256")
    config_path = contract_path.parent / _safe_relative_path(alignment["config"], "alignment.config")
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing pinned alignment config: {config_path}")
    actual_config_sha256 = _sha256(config_path)
    if actual_config_sha256 != config_sha256:
        raise ValueError(f"Alignment config hash mismatch: expected={config_sha256}, actual={actual_config_sha256}")
    return contract, actual_sha256


def quantize_seconds(seconds: object) -> int:
    """Quantize seconds to 80 ms ticks using explicit round-half-up."""
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float, str, Decimal)):
        raise TypeError(f"Timestamp must be numeric, got {seconds!r}")
    try:
        value = Decimal(str(seconds))
    except InvalidOperation as error:
        raise ValueError(f"Invalid timestamp: {seconds!r}") from error
    if not value.is_finite() or value < 0:
        raise ValueError(f"Timestamp must be finite and non-negative, got {seconds!r}")
    return int((value / TICK_SECONDS).to_integral_value(rounding=ROUND_HALF_UP))


def _validate_row(
    row: object,
    *,
    line_number: int,
    artifact_root: Path,
    seen_ids: set[str],
    seen_audio_paths: set[Path],
) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    row = _require_exact_fields(row, MANIFEST_FIELDS, f"manifest row {line_number}")
    sample_id = row["id"]
    if not isinstance(sample_id, str) or _SAMPLE_ID_RE.fullmatch(sample_id) is None:
        raise ValueError(f"Unsafe sample id at line {line_number}: {sample_id!r}")
    if sample_id in seen_ids:
        raise ValueError(f"Duplicate sample id at line {line_number}: {sample_id!r}")
    seen_ids.add(sample_id)

    source = row["source"]
    if source not in SOURCE_SPECS:
        raise ValueError(f"Unexpected source at line {line_number}: {source!r}")
    relative_audio_path = _safe_relative_path(row["audio_filepath"], f"audio_filepath at line {line_number}")
    if relative_audio_path in seen_audio_paths:
        raise ValueError(f"Duplicate audio_filepath at line {line_number}: {row['audio_filepath']!r}")
    seen_audio_paths.add(relative_audio_path)
    audio_path = artifact_root / relative_audio_path
    expected_audio_sha256 = _require_sha256(row["audio_sha256"], f"audio_sha256 at line {line_number}")
    expected_audio_bytes = _positive_int(row["audio_bytes"], f"audio_bytes at line {line_number}")
    if not audio_path.is_file():
        raise FileNotFoundError(f"Missing pinned audio at line {line_number}: {audio_path}")
    actual_audio_bytes = audio_path.stat().st_size
    if actual_audio_bytes != expected_audio_bytes:
        raise ValueError(
            f"Audio size mismatch at line {line_number}: expected={expected_audio_bytes}, "
            f"actual={actual_audio_bytes}, path={audio_path}"
        )
    actual_audio_sha256 = _sha256(audio_path)
    if actual_audio_sha256 != expected_audio_sha256:
        raise ValueError(
            f"Audio SHA-256 mismatch at line {line_number}: expected={expected_audio_sha256}, "
            f"actual={actual_audio_sha256}, path={audio_path}"
        )

    duration = row["duration"]
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration):
        raise ValueError(f"Invalid duration at line {line_number}: {duration!r}")
    duration = float(duration)
    if duration <= 0:
        raise ValueError(f"Non-positive duration at line {line_number}: {duration!r}")

    words = row["words"]
    if not isinstance(words, list) or not words:
        raise ValueError(f"Row {line_number} must contain non-empty words")
    reference_words = []
    previous_end = 0.0
    for word_index, item in enumerate(words):
        item = _require_exact_fields(item, WORD_FIELDS, f"row {line_number} word {word_index}")
        word = item["word"]
        if not isinstance(word, str) or not word or word.split() != [word] or _TIMESTAMP_TOKEN_RE.search(word):
            raise ValueError(f"Invalid word at row {line_number}, index {word_index}: {word!r}")
        start = item["start"]
        end = item["end"]
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not math.isfinite(start)
            or not math.isfinite(end)
        ):
            raise ValueError(f"Invalid word times at row {line_number}, index {word_index}")
        start = float(start)
        end = float(end)
        if start < previous_end or end <= start or end > duration:
            raise ValueError(
                f"Invalid/non-monotonic word interval at row {line_number}, index {word_index}: "
                f"start={start}, end={end}, previous_end={previous_end}, duration={duration}"
            )
        previous_end = end
        reference_words.append(
            {
                "word": word,
                "start_tick": quantize_seconds(start),
                "end_tick": quantize_seconds(end),
            }
        )

    return row, audio_path, reference_words


def _load_manifest(
    manifest_path: Path,
    *,
    artifact_root: Path,
    expected_sha256: str,
    expected_rows: int,
) -> list[tuple[dict[str, Any], Path, list[dict[str, Any]]]]:
    actual_sha256 = _sha256(manifest_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Artifact manifest hash mismatch: expected={expected_sha256}, actual={actual_sha256}")
    rows = []
    seen_ids: set[str] = set()
    seen_audio_paths: set[Path] = set()
    with manifest_path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                raise ValueError(f"Blank manifest line at {manifest_path}:{line_number}")
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {manifest_path}:{line_number}: {error}") from error
            rows.append(
                _validate_row(
                    row,
                    line_number=line_number,
                    artifact_root=artifact_root,
                    seen_ids=seen_ids,
                    seen_audio_paths=seen_audio_paths,
                )
            )
    if len(rows) != expected_rows:
        raise ValueError(f"Artifact row-count drift: expected={expected_rows}, actual={len(rows)}")
    return rows


def _format_expected_answer(reference_words: list[dict[str, Any]]) -> str:
    return " ".join(f"<|{item['start_tick']}|>{item['word']}<|{item['end_tick']}|>" for item in reference_words)


def _validate_existing_audio(destination: Path, expected_bytes: int, expected_sha256: str) -> None:
    if not destination.is_file():
        raise ValueError(f"Audio destination exists but is not a file: {destination}")
    if destination.stat().st_size != expected_bytes or _sha256(destination) != expected_sha256:
        raise ValueError(f"Pinned destination audio drift: {destination}")


def _materialize_audio(source: Path, destination: Path, *, expected_bytes: int, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.resolve() == destination.resolve():
            _validate_existing_audio(destination, expected_bytes, expected_sha256)
            return
        _validate_existing_audio(destination, expected_bytes, expected_sha256)
        return
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    shutil.copyfile(source, partial)
    _validate_existing_audio(partial, expected_bytes, expected_sha256)
    partial.replace(destination)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    partial.replace(path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    partial.replace(path)


def prepare_artifact(
    *,
    artifact_contract: Path,
    artifact_contract_sha256: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Validate and materialize one exact caller-pinned reconstruction."""
    artifact_contract = artifact_contract.resolve()
    artifact_root = artifact_contract.parent
    contract, contract_sha256 = _load_contract(artifact_contract, artifact_contract_sha256)
    manifest_path = artifact_root / _safe_relative_path(contract["manifest"], "contract manifest")
    rows = _load_manifest(
        manifest_path,
        artifact_root=artifact_root,
        expected_sha256=contract["manifest_sha256"],
        expected_rows=contract["manifest_rows"],
    )
    source_counts = Counter(row[0]["source"] for row in rows)
    expected_counts = {source: details["rows"] for source, details in contract["sources"].items()}
    if dict(source_counts) != expected_counts:
        raise ValueError(f"Per-source row-count drift: expected={expected_counts}, actual={dict(source_counts)}")

    output_dir = output_dir.resolve()
    audio_root = os.getenv("NEMO_SKILLS_AUDIO_ROOT", "/data").rstrip("/")
    output_rows = []
    for row, source_audio, reference_words in rows:
        source_key = row["source"]
        suffix = Path(row["audio_filepath"]).suffix
        if not suffix:
            raise ValueError(f"Audio path has no suffix: {row['audio_filepath']!r}")
        output_audio = output_dir / "audio" / source_key / f"{row['id']}{suffix}"
        _materialize_audio(
            source_audio,
            output_audio,
            expected_bytes=row["audio_bytes"],
            expected_sha256=row["audio_sha256"],
        )
        container_audio = f"{audio_root}/{BENCHMARK_NAME}/audio/{source_key}/{row['id']}{suffix}"
        duration = float(row["duration"])
        audio = {"path": container_audio, "duration": duration}
        expected_answer = _format_expected_answer(reference_words)
        source_contract = contract["sources"][source_key]
        output_rows.append(
            {
                "id": row["id"],
                "sample_id": row["id"],
                "task_type": "ASR-word-timestamp-reconstruction",
                "audio_filepath": container_audio,
                "duration": duration,
                "text": " ".join(item["word"] for item in reference_words),
                "expected_answer": expected_answer,
                "reference_words": reference_words,
                "subset_for_metrics": source_key,
                "protocol_profile": PROTOCOL_VERSION,
                "comparability": COMPARABILITY,
                "comparability_limit": COMPARABILITY_LIMIT,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": USER_PROMPT,
                        "audio": audio,
                        "audios": [audio],
                    },
                ],
                "source_provenance": {
                    "artifact_contract_sha256": contract_sha256,
                    "artifact_manifest_sha256": contract["manifest_sha256"],
                    "source_manifest_sha256": source_contract["source_manifest_sha256"],
                    "dataset": source_contract["dataset"],
                    "release": source_contract["release"],
                    "split": source_contract["split"],
                    "language": source_contract["language"],
                    "alignment_tool": contract["alignment"]["tool"],
                    "alignment_revision": contract["alignment"]["revision"],
                    "alignment_config_sha256": contract["alignment"]["config_sha256"],
                    "quantization": "80ms-round-half-up",
                    "true_positive_tolerance": "start_error<240ms-and-end_error<240ms",
                },
            }
        )

    _write_jsonl_atomic(output_dir / "test.jsonl", output_rows)
    _write_json_atomic(
        output_dir / PROVENANCE_FILENAME,
        {
            "artifact_schema": ARTIFACT_SCHEMA,
            "protocol_profile": PROTOCOL_VERSION,
            "comparability": COMPARABILITY,
            "comparability_limit": COMPARABILITY_LIMIT,
            "artifact_contract": str(artifact_contract),
            "artifact_contract_sha256": contract_sha256,
            "artifact_manifest_sha256": contract["manifest_sha256"],
            "manifest_rows": len(output_rows),
            "source_rows": dict(source_counts),
            "sources": contract["sources"],
            "alignment": contract["alignment"],
            "timestamp_tick_ms": TICK_MILLISECONDS,
            "true_positive_tolerance_ms": STRICT_TOLERANCE_MILLISECONDS,
            "true_positive_tolerance_operator": "<",
            "quantization_rounding": "ROUND_HALF_UP",
        },
    )
    return output_rows


def _default_output_dir() -> Path:
    data_root = os.getenv("NEMO_SKILLS_DATA_DIR")
    if data_root:
        return Path(data_root) / BENCHMARK_NAME
    package_dir = Path(__file__).parent
    if "site-packages" in str(package_dir) or "dist-packages" in str(package_dir):
        raise SystemExit("Missing --output-dir and NEMO_SKILLS_DATA_DIR. Refusing to write into an installed package.")
    return package_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-contract", type=Path, required=True)
    parser.add_argument("--artifact-contract-sha256", required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or _default_output_dir()
    rows = prepare_artifact(
        artifact_contract=args.artifact_contract,
        artifact_contract_sha256=args.artifact_contract_sha256,
        output_dir=output_dir,
    )
    print(f"Prepared {len(rows)} pinned {COMPARABILITY} rows at {output_dir}")
    print(f"LIMITATION: {COMPARABILITY_LIMIT}")


if __name__ == "__main__":
    main()
