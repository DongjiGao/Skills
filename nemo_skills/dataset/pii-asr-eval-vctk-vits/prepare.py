# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Prepare the fixed synthetic VCTK-VITS PII redaction evaluation set.

Only canonical metadata and referenced WAVs are read. Prepared output contains
only ``test.jsonl``, ``provenance.json``, and the exact audio closure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import soundfile as sf

BENCHMARK_NAME = "pii-asr-eval-vctk-vits"
TWO_STAGE_BENCHMARK_NAME = "pii-asr-eval-vctk-vits-two-stage"
TWO_STAGE_PII_AWARE_BENCHMARK_NAME = "pii-asr-eval-vctk-vits-two-stage-pii-aware"
SCHEMA_VERSION = "pii-asr-eval-vctk-vits-v1"
BUILD_ID = "pii-asr-eval-vctk-vits-20260831"
SOURCE_METADATA_FILENAME = "metadata.jsonl"
SOURCE_METADATA_SHA256 = "c4eaf9a8926377a583c5a221551e79c92fa81a54ba55a267575db37925b53ce0"
SOURCE_BUILD_SHA256 = "7da89a19616dc463caa09ce14c038c97893c45e93f1406c89b47399d484ee7d9"
EXPECTED_ROWS = 1_000
EXPECTED_ENTITIES = 1_080
EXPECTED_CATEGORIES = 25
EXPECTED_ROWS_PER_CATEGORY = 40
EXPECTED_SPEAKERS = 108
EXPECTED_DURATION_SECONDS = 8815.804081632654
EXPECTED_SAMPLE_RATE = 22_050
EXPECTED_CHANNELS = 1
EXPECTED_FORMAT = "WAV"
EXPECTED_SUBTYPE = "PCM_16"
SYSTEM_PROMPT = "You are a helpful assistant. /no_think"
USER_PROMPT = (
    "Transcribe the audio with punctuation and capitalization, and redact only high-risk "
    "private identifiers: email and phone/fax; complete street addresses; payment cards "
    "and bank accounts; passwords, PINs, API keys, and similar credentials; and government "
    "or medical identifiers. Do not redact names, ordinary dates, generic places, URLs, or "
    "public business details unless they are part of a listed high-risk identifier. Replace "
    "each redacted span with its lower-case hyphenated type in angle brackets, and return "
    "only the redacted transcript."
)
TRANSCRIBE_PROMPT = "Produce a verbatim transcript of the audio."
PII_AWARE_TRANSCRIBE_PROMPT = (
    "Produce an exact, verbatim transcript of the audio. The audio may contain PII that a later second pass "
    "must redact, so accurately preserve every spoken letter, digit, separator, and formatting word. Do not "
    "redact, classify, normalize, or replace anything with placeholders in this pass. Possible PII categories "
    "and synthetic spoken examples are: API key (test dash B V U Y dash zero zero two six dash N L Q); bank "
    "account (test account zero zero zero zero three nine zero five one four); bank routing number (nine nine six "
    "nine seven four nine three two); card security code or CVV (one zero one); certificate or license number "
    "(test license B U H zero zero three nine); credit or debit card (nine nine nine nine, zero zero one five, "
    "zero five five five, one zero six eight); customer ID (customer test J N zero zero zero three); spelled "
    "email (D A H L dot amber at example dot org); spoken email (pebble dot lantern zero zero three four at "
    "example dot com); employee ID (employee test J W zero zero zero two); fax number (eight zero eight five five "
    "five zero one three nine); health-plan beneficiary number (H P B test H A zero zero two four); HTTP cookie "
    "(session underscore test equals cedar dash zero zero two four); medical-record number (M R N test zero zero "
    "three six N P); mixed identifiers (customer ID plus phone number plus email); national ID (test N I D Q N "
    "zero zero three eight); password (river five cedar hash five); domestic phone (plus one five zero five five "
    "five five zero one zero eight); international phone (plus nine nine nine seven zero zero zero zero zero zero "
    "zero one seven); PIN (one seven seven seven); Social Security number or SSN (nine nine nine one one three "
    "seven seven four); UK street address (flat two one eight Juniper Example Mews Sampleton Q Q zero zero Q Q); "
    "US street address (one zero zero three two Juniper Archive Lane Testville Arcadia zero zero zero zero zero); "
    "SWIFT or BIC (T E S T U S zero zero Z F Q); and tax ID (zero zero eight one nine seven six six four). Return "
    "only the transcript."
)
REDACTION_FOLLOWUP_PROMPT = (
    "Now redact only high-risk private identifiers in the transcript: email and phone/fax; "
    "complete street addresses; payment cards and bank accounts; passwords, PINs, API keys, "
    "and similar credentials; and government or medical identifiers. Do not redact names, "
    "ordinary dates, generic places, URLs, or public business details unless they are part "
    "of a listed high-risk identifier. Replace each redacted span with its lower-case "
    "hyphenated type in angle brackets, and return only the redacted transcript."
)
PROVENANCE_FILENAME = "provenance.json"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PLACEHOLDER_RE = re.compile(r"<[a-z]+(?:-[a-z]+)*>")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_source_rows(source_dir: Path) -> list[dict[str, Any]]:
    metadata_path = source_dir / SOURCE_METADATA_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(f"PII-ASR metadata is missing: {metadata_path}")
    actual_sha256 = sha256_file(metadata_path)
    if actual_sha256 != SOURCE_METADATA_SHA256:
        raise ValueError(f"PII-ASR metadata hash drift: expected {SOURCE_METADATA_SHA256}, got {actual_sha256}")
    rows: list[dict[str, Any]] = []
    with metadata_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in PII-ASR metadata line {line_number}") from error
            if not isinstance(row, dict):
                raise TypeError(f"PII-ASR metadata line {line_number} is not an object")
            rows.append(row)
    return rows


def _validate_rows(rows: list[dict[str, Any]], source_dir: Path) -> list[tuple[dict[str, Any], Path]]:
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"PII-ASR row-count drift: expected {EXPECTED_ROWS}, got {len(rows)}")
    ids: set[str] = set()
    speakers: set[str] = set()
    categories: Counter[str] = Counter()
    entity_count = 0
    total_duration = 0.0
    validated: list[tuple[dict[str, Any], Path]] = []
    for row_number, row in enumerate(rows, start=1):
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in ids:
            raise ValueError(f"PII-ASR row {row_number} has invalid/duplicate id: {sample_id!r}")
        ids.add(sample_id)
        if row.get("sample_rate") != EXPECTED_SAMPLE_RATE:
            raise ValueError(f"PII-ASR row {sample_id} has unexpected sample rate: {row.get('sample_rate')!r}")
        category = row.get("category")
        speaker_id = row.get("speaker_id")
        if not isinstance(category, str) or not category or not isinstance(speaker_id, str) or not speaker_id:
            raise ValueError(f"PII-ASR row {sample_id} has invalid category/speaker")
        categories[category] += 1
        speakers.add(speaker_id)
        transcript = row.get("transcript")
        redacted_text = row.get("redacted_text")
        entities = row.get("entities")
        if (
            not isinstance(transcript, str)
            or not transcript
            or not isinstance(redacted_text, str)
            or not redacted_text
        ):
            raise ValueError(f"PII-ASR row {sample_id} has invalid transcript text")
        if not isinstance(entities, list) or not entities:
            raise ValueError(f"PII-ASR row {sample_id} has no entity annotations")
        expected_placeholders = []
        for entity in entities:
            start, end = int(entity["start"]), int(entity["end"])
            if not 0 <= start < end <= len(transcript) or transcript[start:end] != entity.get("text"):
                raise ValueError(f"PII-ASR row {sample_id} has an invalid entity span")
            placeholder = entity.get("placeholder")
            if not isinstance(placeholder, str) or _PLACEHOLDER_RE.fullmatch(placeholder) is None:
                raise ValueError(f"PII-ASR row {sample_id} has an invalid placeholder: {placeholder!r}")
            expected_placeholders.append(placeholder)
        if Counter(_PLACEHOLDER_RE.findall(redacted_text)) != Counter(expected_placeholders):
            raise ValueError(f"PII-ASR row {sample_id} redacted target disagrees with entity placeholders")
        entity_count += len(entities)
        duration = row.get("duration")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
            raise ValueError(f"PII-ASR row {sample_id} has invalid duration: {duration!r}")
        total_duration += float(duration)
        audio_path = source_dir / "audio" / f"{sample_id}.wav"
        if not audio_path.is_file():
            raise FileNotFoundError(f"PII-ASR audio is missing: {audio_path}")
        expected_sha256 = row.get("audio_sha256")
        if not isinstance(expected_sha256, str) or _SHA256_RE.fullmatch(expected_sha256) is None:
            raise ValueError(f"PII-ASR row {sample_id} has invalid audio SHA-256")
        if sha256_file(audio_path) != expected_sha256:
            raise ValueError(f"PII-ASR audio hash mismatch: {audio_path}")
        try:
            with sf.SoundFile(str(audio_path)) as stream:
                observed = (stream.format, stream.subtype, stream.samplerate, stream.channels)
                expected = (EXPECTED_FORMAT, EXPECTED_SUBTYPE, EXPECTED_SAMPLE_RATE, EXPECTED_CHANNELS)
                if observed != expected:
                    raise ValueError(
                        f"PII-ASR audio format drift for {audio_path}: expected {expected}, got {observed}"
                    )
                actual_duration = stream.frames / stream.samplerate
                if stream.frames < 320:
                    raise ValueError(f"PII-ASR audio is shorter than 320 samples: {audio_path}")
        except RuntimeError as error:
            raise ValueError(f"PII-ASR audio has an unreadable WAV header: {audio_path}") from error
        if abs(actual_duration - float(duration)) > 1.0 / EXPECTED_SAMPLE_RATE:
            raise ValueError(f"PII-ASR duration drift for {audio_path}: metadata={duration}, WAV={actual_duration}")
        validated.append((row, audio_path))
    if len(categories) != EXPECTED_CATEGORIES or set(categories.values()) != {EXPECTED_ROWS_PER_CATEGORY}:
        raise ValueError(f"PII-ASR category contract drift: {dict(sorted(categories.items()))}")
    if len(speakers) != EXPECTED_SPEAKERS:
        raise ValueError(f"PII-ASR speaker-count drift: expected {EXPECTED_SPEAKERS}, got {len(speakers)}")
    if entity_count != EXPECTED_ENTITIES:
        raise ValueError(f"PII-ASR entity-count drift: expected {EXPECTED_ENTITIES}, got {entity_count}")
    if abs(total_duration - EXPECTED_DURATION_SECONDS) > 1e-6:
        raise ValueError(f"PII-ASR total-duration drift: expected {EXPECTED_DURATION_SECONDS}, got {total_duration}")
    return validated


def _publish_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_sha256 = sha256_file(source)
    if destination.is_file() and sha256_file(destination) == source_sha256:
        return
    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.unlink(missing_ok=True)
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _record(row: dict[str, Any], audio_prefix: str, scenario: str = "one-shot") -> dict[str, Any]:
    audio_path = f"{audio_prefix.rstrip('/')}/audio/{row['id']}.wav"
    if scenario == "one-shot":
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT, "audio": {"path": audio_path, "duration": row["duration"]}},
        ]
        scenario_fields: dict[str, Any] = {}
        task_type = "PII-ASR-REDACTION"
    elif scenario in {"two-stage", "two-stage-pii-aware"}:
        transcribe_prompt = TRANSCRIBE_PROMPT if scenario == "two-stage" else PII_AWARE_TRANSCRIBE_PROMPT
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcribe_prompt, "audio": {"path": audio_path, "duration": row["duration"]}},
        ]
        scenario_fields = {"redaction_prompt": REDACTION_FOLLOWUP_PROMPT, "expected_asr": row["transcript"]}
        task_type = (
            "PII-ASR-TWO-STAGE-REDACTION" if scenario == "two-stage" else "PII-ASR-TWO-STAGE-PII-AWARE-REDACTION"
        )
    else:
        raise ValueError(f"Unsupported PII-ASR scenario: {scenario!r}")
    return {
        "id": row["id"],
        "messages": messages,
        "expected_answer": row["redacted_text"],
        "expected_placeholders": [entity["placeholder"] for entity in row["entities"]],
        "pii_entity_texts": [entity["text"] for entity in row["entities"]],
        "audio_filepath": audio_path,
        "audio_duration": row["duration"],
        "subset_for_metrics": row["category"],
        "task_type": task_type,
        "extra_fields": {
            "schema_version": SCHEMA_VERSION,
            "build_id": BUILD_ID,
            "speaker_id": row["speaker_id"],
            "category": row["category"],
            "source_audio_sha256": row["audio_sha256"],
            "synthetic_only": True,
        },
        **scenario_fields,
    }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_dataset(
    source_dir: Path, output_dir: Path, audio_prefix: str, scenario: str = "one-shot", publish_audio: bool = True
) -> Path:
    validated = _validate_rows(_load_source_rows(source_dir), source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if publish_audio:
        for _row, source_audio in validated:
            _publish_file(source_audio, output_dir / "audio" / source_audio.name)
    manifest_path = output_dir / "test.jsonl"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=output_dir, prefix=".test.", suffix=".jsonl.tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            for row, _audio in validated:
                stream.write(json.dumps(_record(row, audio_prefix, scenario=scenario), ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        manifest_sha256 = sha256_file(temporary)
        os.replace(temporary, manifest_path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    dataset_name = {
        "one-shot": BENCHMARK_NAME,
        "two-stage": TWO_STAGE_BENCHMARK_NAME,
        "two-stage-pii-aware": TWO_STAGE_PII_AWARE_BENCHMARK_NAME,
    }[scenario]
    task = {
        "one-shot": "single-shot speech-to-redacted-transcript",
        "two-stage": "model-ASR-then-model-redaction",
        "two-stage-pii-aware": "PII-aware-model-ASR-then-model-redaction",
    }[scenario]
    _write_json_atomic(
        output_dir / PROVENANCE_FILENAME,
        {
            "schema_version": 1,
            "dataset": dataset_name,
            "build_id": BUILD_ID,
            "source_schema_version": SCHEMA_VERSION,
            "source_metadata_sha256": SOURCE_METADATA_SHA256,
            "source_build_sha256": SOURCE_BUILD_SHA256,
            "manifest": manifest_path.name,
            "manifest_sha256": manifest_sha256,
            "rows": EXPECTED_ROWS,
            "entities": EXPECTED_ENTITIES,
            "categories": EXPECTED_CATEGORIES,
            "speakers": EXPECTED_SPEAKERS,
            "duration_seconds": EXPECTED_DURATION_SECONDS,
            "audio_files": EXPECTED_ROWS,
            "audio_prefix": audio_prefix,
            "audio_published_here": publish_audio,
            "audio_contract": {
                "format": EXPECTED_FORMAT,
                "subtype": EXPECTED_SUBTYPE,
                "sample_rate": EXPECTED_SAMPLE_RATE,
                "channels": EXPECTED_CHANNELS,
                "per_item_sha256": True,
            },
            "task": task,
            "synthetic_only": True,
        },
    )
    return manifest_path


def main(default_scenario: str = "one-shot") -> None:
    parser = argparse.ArgumentParser(description="Prepare the fixed VCTK-VITS PII-ASR redaction benchmark")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--data_dir", default=os.getenv("NEMO_SKILLS_DATA_DIR"))
    parser.add_argument(
        "--scenario", choices=("one-shot", "two-stage", "two-stage-pii-aware"), default=default_scenario
    )
    parser.add_argument("--audio-prefix")
    parser.add_argument("--skip-audio-copy", action="store_true")
    args = parser.parse_args()
    if not args.data_dir:
        raise SystemExit("Missing --data_dir and NEMO_SKILLS_DATA_DIR; refusing to write into staged code")
    dataset_name = {
        "one-shot": BENCHMARK_NAME,
        "two-stage": TWO_STAGE_BENCHMARK_NAME,
        "two-stage-pii-aware": TWO_STAGE_PII_AWARE_BENCHMARK_NAME,
    }[args.scenario]
    audio_prefix = args.audio_prefix or f"/data/{dataset_name}"
    print(
        f"Prepared {prepare_dataset(args.source_dir, Path(args.data_dir) / dataset_name, audio_prefix, scenario=args.scenario, publish_audio=not args.skip_audio_copy)}"
    )


if __name__ == "__main__":
    main()
