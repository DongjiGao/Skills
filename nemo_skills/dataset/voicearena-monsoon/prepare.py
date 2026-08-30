# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Prepare pinned public VoiceArena Monsoon ASR self-scoring splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

BENCHMARK_NAME = "voicearena-monsoon"
SOURCE_LICENSE = "CC-BY-4.0"
OFFICIAL_SPLIT = "test"
EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHANNELS = 1
EXPECTED_SUBTYPE = "PCM_16"
MINIMUM_AUDIO_SAMPLES = 320
PROVENANCE_FILENAME = "provenance.json"
PROMPT = "Transcribe the following:"
_SAFE_ID = re.compile(r"[A-Za-z0-9._-]+")


@dataclass(frozen=True)
class SourceContract:
    profile: str
    language: str
    source_language: str
    repo: str
    revision: str
    expected_rows: int
    expected_speakers: int
    reference_field: str
    official_metric: str


SOURCE_CONTRACTS = {
    "en-in.public": SourceContract(
        profile="en-in.public",
        language="en-in",
        source_language="en_in",
        repo="VoiceArena/MonsoonASR-Open-ASR-leaderboard-en-IN",
        revision="bc1da7b42ef6e2853123c97bf6d22067e4802d11",
        expected_rows=2102,
        expected_speakers=1444,
        reference_field="text",
        official_metric="corpus_micro_wer",
    ),
    "hi-in.public": SourceContract(
        profile="hi-in.public",
        language="hi-in",
        source_language="hi_in",
        repo="VoiceArena/MonsoonASR-Open-ASR-leaderboard-hi-IN",
        revision="a4b7375e9adea16c16100295976cdb0aabd9fb6d",
        expected_rows=753,
        expected_speakers=468,
        reference_field="lattice",
        official_metric="corpus_micro_oiwer",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_data_root() -> Path:
    if os.environ.get("NEMO_SKILLS_DATA_DIR"):
        return Path(os.environ["NEMO_SKILLS_DATA_DIR"]) / BENCHMARK_NAME
    return Path.home() / ".cache" / "nemo-skills-data" / BENCHMARK_NAME


def _manifest_dir(contract: SourceContract) -> Path:
    language, visibility = contract.profile.split(".", 1)
    return Path(__file__).parent / language / visibility


def _audio_prefix(contract: SourceContract) -> str:
    audio_root = os.environ.get("NEMO_SKILLS_AUDIO_ROOT", "/data").rstrip("/")
    language, visibility = contract.profile.split(".", 1)
    return f"{audio_root}/{BENCHMARK_NAME}/{language}/{visibility}/audio"


def _validate_lattice(value: object, row_number: int) -> list[list[str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Monsoon hi-IN row {row_number} has an invalid lattice")
    validated: list[list[str]] = []
    for slot in value:
        if not isinstance(slot, list) or not slot:
            raise ValueError(f"Monsoon hi-IN row {row_number} has an empty/non-list lattice slot")
        variants = [variant.strip() for variant in slot if isinstance(variant, str) and variant.strip()]
        if not variants or len(variants) != len(slot):
            raise ValueError(f"Monsoon hi-IN row {row_number} has an invalid lattice variant")
        validated.append(variants)
    return validated


def _validate_metadata(rows: list[dict], contract: SourceContract) -> list[dict]:
    if len(rows) != contract.expected_rows:
        raise ValueError(
            f"Monsoon {contract.language} row-count drift: expected {contract.expected_rows}, found {len(rows)}"
        )
    seen_ids: set[str] = set()
    speakers: set[int] = set()
    validated: list[dict] = []
    for row_number, source in enumerate(rows, start=1):
        row = dict(source)
        source_id = row.get("id")
        if not isinstance(source_id, str) or _SAFE_ID.fullmatch(source_id) is None:
            raise ValueError(f"Monsoon {contract.language} row {row_number} has unsafe id: {source_id!r}")
        if source_id in seen_ids:
            raise ValueError(f"Monsoon {contract.language} has duplicate id: {source_id}")
        seen_ids.add(source_id)
        speaker = row.get("speaker_id")
        if isinstance(speaker, bool) or not isinstance(speaker, int):
            raise TypeError(f"Monsoon {contract.language} row {row_number} has invalid speaker_id")
        speakers.add(speaker)
        if row.get("language") != contract.source_language:
            raise ValueError(f"Monsoon {contract.language} row {row_number} has unexpected language")
        duration = row.get("audio_length_s")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
            raise ValueError(f"Monsoon {contract.language} row {row_number} has invalid duration")
        if contract.reference_field == "text":
            reference = row.get("text")
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError(f"Monsoon en-IN row {row_number} has an empty reference")
        else:
            row["lattice"] = _validate_lattice(row.get("lattice"), row_number)
        validated.append(row)
    if len(speakers) != contract.expected_speakers:
        raise ValueError(
            f"Monsoon {contract.language} speaker-count drift: "
            f"expected {contract.expected_speakers}, found {len(speakers)}"
        )
    return validated


def _decode_audio(value: object) -> tuple[np.ndarray, int]:
    if isinstance(value, dict):
        samples = np.asarray(value["array"])
        sample_rate = int(value["sampling_rate"])
    elif hasattr(value, "get_all_samples"):
        decoded = value.get_all_samples()
        samples = np.asarray(decoded.data)
        sample_rate = int(decoded.sample_rate)
    else:
        raise TypeError(f"Unsupported Hugging Face audio value: {type(value)!r}")
    samples = np.squeeze(samples)
    if samples.ndim != 1 or samples.size < MINIMUM_AUDIO_SAMPLES:
        raise ValueError(f"Monsoon source audio must be nonempty mono; got shape={samples.shape}")
    if sample_rate != EXPECTED_SAMPLE_RATE:
        raise ValueError(f"Monsoon source audio sample-rate drift: expected 16000, got {sample_rate}")
    return samples.astype(np.float32, copy=False), sample_rate


def _validate_wav(path: Path, expected_duration: float) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Monsoon audio is missing: {path}")
    try:
        with sf.SoundFile(str(path)) as stream:
            observed = (stream.samplerate, stream.channels, stream.subtype)
            expected = (EXPECTED_SAMPLE_RATE, EXPECTED_CHANNELS, EXPECTED_SUBTYPE)
            if observed != expected or stream.frames < MINIMUM_AUDIO_SAMPLES:
                raise ValueError(f"Monsoon audio contract failed for {path}: {observed}, frames={stream.frames}")
            duration = stream.frames / stream.samplerate
    except RuntimeError as error:
        raise ValueError(f"Monsoon audio has an unreadable header: {path}") from error
    if abs(duration - expected_duration) > 0.11:
        raise ValueError(f"Monsoon audio duration drift for {path}: metadata={expected_duration}, WAV={duration}")


def _materialize_audio(dataset, rows: list[dict], contract: SourceContract, data_root: Path, no_audio: bool) -> None:
    audio_dir = data_root.joinpath(*contract.profile.split("."), "audio")
    audio_dir.mkdir(parents=True, exist_ok=True)
    if len(dataset) != len(rows):
        raise ValueError("Monsoon decoded/metadata row counts diverged")
    for index, metadata in enumerate(rows):
        path = audio_dir / f"{metadata['id']}.wav"
        if not no_audio:
            samples, sample_rate = _decode_audio(dataset[index]["audio"])
            temporary = path.with_name(f".{path.name}.partial")
            temporary.unlink(missing_ok=True)
            try:
                sf.write(temporary, samples, sample_rate, format="WAV", subtype=EXPECTED_SUBTYPE)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        _validate_wav(path, float(metadata["audio_length_s"]))


def _build_record(row: dict, contract: SourceContract) -> dict:
    audio_path = f"{_audio_prefix(contract)}/{row['id']}.wav"
    if contract.reference_field == "text":
        expected_answer = row["text"].strip()
        lattice = None
    else:
        lattice = row["lattice"]
        expected_answer = " ".join(slot[0] for slot in lattice)
    record = {
        "id": f"{BENCHMARK_NAME}-{contract.profile}-{row['id']}",
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
                "audio": {"path": audio_path, "duration": float(row["audio_length_s"])},
            }
        ],
        "expected_answer": expected_answer,
        "audio_filepath": audio_path,
        "audio_duration": float(row["audio_length_s"]),
        "monsoon_language": contract.language,
        "speaker_id": row["speaker_id"],
        "subset_for_metrics": contract.profile,
        "extra_fields": {
            "source_repo": contract.repo,
            "source_revision": contract.revision,
            "source_split": OFFICIAL_SPLIT,
            "source_license": SOURCE_LICENSE,
            "source_id": row["id"],
        },
    }
    if lattice is not None:
        record["reference_lattice"] = lattice
    return record


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as output:
            temporary = Path(output.name)
            json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_profile(contract: SourceContract, data_root: Path, no_audio: bool = False) -> Path:
    from datasets import load_dataset

    dataset = load_dataset(contract.repo, split=OFFICIAL_SPLIT, revision=contract.revision)
    required = {"audio", "id", contract.reference_field, "speaker_id", "language", "audio_length_s"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Monsoon {contract.language} source fields missing: {sorted(missing)}")
    metadata_dataset = dataset.remove_columns("audio")
    rows = _validate_metadata([dict(row) for row in metadata_dataset], contract)
    _materialize_audio(dataset, rows, contract, data_root, no_audio)

    manifest_dir = _manifest_dir(contract)
    output_path = manifest_dir / "test.jsonl"
    temporary: Path | None = None
    try:
        manifest_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=manifest_dir, prefix=".test.", suffix=".jsonl.tmp", delete=False
        ) as output:
            temporary = Path(output.name)
            for row in rows:
                output.write(json.dumps(_build_record(row, contract), ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        manifest_sha256 = sha256_file(temporary)
        os.replace(temporary, output_path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    receipt = {
        "schema_version": 1,
        "dataset": BENCHMARK_NAME,
        "profile": contract.profile,
        "manifest": output_path.name,
        "manifest_sha256": manifest_sha256,
        "source": asdict(contract),
        "source_split": OFFICIAL_SPLIT,
        "source_license": SOURCE_LICENSE,
        "audio_contract": {
            "sample_rate": EXPECTED_SAMPLE_RATE,
            "channels": EXPECTED_CHANNELS,
            "subtype": EXPECTED_SUBTYPE,
            "minimum_samples": MINIMUM_AUDIO_SAMPLES,
            "complete_closure": True,
        },
        "quality_scoring_contract": {
            "official_metric": contract.official_metric,
            "aggregate": "sum_errors_divided_by_sum_reference_words",
            "oiwer_distribution": "voi-oiwer==0.1.4" if contract.language == "hi-in" else None,
        },
    }
    _atomic_json(manifest_dir / PROVENANCE_FILENAME, receipt)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--profile", action="append", choices=sorted(SOURCE_CONTRACTS))
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args()
    data_root = args.data_root or _default_data_root()
    profiles = args.profile or list(SOURCE_CONTRACTS)
    for profile in profiles:
        output_path = prepare_profile(SOURCE_CONTRACTS[profile], data_root, no_audio=args.no_audio)
        print(f"Wrote pinned Monsoon {profile} manifest to {output_path}")


if __name__ == "__main__":
    main()
