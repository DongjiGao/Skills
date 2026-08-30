# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Prepare the public MUCS 2021 Subtask 2 blind-test release.

The source is the official OpenSLR SLR104 archive. Preparation verifies the
archive byte hash, complete member inventory, released Kaldi metadata hashes,
row/recording/empty-reference contracts, packaged scorer assets, and audio
format/boundaries. It then slices the released recording WAVs at the official
``segments`` boundaries without resampling. All temporary and durable files
remain beneath the requested data directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import soundfile as sf

from nemo_skills.dataset.mucs2021.scorer import ASSET_CONTRACTS, ASSETS_DIR

SOURCE_URL = "https://www.openslr.org/resources/104/subtask2_blindtest_wReadme.tar.gz"
SOURCE_SHA256 = "b134ba007e6a310a2ca2a89619257ea6e7de744e135043c8c517cc05a404889a"
SOURCE_BYTES = 1_008_263_116
ARCHIVE_NAME = "subtask2_blindtest_wReadme.tar.gz"
ARCHIVE_ROOT = "subtask2_blindtest_wReadme"
ARCHIVE_MEMBER_COUNT = 88
PROMPT = "Transcribe the following audio."


@dataclass(frozen=True)
class LanguageContract:
    leaf: str
    source_dir: str
    rows: int
    recordings: int
    empty_references: int
    metadata_sha256: dict[str, str]
    word_mapping_asset: str


LANGUAGE_CONTRACTS = {
    "hindi-english": LanguageContract(
        leaf="hindi-english",
        source_dir="hindi",
        rows=4034,
        recordings=35,
        empty_references=8,
        metadata_sha256={
            "text": "83eab2b9420e252474964e2b0e916412a823b0754d04910d808a93a8832a69fe",
            "segments": "c1c7891dbe65af9013a28a19a59498921d0e31e8a0516b28f113724cb7189d3c",
            "wav.scp": "36a7be055cc3595ccc152d8529a6ee067aa69537b77ad0562a94e539dba7284d",
            "utt2spk": "861cdf50f2c884c4c35bcd775af90a5c288ff026da353e39e6b8480e584114ef",
            "spk2utt": "11607c701e8dbe74abdc7d6a774377e38127607c3c51af2135779e1e6011eef7",
        },
        word_mapping_asset="hin2engwords.txt",
    ),
    "bengali-english": LanguageContract(
        leaf="bengali-english",
        source_dir="bengali",
        rows=3130,
        recordings=32,
        empty_references=4,
        metadata_sha256={
            "text": "dbb1308dd8f2a598cf9334a28dfea6ed498871e7ff4f60464989dd5989cd5326",
            "segments": "6ca79ae4155156b472bf2371e91583692480ba69f2c59211e6a204ff7fc64c50",
            "wav.scp": "4cb4a73e736ae5f7611d639fe10236efb1ee925e98cf0b7d61b6165342759bb7",
            "utt2spk": "95a09f27cfc8ced6c735b3d0a443cb2a17478c9c96dc2c00156fc2c318e37427",
            "spk2utt": "c541345b9f3cd3c67252b34b18756d0f0486049df390ce3d823fb5fafffb01d4",
        },
        word_mapping_asset="ben2engwords.txt",
    ),
}
SMALL_SOURCE_CONTRACTS = {
    f"inference/{name if name != 'official_compute_wer.py' else 'compute_wer.py'}": sha
    for name, (sha, _) in ASSET_CONTRACTS.items()
}
SMALL_SOURCE_CONTRACTS["readme"] = "9f3d7ccdbbf6352a0a812fb924430b2da481e84b10c1161ff66201478f7991cf"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_archive(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _validate_archive_file(destination)
        return destination
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "nemo-skills/mucs2021"})
    with urllib.request.urlopen(request, timeout=300) as source, partial.open("wb") as output:
        shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    _validate_archive_file(partial)
    os.replace(partial, destination)
    return destination


def _validate_archive_file(path: Path) -> None:
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != SOURCE_BYTES or digest != SOURCE_SHA256:
        raise ValueError(
            f"MUCS source archive contract changed at {path}: bytes={size}, sha256={digest}; "
            f"expected bytes={SOURCE_BYTES}, sha256={SOURCE_SHA256}"
        )


def _extract_regular_member(archive: tarfile.TarFile, member: tarfile.TarInfo, destination: Path) -> Path:
    if not member.isfile():
        raise ValueError(f"Expected regular archive member, found {member.name!r}")
    relative = Path(member.name).relative_to(ARCHIVE_ROOT)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe archive path: {member.name!r}")
    target = destination / ARCHIVE_ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"Could not read archive member: {member.name}")
    with source, target.open("wb") as output:
        shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    return target


def _parse_kaldi_text(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, raw in enumerate(source, start=1):
            line = raw.rstrip("\r\n")
            if not line:
                raise ValueError(f"Blank Kaldi text row at {path}:{line_number}")
            first_space = next((i for i, character in enumerate(line) if character.isspace()), len(line))
            utterance_id = line[:first_space]
            reference = line[first_space + 1 :] if first_space < len(line) else ""
            if not utterance_id or utterance_id in rows:
                raise ValueError(f"Invalid/duplicate utterance ID at {path}:{line_number}: {utterance_id!r}")
            rows[utterance_id] = reference
    return rows


def _parse_two_column(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, raw in enumerate(source, start=1):
            fields = raw.strip().split()
            if len(fields) != 2 or fields[0] in rows:
                raise ValueError(f"Invalid two-column row at {path}:{line_number}: {raw!r}")
            rows[fields[0]] = fields[1]
    return rows


def _parse_segments(path: Path) -> list[tuple[str, str, str, str]]:
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as source:
        for line_number, raw in enumerate(source, start=1):
            fields = raw.strip().split()
            if len(fields) != 4 or fields[0] in seen:
                raise ValueError(f"Invalid segment row at {path}:{line_number}: {raw!r}")
            start, end = Decimal(fields[2]), Decimal(fields[3])
            if start < 0 or end <= start:
                raise ValueError(f"Invalid segment bounds at {path}:{line_number}: {start}, {end}")
            rows.append((fields[0], fields[1], fields[2], fields[3]))
            seen.add(fields[0])
    return rows


def _validate_metadata(source_root: Path, contract: LanguageContract):
    files_dir = source_root / contract.source_dir / "files"
    for name, expected_sha in contract.metadata_sha256.items():
        path = files_dir / name
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(f"MUCS {contract.source_dir}/{name} drift: {actual_sha} != {expected_sha}")

    references = _parse_kaldi_text(files_dir / "text")
    segments = _parse_segments(files_dir / "segments")
    wav_scp = _parse_two_column(files_dir / "wav.scp")
    utt2spk = _parse_two_column(files_dir / "utt2spk")
    if len(references) != contract.rows or len(segments) != contract.rows or len(utt2spk) != contract.rows:
        raise ValueError(
            f"MUCS {contract.source_dir} row contract changed: "
            f"text={len(references)}, segments={len(segments)}, utt2spk={len(utt2spk)}; expected={contract.rows}"
        )
    if len(wav_scp) != contract.recordings:
        raise ValueError(
            f"MUCS {contract.source_dir} recording contract changed: {len(wav_scp)} != {contract.recordings}"
        )
    ids = set(references)
    if {row[0] for row in segments} != ids or set(utt2spk) != ids:
        raise ValueError(f"MUCS {contract.source_dir} text/segments/utt2spk IDs do not match exactly")
    empty_refs = sum(not value.strip() for value in references.values())
    if empty_refs != contract.empty_references:
        raise ValueError(
            f"MUCS {contract.source_dir} empty-reference contract changed: {empty_refs} != {contract.empty_references}"
        )
    for recording_id, wav_name in wav_scp.items():
        if wav_name != f"{recording_id}.wav":
            raise ValueError(f"Unsupported wav.scp command/path for {recording_id}: {wav_name!r}")
    if any(row[1] not in wav_scp for row in segments):
        raise ValueError(f"MUCS {contract.source_dir} segments reference an unknown recording")
    return references, segments, wav_scp, utt2spk


def _expected_archive_names(parsed_metadata: dict[str, tuple]) -> set[str]:
    root = ARCHIVE_ROOT
    expected = {
        root,
        f"{root}/readme",
        f"{root}/inference",
        f"{root}/inference/compute_wer.py",
        f"{root}/inference/punct.mapping.txt",
        f"{root}/inference/hin2engwords.txt",
        f"{root}/inference/ben2engwords.txt",
    }
    for contract in LANGUAGE_CONTRACTS.values():
        expected.update({f"{root}/{contract.source_dir}", f"{root}/{contract.source_dir}/files"})
        expected.update(f"{root}/{contract.source_dir}/files/{name}" for name in contract.metadata_sha256)
        wav_scp = parsed_metadata[contract.leaf][2]
        expected.update(f"{root}/{contract.source_dir}/{name}" for name in wav_scp.values())
    return expected


def _extract_and_validate(archive_path: Path, destination: Path) -> tuple[Path, dict[str, tuple]]:
    source_root = destination / ARCHIVE_ROOT
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(members) != ARCHIVE_MEMBER_COUNT or len(set(names)) != ARCHIVE_MEMBER_COUNT:
            raise ValueError(
                f"MUCS archive inventory changed: members={len(members)}, unique={len(set(names))}; "
                f"expected={ARCHIVE_MEMBER_COUNT}"
            )
        by_name = {member.name: member for member in members}
        small_names = set(SMALL_SOURCE_CONTRACTS)
        for contract in LANGUAGE_CONTRACTS.values():
            small_names.update(f"{contract.source_dir}/files/{name}" for name in contract.metadata_sha256)
        for relative in sorted(small_names):
            archive_name = f"{ARCHIVE_ROOT}/{relative}"
            if archive_name not in by_name:
                raise ValueError(f"Missing MUCS archive member: {archive_name}")
            _extract_regular_member(archive, by_name[archive_name], destination)

        for relative, expected_sha in SMALL_SOURCE_CONTRACTS.items():
            actual_sha = sha256_file(source_root / relative)
            if actual_sha != expected_sha:
                raise ValueError(f"MUCS {relative} drift: {actual_sha} != {expected_sha}")
        for asset_name in ASSET_CONTRACTS:
            source_name = "compute_wer.py" if asset_name == "official_compute_wer.py" else asset_name
            if (source_root / "inference" / source_name).read_bytes() != (ASSETS_DIR / asset_name).read_bytes():
                raise ValueError(f"Vendored MUCS scorer asset differs from pinned archive: {asset_name}")

        parsed = {contract.leaf: _validate_metadata(source_root, contract) for contract in LANGUAGE_CONTRACTS.values()}
        expected_names = _expected_archive_names(parsed)
        if set(names) != expected_names:
            missing = sorted(expected_names - set(names))
            extra = sorted(set(names) - expected_names)
            raise ValueError(f"MUCS archive member contract changed; missing={missing}, extra={extra}")

        for contract in LANGUAGE_CONTRACTS.values():
            wav_scp = parsed[contract.leaf][2]
            for wav_name in wav_scp.values():
                archive_name = f"{ARCHIVE_ROOT}/{contract.source_dir}/{wav_name}"
                _extract_regular_member(archive, by_name[archive_name], destination)
    return source_root, parsed


def _frame_index(seconds: str, sampling_rate: int) -> int:
    exact = Decimal(seconds) * sampling_rate
    integral = exact.to_integral_value()
    if exact != integral:
        raise ValueError(f"Segment boundary {seconds}s is not sample-aligned at {sampling_rate} Hz")
    return int(integral)


def _slice_recording(source_path: Path, output_path: Path, start_seconds: str, end_seconds: str) -> float:
    with sf.SoundFile(source_path) as source:
        if source.samplerate != 16000 or source.channels != 1 or source.format != "WAV" or source.subtype != "PCM_16":
            raise ValueError(
                f"Unexpected MUCS source audio format for {source_path}: "
                f"rate={source.samplerate}, channels={source.channels}, format={source.format}, subtype={source.subtype}"
            )
        start_frame = _frame_index(start_seconds, source.samplerate)
        end_frame = _frame_index(end_seconds, source.samplerate)
        if end_frame > source.frames:
            raise ValueError(f"Segment exceeds source recording {source_path}: {end_frame} > {source.frames}")
        source.seek(start_frame)
        samples = source.read(end_frame - start_frame, dtype="int16", always_2d=False)
        if len(samples) != end_frame - start_frame:
            raise ValueError(f"Short read while slicing {source_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, samples, 16000, format="WAV", subtype="PCM_16")
    return (end_frame - start_frame) / 16000


def _container_audio_path(leaf: str, utterance_id: str) -> str:
    audio_root = os.getenv("NEMO_SKILLS_AUDIO_ROOT", "/data").rstrip("/")
    return f"{audio_root}/mucs2021/{leaf}/audio/{utterance_id}.wav"


def _prepare_language(
    data_dir: Path,
    source_root: Path,
    contract: LanguageContract,
    parsed: tuple,
    no_audio: bool,
) -> dict:
    references, segments, _, utt2spk = parsed
    leaf_dir = data_dir / contract.leaf
    audio_dir = leaf_dir / "audio"
    leaf_dir.mkdir(parents=True, exist_ok=True)
    if not no_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = leaf_dir / "test.jsonl"
    expected_audio_names = {f"{utterance_id}.wav" for utterance_id in references}
    total_duration = 0.0
    with manifest_path.open("w", encoding="utf-8") as output:
        for utterance_id, recording_id, start, end in segments:
            if not utterance_id.replace("_", "").isalnum():
                raise ValueError(f"Unsafe MUCS utterance ID: {utterance_id!r}")
            duration = float(Decimal(end) - Decimal(start))
            if not no_audio:
                duration = _slice_recording(
                    source_root / contract.source_dir / f"{recording_id}.wav",
                    audio_dir / f"{utterance_id}.wav",
                    start,
                    end,
                )
            total_duration += duration
            audio_path = _container_audio_path(contract.leaf, utterance_id)
            audio = {"path": audio_path, "duration": duration}
            record = {
                "id": utterance_id,
                "expected_answer": references[utterance_id],
                "audio_path": audio_path,
                "duration": duration,
                "messages": [{"role": "user", "content": PROMPT, "audio": audio}],
                "subset_for_metrics": contract.leaf,
                "task_type": "Code-Switched-ASR",
                "extra_fields": {
                    "mucs_language": contract.leaf,
                    "speaker_id": utt2spk[utterance_id],
                    "source_recording_id": recording_id,
                    "segment_start": start,
                    "segment_end": end,
                    "source_url": SOURCE_URL,
                    "source_sha256": SOURCE_SHA256,
                    "source_archive_member": f"{ARCHIVE_ROOT}/{contract.source_dir}/{recording_id}.wav",
                    "reference_source": f"{ARCHIVE_ROOT}/{contract.source_dir}/files/text",
                    "segments_source": f"{ARCHIVE_ROOT}/{contract.source_dir}/files/segments",
                    "official_word_mapping": contract.word_mapping_asset,
                },
            }
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    if not no_audio:
        actual_audio_names = {path.name for path in audio_dir.glob("*.wav")}
        if actual_audio_names != expected_audio_names:
            raise ValueError(
                f"MUCS {contract.leaf} prepared audio inventory mismatch: "
                f"missing={sorted(expected_audio_names - actual_audio_names)}, "
                f"extra={sorted(actual_audio_names - expected_audio_names)}"
            )
    if sum(1 for _ in manifest_path.open(encoding="utf-8")) != contract.rows:
        raise ValueError(f"MUCS {contract.leaf} manifest row contract was not preserved")
    return {
        "manifest": str(Path("mucs2021") / contract.leaf / "test.jsonl"),
        "manifest_sha256": sha256_file(manifest_path),
        "rows": contract.rows,
        "recordings": contract.recordings,
        "empty_references_skipped_by_official_scorer": contract.empty_references,
        "total_segment_duration_seconds": total_duration,
        "audio_materialized": not no_audio,
    }


def prepare_mucs2021(data_dir: Path, archive_path: Path | None = None, no_audio: bool = False) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_path or raw_dir / ARCHIVE_NAME
    if archive_path.exists():
        _validate_archive_file(archive_path)
    else:
        archive_path = _download_archive(archive_path)

    with tempfile.TemporaryDirectory(prefix=".mucs2021-extract-", dir=data_dir) as temporary:
        source_root, parsed = _extract_and_validate(archive_path, Path(temporary))
        leaves = {
            contract.leaf: _prepare_language(data_dir, source_root, contract, parsed[contract.leaf], no_audio)
            for contract in LANGUAGE_CONTRACTS.values()
        }

    provenance = {
        "benchmark": "MUCS 2021 Subtask 2 blind test",
        "source": {
            "url": SOURCE_URL,
            "sha256": SOURCE_SHA256,
            "bytes": SOURCE_BYTES,
            "archive_member_count": ARCHIVE_MEMBER_COUNT,
        },
        "protocol": {
            "official_script": "inference/compute_wer.py",
            "normal_wer": "lowercase whitespace tokens; punctuation map on hypothesis only",
            "transliterated_wer": "normal-WER hypothesis then language word map on hypothesis only",
            "empty_references": "skipped exactly as released scorer",
            "aggregate": "each language reported separately; no official cross-language overall",
        },
        "scorer_assets": {name: {"sha256": sha, "rows": rows} for name, (sha, rows) in ASSET_CONTRACTS.items()},
        "leaves": leaves,
    }
    provenance_path = data_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return provenance_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official MUCS 2021 Subtask 2 blind-test data")
    parser.add_argument("--data_dir", default=os.getenv("NEMO_SKILLS_DATA_DIR"))
    parser.add_argument("--archive", type=Path, help="Use an already downloaded pinned archive")
    parser.add_argument(
        "--no-audio", action="store_true", help="Write manifests but skip per-segment WAV materialization"
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir) / "mucs2021" if args.data_dir else Path(__file__).parent
    provenance_path = prepare_mucs2021(data_dir, archive_path=args.archive, no_audio=args.no_audio)
    print(f"MUCS 2021 prepared at {provenance_path}")


if __name__ == "__main__":
    main()
