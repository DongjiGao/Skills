# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the official public VoxLingua107 development set.

The paper's public evaluation protocol is the frozen 33-language development
archive, not the unlabeled 107-language training corpus. The archive checksum,
label space, and per-label cardinalities are pinned below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import soundfile as sf

BENCHMARK_NAME = "voxlingua107"
SOURCE_URL = "https://cs.taltech.ee/staff/tanel.alumae/data/voxlingua107/dev.zip"
SOURCE_SHA256 = "54c352d2ecbcbc03a4bb24151750a287573a5c2738636387a2240700bb89d2d2"
SOURCE_SPLIT = "dev"
EXPECTED_LABEL_COUNTS = {
    "ar": 100,
    "az": 34,
    "da": 99,
    "de": 82,
    "el": 5,
    "en": 80,
    "es": 55,
    "et": 85,
    "fa": 100,
    "fi": 93,
    "fr": 100,
    "hr": 4,
    "hu": 4,
    "hy": 100,
    "is": 26,
    "it": 49,
    "ja": 42,
    "lt": 13,
    "lv": 95,
    "mk": 4,
    "nl": 100,
    "nn": 3,
    "no": 24,
    "pl": 2,
    "pt": 1,
    "ru": 29,
    "sl": 9,
    "sr": 7,
    "sv": 100,
    "tr": 72,
    "uk": 17,
    "ur": 52,
    "zh": 23,
}
CANONICAL_LABELS = tuple(sorted(EXPECTED_LABEL_COUNTS))
EXPECTED_ROWS = sum(EXPECTED_LABEL_COUNTS.values())
PROMPT = (
    "Identify the language spoken in this audio. Respond with exactly one of "
    "the following language labels and nothing else: "
    + ", ".join(CANONICAL_LABELS)
    + "."
)


def _default_data_dir() -> Path:
    root = os.environ.get("NEMO_SKILLS_DATA_DIR")
    return Path(root) / BENCHMARK_NAME if root else Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_archive(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and _sha256(destination) == SOURCE_SHA256:
        return destination
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "nemo-skills/voxlingua107"})
    with urllib.request.urlopen(request, timeout=180) as source, partial.open("wb") as output:
        shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    actual = _sha256(partial)
    if actual != SOURCE_SHA256:
        partial.unlink(missing_ok=True)
        raise ValueError(f"VoxLingua107 archive checksum mismatch: {actual}")
    partial.replace(destination)
    return destination


def _member_identity(name: str) -> tuple[str, str]:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 2:
        raise ValueError(f"Unsafe or unexpected VoxLingua107 archive member: {name!r}")
    label, filename = path.parts
    if label not in EXPECTED_LABEL_COUNTS or not filename.lower().endswith(".wav"):
        raise ValueError(f"Unexpected VoxLingua107 archive member: {name!r}")
    return label, filename


def _inventory(archive: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, str, str]]:
    rows = []
    counts = {label: 0 for label in CANONICAL_LABELS}
    identities: set[tuple[str, str]] = set()
    for member in archive.infolist():
        if member.is_dir():
            continue
        label, filename = _member_identity(member.filename)
        identity = (label, filename)
        if identity in identities:
            raise ValueError(f"Duplicate VoxLingua107 archive member: {member.filename}")
        identities.add(identity)
        counts[label] += 1
        rows.append((member, label, filename))
    if counts != EXPECTED_LABEL_COUNTS:
        raise ValueError(
            "Pinned VoxLingua107 archive cardinalities changed: "
            f"expected={EXPECTED_LABEL_COUNTS}, actual={counts}"
        )
    return sorted(rows, key=lambda item: (item[1], item[2]))


def _extract_audio(
    archive: zipfile.ZipFile,
    members: list[tuple[zipfile.ZipInfo, str, str]],
    audio_dir: Path,
) -> None:
    for member, label, filename in members:
        destination = audio_dir / label / filename
        if destination.is_file() and destination.stat().st_size == member.file_size:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".partial")
        with archive.open(member) as source, partial.open("wb") as output:
            shutil.copyfileobj(source, output)
        partial.replace(destination)


def _record(label: str, filename: str, duration: float, audio_prefix: str) -> dict:
    audio_path = f"{audio_prefix.rstrip('/')}/{label}/{filename}"
    audio = {"path": audio_path, "duration": duration}
    return {
        "id": f"{label}-{Path(filename).stem}",
        "expected_answer": label,
        "messages": [{"role": "user", "content": PROMPT, "audio": audio}],
        "audio_path": audio_path,
        "duration": duration,
        "subset_for_metrics": label,
        "task_type": "VoxLingua107-LID",
        "extra_fields": {
            "language_label": label,
            "source_url": SOURCE_URL,
            "source_sha256": SOURCE_SHA256,
            "source_split": SOURCE_SPLIT,
        },
    }


def prepare_voxlingua107(
    data_dir: Path,
    *,
    archive_path: Path | None = None,
    audio_prefix: str = f"/data/{BENCHMARK_NAME}/audio",
) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_path or _download_archive(data_dir / "source" / "dev.zip")
    actual = _sha256(archive_path)
    if actual != SOURCE_SHA256:
        raise ValueError(
            f"VoxLingua107 archive checksum mismatch: expected={SOURCE_SHA256}, actual={actual}"
        )
    audio_dir = data_dir / "audio"
    output_path = data_dir / "test.jsonl"
    with zipfile.ZipFile(archive_path) as archive:
        members = _inventory(archive)
        _extract_audio(archive, members, audio_dir)

    with output_path.open("w", encoding="utf-8") as output:
        for _member, label, filename in members:
            local_audio = audio_dir / label / filename
            duration = float(sf.info(local_audio).duration)
            output.write(json.dumps(_record(label, filename, duration, audio_prefix), ensure_ascii=False) + "\n")
    if len(members) != EXPECTED_ROWS:
        raise AssertionError(f"Expected {EXPECTED_ROWS} VoxLingua107 rows, found {len(members)}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official VoxLingua107 development-set LID")
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--archive-path", type=Path, default=None)
    parser.add_argument("--audio-prefix", default=f"/data/{BENCHMARK_NAME}/audio")
    args = parser.parse_args()
    output = prepare_voxlingua107(
        args.data_dir or _default_data_dir(),
        archive_path=args.archive_path,
        audio_prefix=args.audio_prefix,
    )
    print(f"VoxLingua107: wrote {EXPECTED_ROWS} rows / {len(CANONICAL_LABELS)} labels to {output}")


if __name__ == "__main__":
    main()
