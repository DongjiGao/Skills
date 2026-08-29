# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare AliMeeting's official public far-field test split.

Corpus: OpenSLR-119 (CC BY-SA 4.0). The official archive contains 20 test
meetings with near- and far-field audio; only ``Test_Ali_far`` is extracted.
Channel zero is used to match the released SDM/single-channel recipe.
"""

from __future__ import annotations

import argparse
import os
import shutil
import urllib.request
from pathlib import Path

from nemo_skills.dataset.mandarin_meeting_common import (
    prepare_meeting_tree,
    safe_stream_extract_tar_gz,
    sha256_file,
)

BENCHMARK_NAME = "alimeeting-sot"
SOURCE_URL = "https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Test_Ali.tar.gz"
SOURCE_SPLIT = "Test_Ali_far"
EXPECTED_SESSIONS = 20
OFFICIAL_REPOSITORY_REVISION = "692a034fd510f1720f547c2c91e0fbadc9c24bf2"


def _default_data_dir() -> Path:
    root = os.getenv("NEMO_SKILLS_DATA_DIR")
    return Path(root) / BENCHMARK_NAME if root else Path(__file__).resolve().parent


def _download(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "nemo-skills/alimeeting-sot"})
    with urllib.request.urlopen(request, timeout=300) as source, partial.open("wb") as output:
        shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    partial.replace(destination)
    return destination


def _locate_far_tree(root: Path) -> tuple[Path, Path]:
    candidates = [path for path in root.rglob(SOURCE_SPLIT) if path.is_dir()]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one {SOURCE_SPLIT} directory under {root}, found {candidates}")
    far = candidates[0]
    audio = far / "audio_dir"
    textgrid = far / "textgrid_dir"
    if not audio.is_dir() or not textgrid.is_dir():
        raise FileNotFoundError(f"Incomplete AliMeeting far-field tree: {far}")
    return audio, textgrid


def prepare_alimeeting(
    output_dir: Path,
    *,
    archive_path: Path | None = None,
    corpus_dir: Path | None = None,
    audio_prefix: str = f"/data/{BENCHMARK_NAME}/audio",
    no_audio: bool = False,
    max_samples: int | None = None,
) -> Path:
    source_root = output_dir / "source"
    archive = None
    if corpus_dir is None:
        archive = archive_path or _download(source_root / "Test_Ali.tar.gz")
        extracted = source_root / "extracted"
        if not any(extracted.rglob(SOURCE_SPLIT)):
            safe_stream_extract_tar_gz(archive, extracted, SOURCE_SPLIT)
        corpus_dir = extracted
    audio_dir, textgrid_dir = _locate_far_tree(corpus_dir)
    metadata = {
        "source_url": SOURCE_URL,
        "source_split": SOURCE_SPLIT,
        "source_archive_sha256": sha256_file(archive) if archive is not None else None,
        "official_repository_revision": OFFICIAL_REPOSITORY_REVISION,
        "license": "CC-BY-SA-4.0",
    }
    manifest = prepare_meeting_tree(
        benchmark_name=BENCHMARK_NAME,
        source_audio_dir=audio_dir,
        source_textgrid_dir=textgrid_dir,
        output_dir=output_dir,
        audio_prefix=audio_prefix,
        source_metadata=metadata,
        no_audio=no_audio,
        max_samples=max_samples,
    )
    if max_samples is None:
        rows = sum(1 for line in manifest.open(encoding="utf-8") if line.strip())
        if rows != EXPECTED_SESSIONS:
            raise ValueError(f"Expected {EXPECTED_SESSIONS} AliMeeting test sessions, found {rows}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--archive-path", type=Path, default=None)
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--audio-prefix", default=f"/data/{BENCHMARK_NAME}/audio")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--max-samples", type=int, default=-1)
    args = parser.parse_args()
    output = prepare_alimeeting(
        args.data_dir / BENCHMARK_NAME if args.data_dir else _default_data_dir(),
        archive_path=args.archive_path,
        corpus_dir=args.corpus_dir,
        audio_prefix=args.audio_prefix,
        no_audio=args.no_audio,
        max_samples=args.max_samples if args.max_samples > 0 else None,
    )
    print(f"AliMeeting SOT prepared at {output}")


if __name__ == "__main__":
    main()
