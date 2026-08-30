# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare AliMeeting's official public far-field test split.

Corpus: OpenSLR-119 (CC BY-SA 4.0). The official archive contains 20 test
meetings with near- and far-field audio; only ``Test_Ali_far`` is extracted.
Channel zero deliberately reproduces the released SDM/single-channel condition.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from nemo_skills.dataset.mandarin_meeting_common import (
    prepare_meeting_tree,
    resumable_http_download,
    safe_stream_extract_tar_gz,
    sha256_file,
)

BENCHMARK_NAME = "alimeeting-sot"
SOURCE_URL = "https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Test_Ali.tar.gz"
SOURCE_SPLIT = "Test_Ali_far"
AUDIO_STEM_BY_SESSION = {
    "R8002_M8002": "R8002_M8002_MS802",
    "R8002_M8003": "R8002_M8003_MS803",
    "R8004_M8005": "R8004_M8005_MS803",
    "R8004_M8006": "R8004_M8006_MS805",
    "R8005_M8007": "R8005_M8007_MS806",
    "R8005_M8008": "R8005_M8008_MS806",
    "R8005_M8009": "R8005_M8009_MS802",
    "R8006_M8012": "R8006_M8012_MS803",
    "R8008_M8014": "R8008_M8014_MS807",
    "R8008_M8015": "R8008_M8015_MS808",
    "R8008_M8016": "R8008_M8016_MS808",
    "R8008_M8017": "R8008_M8017_MS808",
    "R8009_M8021": "R8009_M8021_MS810",
    "R8009_M8022": "R8009_M8022_MS810",
    "R8009_M8023": "R8009_M8023_MS811",
    "R8009_M8024": "R8009_M8024_MS811",
    "R8009_M8025": "R8009_M8025_MS811",
    "R8009_M8026": "R8009_M8026_MS812",
    "R8009_M8027": "R8009_M8027_MS812",
    "R8009_M8028": "R8009_M8028_MS812",
}
EXPECTED_SESSION_IDS = tuple(AUDIO_STEM_BY_SESSION)
EXPECTED_SESSIONS = len(EXPECTED_SESSION_IDS)
OFFICIAL_REPOSITORY_REVISION = "692a034fd510f1720f547c2c91e0fbadc9c24bf2"
SOURCE_ARCHIVE_BYTES = 9_555_497_666


def _default_data_dir() -> Path:
    root = os.getenv("NEMO_SKILLS_DATA_DIR")
    return Path(root) / BENCHMARK_NAME if root else Path(__file__).resolve().parent


def _download(destination: Path) -> Path:
    return resumable_http_download(
        SOURCE_URL,
        destination,
        expected_size=SOURCE_ARCHIVE_BYTES,
        user_agent="nemo-skills/alimeeting-sot",
    )


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
        expected_session_ids=EXPECTED_SESSION_IDS,
        audio_stem_by_session=AUDIO_STEM_BY_SESSION,
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
