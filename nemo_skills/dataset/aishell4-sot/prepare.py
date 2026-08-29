# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare AISHELL-4's official public test split from a pinned HF mirror.

The mirror preserves the 20 OpenSLR-111 test meetings and annotations. Only
``test/**`` is fetched at immutable revision ``aada727...``; channel zero is
materialized to match the official single-channel recipe.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from nemo_skills.dataset.mandarin_meeting_common import prepare_meeting_tree

BENCHMARK_NAME = "aishell4-sot"
HF_REPO_ID = "AISHELL/AISHELL-4"
HF_REVISION = "aada72727856313b19d4a030383c426364931dbf"
SOURCE_SPLIT = "test"
EXPECTED_SESSIONS = 20
OFFICIAL_REPOSITORY_REVISION = "bad82b77c3753df1b232c5c6491cd3e2f2e32d24"


def _default_data_dir() -> Path:
    root = os.getenv("NEMO_SKILLS_DATA_DIR")
    return Path(root) / BENCHMARK_NAME if root else Path(__file__).resolve().parent


def _snapshot(destination: Path) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        revision=HF_REVISION,
        allow_patterns=["test/**"],
        local_dir=destination,
    )
    return destination


def prepare_aishell4(
    output_dir: Path,
    *,
    corpus_dir: Path | None = None,
    audio_prefix: str = f"/data/{BENCHMARK_NAME}/audio",
    no_audio: bool = False,
    max_samples: int | None = None,
) -> Path:
    corpus = corpus_dir or _snapshot(output_dir / "source" / "snapshot")
    test_dir = corpus / SOURCE_SPLIT
    if not test_dir.is_dir() and corpus.name == SOURCE_SPLIT:
        test_dir = corpus
    metadata = {
        "source_repo": HF_REPO_ID,
        "source_revision": HF_REVISION,
        "source_split": SOURCE_SPLIT,
        "official_repository_revision": OFFICIAL_REPOSITORY_REVISION,
        "license": "CC-BY-SA-4.0",
    }
    manifest = prepare_meeting_tree(
        benchmark_name=BENCHMARK_NAME,
        source_audio_dir=test_dir / "wav",
        source_textgrid_dir=test_dir / "TextGrid",
        output_dir=output_dir,
        audio_prefix=audio_prefix,
        source_metadata=metadata,
        no_audio=no_audio,
        max_samples=max_samples,
    )
    if max_samples is None:
        rows = sum(1 for line in manifest.open(encoding="utf-8") if line.strip())
        if rows != EXPECTED_SESSIONS:
            raise ValueError(f"Expected {EXPECTED_SESSIONS} AISHELL-4 test sessions, found {rows}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--audio-prefix", default=f"/data/{BENCHMARK_NAME}/audio")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--max-samples", type=int, default=-1)
    args = parser.parse_args()
    output = prepare_aishell4(
        args.data_dir / BENCHMARK_NAME if args.data_dir else _default_data_dir(),
        corpus_dir=args.corpus_dir,
        audio_prefix=args.audio_prefix,
        no_audio=args.no_audio,
        max_samples=args.max_samples if args.max_samples > 0 else None,
    )
    print(f"AISHELL-4 SOT prepared at {output}")


if __name__ == "__main__":
    main()
