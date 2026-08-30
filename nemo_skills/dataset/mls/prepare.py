# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare the official Multilingual LibriSpeech test benchmark.

MLS itself and the paper's monolingual baseline table contain eight languages,
including English.  Meta's streamable Hugging Face release contains the seven
non-English languages; the public ``parler-tts/mls_eng`` mirror provides the
same OpenSLR English release without downloading the 651-GiB full archive.
Only the eight pinned test Parquets are downloaded.  Their SHA-256 digests,
sizes, and row cardinalities are verified before ``test.jsonl`` is published.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

BENCHMARK_NAME = "mls"
OFFICIAL_SPLIT = "test"
OFFICIAL_RELEASE = "OpenSLR SLR94"
OFFICIAL_RELEASE_URL = "https://www.openslr.org/94/"
PAPER_REVISION = "arxiv:2012.03411v2"
LICENSE = "CC-BY-4.0"

META_REPO_ID = "facebook/multilingual_librispeech"
META_REVISION = "2e83e61823b4c47dcbcb1980bb88601274127609"
ENGLISH_REPO_ID = "parler-tts/mls_eng"
ENGLISH_REVISION = "faf6604dcf0bb9ec0d9c280a140ad74da9c931b0"

EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHANNELS = 1
SYSTEM_MESSAGE = "You are a helpful assistant. /no_think"
USER_MESSAGE = "Transcribe the following audio."
PROVENANCE_FILENAME = "provenance.json"


@dataclass(frozen=True)
class LanguageSpec:
    code: str
    name: str
    repo_id: str
    revision: str
    parquet_path: str
    parquet_size: int
    parquet_sha256: str
    expected_rows: int


# Order matches the primary paper's Table 6.  SHA-256 values are the LFS object
# OIDs at the pinned revisions; each test split is exactly one Parquet shard.
LANGUAGE_SPECS = (
    LanguageSpec(
        "en",
        "English",
        ENGLISH_REPO_ID,
        ENGLISH_REVISION,
        "data/test-00000-of-00001.parquet",
        244_915_636,
        "e22d76c4fdfca21e5126eb07452036d09f1deb00cd0a03b985f39949dc19617b",
        3_769,
    ),
    LanguageSpec(
        "de",
        "German",
        META_REPO_ID,
        META_REVISION,
        "german/test-00000-of-00001.parquet",
        224_781_063,
        "2453b62d4a506015678f7f6252680621e880aac0d9038ae7bd08e4df26e7efe5",
        3_394,
    ),
    LanguageSpec(
        "nl",
        "Dutch",
        META_REPO_ID,
        META_REVISION,
        "dutch/test-00000-of-00001.parquet",
        198_390_898,
        "feb88a282be746e7c14e1f09c017b4f69b07df9c25c94a3943eaac5ca7794ad5",
        3_075,
    ),
    LanguageSpec(
        "fr",
        "French",
        META_REPO_ID,
        META_REVISION,
        "french/test-00000-of-00001.parquet",
        157_633_503,
        "118dad8c119d13564464cc2c5d2d4ffaea4f5752f4cb50a589f502e905ede736",
        2_426,
    ),
    LanguageSpec(
        "es",
        "Spanish",
        META_REPO_ID,
        META_REVISION,
        "spanish/test-00000-of-00001.parquet",
        157_823_544,
        "ffc32cb4a2d015c9b812a3c1dab3b60d4ea3fd01e3c3f8c9631fd51b730c99f3",
        2_385,
    ),
    LanguageSpec(
        "it",
        "Italian",
        META_REPO_ID,
        META_REVISION,
        "italian/test-00000-of-00001.parquet",
        82_876_885,
        "8d77c34025a4735fe18330495cf8d36ac6bc68aa66789630ee3f532e61d771dd",
        1_262,
    ),
    LanguageSpec(
        "pt",
        "Portuguese",
        META_REPO_ID,
        META_REVISION,
        "portuguese/test-00000-of-00001.parquet",
        58_871_912,
        "73317666498b6889851a72bea52b116c4744691dda4c8e66b8bd98af0496ba90",
        871,
    ),
    LanguageSpec(
        "pl",
        "Polish",
        META_REPO_ID,
        META_REVISION,
        "polish/test-00000-of-00001.parquet",
        33_587_477,
        "43656b9106d1e8d8c5cd0a3e2a50ef5a29bc659a3cbccf3541f0c1359255ac3b",
        520,
    ),
)
LANGUAGE_CODES = tuple(spec.code for spec in LANGUAGE_SPECS)
EXPECTED_LANGUAGE_ROWS = {spec.code: spec.expected_rows for spec in LANGUAGE_SPECS}
EXPECTED_TOTAL_ROWS = sum(EXPECTED_LANGUAGE_ROWS.values())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _download_and_verify_source(spec: LanguageSpec, cache_dir: Path) -> Path:
    path = Path(
        hf_hub_download(
            repo_id=spec.repo_id,
            filename=spec.parquet_path,
            repo_type="dataset",
            revision=spec.revision,
            cache_dir=cache_dir,
        )
    )
    actual_size = path.stat().st_size
    if actual_size != spec.parquet_size:
        raise ValueError(
            f"MLS {spec.code} source size mismatch: expected {spec.parquet_size}, observed {actual_size} at {path}"
        )
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != spec.parquet_sha256:
        raise ValueError(
            f"MLS {spec.code} source SHA-256 mismatch: expected {spec.parquet_sha256}, observed {actual_sha256}"
        )
    return path


def _load_source_rows(spec: LanguageSpec, cache_dir: Path):
    source_path = _download_and_verify_source(spec, cache_dir)
    dataset = load_dataset("parquet", data_files={OFFICIAL_SPLIT: str(source_path)}, split=OFFICIAL_SPLIT)
    if len(dataset) != spec.expected_rows:
        raise ValueError(
            f"MLS {spec.code} test cardinality mismatch: expected {spec.expected_rows}, observed {len(dataset)}"
        )
    required_columns = {"audio", "transcript", "original_path"}
    missing_columns = required_columns - set(dataset.column_names)
    if missing_columns:
        raise ValueError(f"MLS {spec.code} source is missing required columns: {sorted(missing_columns)}")
    return dataset.cast_column("audio", Audio(decode=False))


def _sample_id(row: dict[str, Any], spec: LanguageSpec) -> str:
    value = row.get("id")
    if value is None or not str(value).strip():
        original_path = str(row.get("original_path") or "").strip()
        value = PurePosixPath(original_path).stem
    sample_id = str(value).strip()
    if not sample_id or re.fullmatch(r"[A-Za-z0-9_.-]+", sample_id) is None:
        raise ValueError(f"MLS {spec.code} has an unsafe or empty sample id: {sample_id!r}")
    return sample_id


def _decode_audio(row: dict[str, Any], spec: LanguageSpec, sample_id: str) -> tuple[np.ndarray, int]:
    audio = row.get("audio")
    if not isinstance(audio, dict) or not isinstance(audio.get("bytes"), bytes):
        raise TypeError(f"MLS {spec.code}/{sample_id} is missing embedded source audio bytes")
    try:
        samples, sample_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
    except Exception as error:
        raise ValueError(f"MLS {spec.code}/{sample_id} has unreadable source audio") from error
    samples = np.asarray(samples)
    channels = 1 if samples.ndim == 1 else samples.shape[1]
    if sample_rate != EXPECTED_SAMPLE_RATE or channels != EXPECTED_CHANNELS or samples.size == 0:
        raise ValueError(
            f"MLS {spec.code}/{sample_id} must be nonempty {EXPECTED_SAMPLE_RATE}-Hz mono audio; "
            f"observed rate={sample_rate}, channels={channels}, samples={samples.size}"
        )
    return samples, int(sample_rate)


def format_entry(
    row: dict[str, Any],
    spec: LanguageSpec,
    *,
    sample_id: str,
    audio_path: str,
    duration: float,
) -> dict[str, Any]:
    transcript = str(row.get("transcript") or "").strip()
    if not transcript:
        raise ValueError(f"MLS {spec.code}/{sample_id} has an empty transcript")
    return {
        "task_type": "Multilingual-ASR",
        "expected_answer": transcript,
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": USER_MESSAGE, "audio": {"path": audio_path, "duration": duration}},
        ],
        "subset_for_metrics": spec.code,
        "id": f"{spec.code}-{sample_id}",
        "audio_filepath": audio_path,
        "duration": duration,
        "extra_fields": {
            "src_lang": spec.code,
            "remove_diacritics": False,
            "normalize_compound": False,
            "merge_compounds": False,
            "use_cer": False,
        },
        "source_provenance": {
            "release": OFFICIAL_RELEASE,
            "release_url": OFFICIAL_RELEASE_URL,
            "paper_revision": PAPER_REVISION,
            "dataset": spec.repo_id,
            "dataset_revision": spec.revision,
            "source_file": spec.parquet_path,
            "source_file_sha256": spec.parquet_sha256,
            "split": OFFICIAL_SPLIT,
            "expected_language_rows": spec.expected_rows,
            "normalization": "language-aware multilingual; diacritics preserved; no compound-boundary forgiveness",
        },
    }


def _prepare_language(
    spec: LanguageSpec,
    *,
    data_dir: Path,
    cache_dir: Path,
    audio_prefix: str,
    no_audio: bool,
) -> list[dict[str, Any]]:
    dataset = _load_source_rows(spec, cache_dir)
    audio_dir = data_dir / "audio" / spec.code
    if not no_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in tqdm(dataset, desc=f"MLS {spec.code}"):
        sample_id = _sample_id(row, spec)
        if sample_id in seen_ids:
            raise ValueError(f"MLS {spec.code} contains duplicate sample id {sample_id!r}")
        seen_ids.add(sample_id)

        duration_value = row.get("audio_duration")
        samples = None
        sample_rate = None
        if not no_audio:
            samples, sample_rate = _decode_audio(row, spec, sample_id)
            duration = len(samples) / sample_rate
            sf.write(audio_dir / f"{sample_id}.flac", samples, sample_rate)
        elif duration_value is not None and float(duration_value) > 0:
            duration = float(duration_value)
        else:
            begin_time = float(row.get("begin_time") or 0.0)
            end_time = float(row.get("end_time") or 0.0)
            duration = end_time - begin_time
            if duration <= 0:
                raise ValueError(f"MLS {spec.code}/{sample_id} has no valid duration metadata")

        container_audio_path = f"{audio_prefix.rstrip('/')}/{spec.code}/{sample_id}.flac"
        rows.append(
            format_entry(
                row,
                spec,
                sample_id=sample_id,
                audio_path=container_audio_path,
                duration=duration,
            )
        )

    if len(rows) != spec.expected_rows:
        raise ValueError(
            f"MLS {spec.code} formatted cardinality mismatch: expected {spec.expected_rows}, observed {len(rows)}"
        )
    return rows


def prepare_mls(
    data_dir: Path,
    *,
    audio_prefix: str = "/data/mls/audio",
    no_audio: bool = False,
) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = data_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for spec in LANGUAGE_SPECS:
        all_rows.extend(
            _prepare_language(
                spec,
                data_dir=data_dir,
                cache_dir=cache_dir,
                audio_prefix=audio_prefix,
                no_audio=no_audio,
            )
        )

    language_counts = Counter(row["subset_for_metrics"] for row in all_rows)
    if dict(language_counts) != EXPECTED_LANGUAGE_ROWS or len(all_rows) != EXPECTED_TOTAL_ROWS:
        raise ValueError(
            "MLS combined test inventory mismatch: "
            f"expected={EXPECTED_LANGUAGE_ROWS}/{EXPECTED_TOTAL_ROWS}, "
            f"observed={dict(language_counts)}/{len(all_rows)}"
        )

    output_path = data_dir / "test.jsonl"
    temporary_output = output_path.with_name(f".{output_path.name}.tmp")
    provenance_path = data_dir / PROVENANCE_FILENAME
    try:
        with temporary_output.open("w", encoding="utf-8") as output:
            for row in all_rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        receipt = {
            "schema_version": 1,
            "benchmark": BENCHMARK_NAME,
            "license": LICENSE,
            "official_release": OFFICIAL_RELEASE,
            "official_release_url": OFFICIAL_RELEASE_URL,
            "paper_revision": PAPER_REVISION,
            "official_split": OFFICIAL_SPLIT,
            "languages": list(LANGUAGE_CODES),
            "language_rows": EXPECTED_LANGUAGE_ROWS,
            "total_rows": EXPECTED_TOTAL_ROWS,
            "sources": [asdict(spec) for spec in LANGUAGE_SPECS],
            "manifest": output_path.name,
            "manifest_sha256": _sha256_file(temporary_output),
            "audio_mode": "metadata-only" if no_audio else "materialized-and-validated",
            "scoring": {
                "per_language": "corpus WER",
                "headline": "unweighted mean of the eight per-language corpus WERs",
                "normalization": (
                    "language-aware multilingual normalization with diacritics preserved; "
                    "no compound-boundary normalization or merge forgiveness"
                ),
            },
        }
        os.replace(temporary_output, output_path)
        _write_json_atomic(provenance_path, receipt)
    finally:
        temporary_output.unlink(missing_ok=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the pinned official eight-language MLS ASR test set")
    parser.add_argument("--data_dir", default=os.getenv("NEMO_SKILLS_DATA_DIR"))
    parser.add_argument("--audio-prefix", default="/data/mls/audio")
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Publish metadata-only paths after verifying source Parquets (not usable for evaluation)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir) / BENCHMARK_NAME if args.data_dir else Path(__file__).parent
    output_path = prepare_mls(data_dir, audio_prefix=args.audio_prefix, no_audio=args.no_audio)
    print(f"Prepared {EXPECTED_TOTAL_ROWS} MLS test utterances at {output_path}")


if __name__ == "__main__":
    main()
