# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Prepare ContextASR-Bench dataset for NeMo Skills evaluation.

Downloads the ContextASR-Speech English subset from HuggingFace (JSONL + 8
audio tar files, ~22 GB total), extracts audio, and produces three JSONL
split files — one per evaluation mode (contextless, coarse, fine).

The data/output root must be supplied through ``--data_dir`` or
``NEMO_SKILLS_DATA_DIR``. If it already contains the dataset, the download
step is skipped and the existing data is used directly.

Dataset: https://huggingface.co/datasets/MrSupW/ContextASR-Bench

Usage:
    # Auto-download under an immutable benchmark-data root
    NEMO_SKILLS_DATA_DIR=/path/to/skills-data ns prepare_data contextasr-bench

    # Use pre-downloaded data
    ns prepare_data contextasr-bench --data_dir=/path/to/ContextASR-Bench

    # Skip audio download (JSONL only, not recommended)
    ns prepare_data contextasr-bench --no-audio
"""

import argparse
import hashlib
import json
import os
import tarfile
from pathlib import Path

HF_REPO_ID = "MrSupW/ContextASR-Bench"
BENCHMARK_NAME = "contextasr-bench"
HF_REVISION = "7906fd98d1d093bf244df80e726e07fc8afed6a6"
JSONL_FILENAME = "ContextASR-Speech_English.jsonl"
JSONL_SHA256 = "b6bcb3de39aebe36328ab03d7d4d865899712c2dc375b6cd2b9ce156b6a952e4"
EXPECTED_ROWS = 15326
AUDIO_TAR_PREFIX = "audio/ContextASR-Speech/English/ContextASR-Speech_English"
NUM_AUDIO_TARS = 8

PROMPT_CONTEXTLESS = "Transcribe the English audio into text, ensuring all punctuation marks are included."
PROMPT_COARSE = (
    "This audio belongs to the {domain_label} field. "
    "Transcribe the English audio into text, ensuring all punctuation marks are included."
)
PROMPT_FINE = (
    "This audio belongs to the {domain_label} field and may contain the following "
    "words or phrases: {entity_list}. "
    "Transcribe the English audio into text, ensuring all punctuation marks are included."
)


def _validate_metadata(jsonl_path: Path) -> list[dict]:
    payload = jsonl_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != JSONL_SHA256:
        raise ValueError(f"ContextASR metadata hash drift: expected {JSONL_SHA256}, found {digest}")

    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"ContextASR row-count drift: expected {EXPECTED_ROWS}, found {len(rows)}")
    if len({row["uniq_id"] for row in rows}) != len(rows):
        raise ValueError("ContextASR metadata contains duplicate uniq_id values")
    if len({row["audio"] for row in rows}) != len(rows):
        raise ValueError("ContextASR metadata contains duplicate audio paths")
    return rows


def _missing_audio(download_dir: Path, rows: list[dict]) -> list[str]:
    return [row["audio"] for row in rows if not (download_dir / row["audio"]).is_file()]


def _assert_audio_complete(download_dir: Path, rows: list[dict]) -> None:
    missing = _missing_audio(download_dir, rows)
    if missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(f"ContextASR download is incomplete: {len(missing)} files missing ({preview})")


def _default_data_dir() -> Path:
    """Resolve the CLI-managed immutable root and refuse staged-code output."""
    if data_root := os.getenv("NEMO_SKILLS_DATA_DIR"):
        return Path(data_root) / BENCHMARK_NAME
    raise SystemExit(
        f"Missing --data_dir and NEMO_SKILLS_DATA_DIR; refusing to write {BENCHMARK_NAME} into staged code."
    )


def _publish_records(out_path: Path, records: list[dict]) -> bool:
    """Atomically publish a deterministic manifest, or revalidate an existing one."""
    payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records).encode("utf-8")
    if out_path.exists():
        if not out_path.is_file() or out_path.read_bytes() != payload:
            raise ValueError(f"Existing ContextASR manifest fails deterministic closure: {out_path}")
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_suffix(out_path.suffix + ".partial")
    with partial.open("wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    partial.replace(out_path)
    if out_path.read_bytes() != payload:
        raise ValueError(f"Published ContextASR manifest failed byte revalidation: {out_path}")
    return True


def download_dataset(download_dir):
    """Download ContextASR-Speech English data from HuggingFace.

    Downloads the JSONL metadata and 8 audio tar files (~22 GB total),
    then extracts the tars. This can take 30-60 minutes depending on
    network speed.
    """
    from huggingface_hub import hf_hub_download

    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = download_dir / JSONL_FILENAME
    audio_dir = download_dir / "audio" / "ContextASR-Speech" / "English"

    # Check if data already exists
    if jsonl_path.exists() and audio_dir.exists():
        rows = _validate_metadata(jsonl_path)
        missing = _missing_audio(download_dir, rows)
        if not missing:
            print(f"Dataset already exists at {download_dir} ({len(rows)} pinned audio files). Skipping download.")
            return download_dir
        print(f"Dataset is incomplete at {download_dir}: {len(missing)} referenced audio files are missing.")

    print("=" * 70)
    print("DOWNLOADING ContextASR-Bench English Speech data from HuggingFace")
    print(f"Repository: {HF_REPO_ID}")
    print(f"Destination: {download_dir}")
    print("Total download size: ~22 GB (JSONL + 8 audio tar files)")
    print()
    print("WARNING: This download may take 30-60 minutes depending on your")
    print("network speed. You can skip this by pre-downloading the data and")
    print("passing --data_dir=/path/to/ContextASR-Bench.")
    print("=" * 70)

    # Download JSONL
    print(f"\n[1/{NUM_AUDIO_TARS + 1}] Downloading {JSONL_FILENAME}...")
    hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=JSONL_FILENAME,
        repo_type="dataset",
        local_dir=str(download_dir),
        revision=HF_REVISION,
    )
    print(f"  Saved to {jsonl_path}")
    rows = _validate_metadata(jsonl_path)

    # Download and extract audio tars
    audio_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, NUM_AUDIO_TARS + 1):
        tar_filename = f"{AUDIO_TAR_PREFIX}_{i}.tar"
        print(f"\n[{i + 1}/{NUM_AUDIO_TARS + 1}] Downloading {tar_filename}...")

        local_tar = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=tar_filename,
                repo_type="dataset",
                local_dir=str(download_dir),
                revision=HF_REVISION,
            )
        )

        print(f"  Extracting {local_tar.name} to {audio_dir}...")
        with tarfile.open(local_tar) as tf:
            tf.extractall(path=audio_dir, filter="data")

        print("  Extracted. Removing tar file to save space...")
        local_tar.unlink()

    _assert_audio_complete(download_dir, rows)
    print(f"\nDownload complete! {len(rows)} pinned audio files verified under {audio_dir}")

    return download_dir


def build_messages(prompt_text, audio_path, duration):
    """Build OpenAI-format messages with audio metadata."""
    return [
        {
            "role": "user",
            "content": prompt_text,
            "audio": {
                "path": audio_path,
                "duration": float(duration),
            },
        }
    ]


def format_entry(sample, mode, audio_prefix):
    """Format a single dataset sample into a JSONL record for a given mode."""
    audio_path = f"{audio_prefix}/{sample['audio']}"
    entity_list = sample["entity_list"]
    domain_label = sample["domain_label"]

    if mode == "contextless":
        prompt = PROMPT_CONTEXTLESS
    elif mode == "coarse":
        prompt = PROMPT_COARSE.format(domain_label=domain_label)
    elif mode == "fine":
        entity_str = ", ".join(entity_list)
        prompt = PROMPT_FINE.format(domain_label=domain_label, entity_list=entity_str)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return {
        "messages": build_messages(prompt, audio_path, sample["duration"]),
        "expected_answer": sample["text"],
        "entity_list": entity_list,
        "domain_label": domain_label,
        "subset_for_metrics": domain_label,
        "uniq_id": sample["uniq_id"],
        "duration": float(sample["duration"]),
        "audio_filepath": audio_path,
        "source_provenance": {
            "dataset": HF_REPO_ID,
            "dataset_revision": HF_REVISION,
            "metadata_sha256": JSONL_SHA256,
            "expected_rows": EXPECTED_ROWS,
        },
    }


def main():
    """Parse arguments, download data if needed, and write per-mode JSONL splits."""
    parser = argparse.ArgumentParser(description="Prepare ContextASR-Bench for NeMo Skills")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help=(
            "Path to ContextASR-Bench dataset root. If the directory already contains "
            "ContextASR-Speech_English.jsonl, that data is used directly. If the "
            "directory is empty or the file is missing, data will be downloaded there "
            "automatically from HuggingFace. "
            f"If not provided, requires $NEMO_SKILLS_DATA_DIR and uses its {BENCHMARK_NAME} subdirectory."
        ),
    )
    parser.add_argument(
        "--audio-prefix",
        type=str,
        default=None,
        help=(
            "Override audio path prefix written into JSONL files. "
            "Defaults to the data_dir value. Useful for container mount points "
            "(e.g., --audio-prefix /data/contextasr-bench)."
        ),
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip downloading/verifying audio files (not recommended for actual evaluation)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else _default_data_dir()
    jsonl_path = data_dir / JSONL_FILENAME

    if jsonl_path.exists():
        print(f"Using pre-downloaded data from {data_dir}")
        _validate_metadata(jsonl_path)
        if not args.no_audio:
            download_dataset(data_dir)
    elif args.no_audio:
        raise FileNotFoundError(
            f"Dataset file not found: {jsonl_path}\n"
            f"Cannot use --no-audio when data has not been downloaded yet. "
            f"Either run without --no-audio first to download, or point --data_dir "
            f"to a directory that already contains {JSONL_FILENAME}."
        )
    else:
        print(f"Data not found at {data_dir}. Downloading there...")
        download_dataset(data_dir)

    audio_prefix = args.audio_prefix if args.audio_prefix else str(data_dir)
    audio_prefix = audio_prefix.rstrip("/")

    jsonl_path = data_dir / JSONL_FILENAME

    print(f"\nReading dataset from {jsonl_path}")
    samples = _validate_metadata(jsonl_path)

    print(f"Loaded {len(samples)} samples")

    if not args.no_audio:
        _assert_audio_complete(data_dir, samples)
        print(f"All {len(samples)} referenced audio files verified under {data_dir}")

    modes = {
        "contextless": data_dir / "contextless" / "test.jsonl",
        "coarse": data_dir / "coarse" / "test.jsonl",
        "fine": data_dir / "fine" / "test.jsonl",
    }

    print("\nWriting JSONL splits...")
    for mode_name, output_path in modes.items():
        entries = [format_entry(sample, mode_name, audio_prefix) for sample in samples]
        published = _publish_records(output_path, entries)
        action = "wrote" if published else "revalidated existing"
        print(f"  {mode_name}: {action} {len(entries)} samples at {output_path}")

    print(f"\nDone. Total: {len(samples)} samples x 3 modes = {len(samples) * 3} records")


if __name__ == "__main__":
    main()
