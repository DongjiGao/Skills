# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Prepare ASR Leaderboard datasets for evaluation.

Most datasets come from the official HF Open ASR Leaderboard ESB test-only
sorted dataset (hf-audio/esb-datasets-test-only-sorted), the same source used
by the official leaderboard. VoxPopuli is additionally available in the
Artificial Analysis cleaned version (ArtificialAnalysis/VoxPopuli-Cleaned-AA),
which ships audio as separate wav files rather than a decoded audio column.

Leaderboard protocol (the default set prepared with no --datasets):
    librispeech_clean, librispeech_other, voxpopuli_aa, tedlium,
    gigaspeech (samples < 1s dropped), spgispeech, earnings22

AMI and the original (uncleaned) VoxPopuli are intentionally excluded from the
default leaderboard set. They remain available on demand via --datasets for
other consumers (e.g. the ntt-smoke composite eval, which mixes AMI/VoxPopuli
audio with noise).

Audio paths in JSONL: /dataset/asr-leaderboard/data/{dataset}/{sample_id}.flac

Usage:
    ns prepare_data asr-leaderboard                          # leaderboard default set
    ns prepare_data asr-leaderboard --datasets all           # every dataset incl. ami + voxpopuli
    ns prepare_data asr-leaderboard --datasets librispeech_clean ami
    ns prepare_data asr-leaderboard --no-audio
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm

ESB_REPO = "hf-audio/esb-datasets-test-only-sorted"
AA_VOXPOPULI_REPO = "ArtificialAnalysis/VoxPopuli-Cleaned-AA"
SYSTEM_MESSAGE = "You are a helpful assistant. /no_think"
AUDIO_SAMPLE_RATE = 16000


@dataclass
class DatasetSpec:
    """How to load and format one ASR dataset.

    loader:
        "esb"    -> HF config inside ``repo`` with a decoded ``audio`` column.
        "aa_url" -> Artificial Analysis layout: metadata rows carry a ``url``
                    pointing to a separate wav file inside the dataset repo.
    min_duration:
        If set, drop samples shorter than this many seconds.
    """

    repo: str
    config: Optional[str]
    split: str
    text_field: str
    id_field: str
    loader: str = "esb"
    min_duration: Optional[float] = None


DATASET_CONFIGS = {
    "librispeech_clean": DatasetSpec(ESB_REPO, "librispeech", "test.clean", "text", "id"),
    "librispeech_other": DatasetSpec(ESB_REPO, "librispeech", "test.other", "text", "id"),
    "voxpopuli": DatasetSpec(ESB_REPO, "voxpopuli", "test", "text", "id"),
    "voxpopuli_aa": DatasetSpec(AA_VOXPOPULI_REPO, None, "test", "transcript", "id", loader="aa_url"),
    "tedlium": DatasetSpec(ESB_REPO, "tedlium", "test", "text", "id"),
    "gigaspeech": DatasetSpec(ESB_REPO, "gigaspeech", "test", "text", "id", min_duration=1.0),
    "spgispeech": DatasetSpec(ESB_REPO, "spgispeech", "test", "text", "id"),
    "earnings22": DatasetSpec(ESB_REPO, "earnings22", "test", "text", "id"),
    "ami": DatasetSpec(ESB_REPO, "ami", "test", "text", "id"),
}

# Datasets that make up the Open ASR Leaderboard average by default.
# Per the 2026 protocol update: AMI is dropped, GigaSpeech drops sub-1s clips,
# and VoxPopuli is replaced by the Artificial Analysis cleaned set.
DEFAULT_DATASETS = [
    "librispeech_clean",
    "librispeech_other",
    "voxpopuli_aa",
    "tedlium",
    "gigaspeech",
    "spgispeech",
    "earnings22",
]


def extract_audio(audio_info):
    """Extract audio array and sampling rate from a HF dataset audio entry.

    Handles both the legacy dict format ({"array": ..., "sampling_rate": ...})
    and the newer AudioDecoder object from torchcodec-based datasets library.
    """
    if audio_info is None:
        return None, None
    try:
        audio_array = np.array(audio_info["array"])
        sampling_rate = int(audio_info["sampling_rate"])
        return audio_array, sampling_rate
    except (KeyError, TypeError, IndexError):
        return None, None


def format_entry(entry, dataset_name, audio_dir, spec, with_audio):
    """Format a dataset entry into JSONL and optionally save the audio file.

    Returns None to skip the sample (empty text, or shorter than min_duration).
    """
    text = entry[spec.text_field].strip()
    if not text:
        return None

    sample_id = str(entry[spec.id_field]).replace("/", "_").replace(":", "_")
    audio_filename = f"{Path(sample_id).stem}.flac"

    audio_array, sampling_rate = extract_audio(entry.get("audio"))
    duration = None
    if audio_array is not None and sampling_rate is not None:
        duration = len(audio_array) / sampling_rate
    else:
        # Fall back to a duration field when audio is not decoded (e.g. --no-audio).
        for key in ("duration", "audio_length_s"):
            if entry.get(key) is not None:
                duration = float(entry[key])
                break

    # Drop clips shorter than the dataset's minimum (e.g. GigaSpeech < 1s).
    if spec.min_duration is not None and duration is not None and duration < spec.min_duration:
        return None

    if with_audio and audio_array is not None and sampling_rate is not None:
        sf.write(str(audio_dir / audio_filename), audio_array, sampling_rate)

    user_message = {"role": "user", "content": "Transcribe the following audio."}
    audio_meta = {"path": f"/dataset/asr-leaderboard/data/{dataset_name}/{audio_filename}"}
    if duration is not None:
        audio_meta["duration"] = float(duration)
    user_message["audio"] = audio_meta

    formatted = {
        "task_type": "ASR",
        "expected_answer": text,
        "messages": [{"role": "system", "content": SYSTEM_MESSAGE}, user_message],
        "subset_for_metrics": dataset_name,
        "id": entry[spec.id_field],
    }
    if "speaker_id" in entry:
        formatted["speaker_id"] = entry["speaker_id"]

    return formatted


def load_hf_dataset(spec, with_audio):
    """Load a dataset and normalize it to expose a decoded ``audio`` column.

    For the ESB configs the audio column already exists. For the Artificial
    Analysis layout the wav files live in the dataset repo, so we download them
    and attach a decoded ``audio`` column pointing at the local paths.
    """
    if spec.loader == "aa_url":
        dataset = load_dataset(spec.repo, split=spec.split)
        if with_audio:
            from huggingface_hub import snapshot_download

            audio_root = snapshot_download(spec.repo, repo_type="dataset", allow_patterns=["audio/*"])
            dataset = dataset.map(lambda ex: {"audio": str(Path(audio_root) / ex["url"])})
            dataset = dataset.cast_column("audio", Audio(sampling_rate=AUDIO_SAMPLE_RATE))
        return dataset

    dataset = load_dataset(spec.repo, spec.config, split=spec.split)
    if with_audio and "audio" in dataset.column_names:
        dataset = dataset.cast_column("audio", Audio(sampling_rate=AUDIO_SAMPLE_RATE))
    return dataset


def prepare_dataset(dataset_name, output_dir, with_audio=True):
    """Download, decode, and write a single ASR dataset to JSONL + audio files."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")

    spec = DATASET_CONFIGS[dataset_name]

    print(f"Loading {dataset_name} from {spec.repo} (config={spec.config}, split={spec.split})...")
    dataset = load_hf_dataset(spec, with_audio)

    output_file = output_dir / f"{dataset_name}.jsonl"
    audio_dir = output_dir / "data" / dataset_name
    if with_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(dataset)} samples from {dataset_name}...")
    count = 0
    skipped = 0
    with open(output_file, "w", encoding="utf-8") as fout:
        for entry in tqdm(dataset, desc=dataset_name):
            formatted = format_entry(entry, dataset_name, audio_dir, spec, with_audio)
            if formatted is None:
                skipped += 1
                continue
            fout.write(json.dumps(formatted) + "\n")
            count += 1

    if skipped:
        print(f"Skipped {skipped} samples (empty text or shorter than {spec.min_duration}s)")
    print(f"Saved {count} samples to {output_file}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Prepare ASR Leaderboard datasets for evaluation")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["default"],
        choices=list(DATASET_CONFIGS.keys()) + ["all", "default"],
        help=(
            "Datasets to prepare. 'default' = Open ASR Leaderboard set "
            "(no AMI, cleaned VoxPopuli, GigaSpeech >=1s). 'all' = every dataset."
        ),
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip saving audio files (JSONL still includes audio paths)",
    )
    args = parser.parse_args()

    data_dir = Path("/dataset/asr-leaderboard")
    output_dir = data_dir if data_dir.exists() else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with_audio = not args.no_audio
    if not with_audio:
        print("Running without saving audio files.")

    if "all" in args.datasets:
        datasets_to_prepare = list(DATASET_CONFIGS.keys())
    elif "default" in args.datasets:
        datasets_to_prepare = list(DEFAULT_DATASETS)
    else:
        datasets_to_prepare = args.datasets

    total_samples = 0
    for dataset_name in datasets_to_prepare:
        total_samples += prepare_dataset(dataset_name, output_dir, with_audio=with_audio)

    # Combine only the datasets prepared in this run so stale JSONLs from earlier
    # runs (e.g. a previously prepared ami.jsonl) cannot leak into the average.
    combined_file = output_dir / "test.jsonl"
    print(f"\nCreating combined file: {combined_file}")

    dataset_files = [output_dir / f"{name}.jsonl" for name in datasets_to_prepare]
    dataset_files = [f for f in dataset_files if f.exists()]
    combined_count = 0
    with open(combined_file, "w", encoding="utf-8") as fout:
        for dataset_file in dataset_files:
            with open(dataset_file, encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
                    combined_count += 1
            print(f"  Added {dataset_file.name}")

    print(f"Combined {combined_count} samples from {len(dataset_files)} datasets into {combined_file}")
    print(f"\nTotal: {total_samples} samples prepared")


if __name__ == "__main__":
    main()
