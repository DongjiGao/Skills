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

Datasets are sourced from the official HuggingFace Open ASR Leaderboard dataset
(hf-audio/open-asr-leaderboard), using the leaderboard's own config names so the
prepared data matches the leaderboard exactly. In particular the 2026 protocol
update is reflected by consuming the pre-cleaned configs directly:

    - ami          -> config "ami_cleaned"          (sub-1s clips removed upstream)
    - gigaspeech   -> config "gigaspeech_cleaned"   (sub-1s clips removed upstream)
    - voxpopuli_aa -> config "voxpopuli_cleaned_aa" (Artificial Analysis cleaned refs)

The friendly subset key (used for per-dataset metrics and the JSONL filename) is
kept stable (ami, gigaspeech, voxpopuli_aa) even though it maps to a cleaned
config, so the reported leaderboard breakdown reads naturally.

Audio paths in JSONL: /dataset/asr-leaderboard/data/{dataset}/{sample_id}.flac

Usage:
    ns prepare_data asr-leaderboard                          # full leaderboard set
    ns prepare_data asr-leaderboard --datasets librispeech_clean ami
    ns prepare_data asr-leaderboard --no-audio
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm

LEADERBOARD_REPO = "hf-audio/open-asr-leaderboard"
SYSTEM_MESSAGE = "You are a helpful assistant. /no_think"
AUDIO_SAMPLE_RATE = 16000


@dataclass
class DatasetSpec:
    """A leaderboard dataset: which HF config/split to read and its fields."""

    config: str
    split: str
    text_field: str = "text"
    id_field: str = "id"
    repo: str = LEADERBOARD_REPO


# key = subset_for_metrics (friendly leaderboard name); config = official config.
# ami/gigaspeech/voxpopuli_aa map to the pre-cleaned configs (2026 protocol).
DATASET_CONFIGS = {
    "librispeech_clean": DatasetSpec("librispeech", "test.clean"),
    "librispeech_other": DatasetSpec("librispeech", "test.other"),
    "voxpopuli_aa": DatasetSpec("voxpopuli_cleaned_aa", "test"),
    "tedlium": DatasetSpec("tedlium", "test"),
    "gigaspeech": DatasetSpec("gigaspeech_cleaned", "test"),
    "spgispeech": DatasetSpec("spgispeech", "test"),
    "earnings22": DatasetSpec("earnings22", "test"),
    "ami": DatasetSpec("ami_cleaned", "test"),
}


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
    """Format a dataset entry into JSONL and optionally save the audio file."""
    text = entry[spec.text_field].strip()
    if not text:
        return None

    sample_id = str(entry[spec.id_field]).replace("/", "_").replace(":", "_")
    audio_filename = f"{Path(sample_id).stem}.flac"

    audio_array, sampling_rate = extract_audio(entry.get("audio"))
    duration = None
    if audio_array is not None and sampling_rate is not None:
        duration = len(audio_array) / sampling_rate
        if with_audio:
            sf.write(str(audio_dir / audio_filename), audio_array, sampling_rate)
    elif entry.get("audio_length_s") is not None:
        duration = float(entry["audio_length_s"])

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


def prepare_dataset(dataset_name, output_dir, with_audio=True):
    """Download, decode, and write a single ASR dataset to JSONL + audio files."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")

    spec = DATASET_CONFIGS[dataset_name]

    print(f"Loading {dataset_name} from {spec.repo} (config={spec.config}, split={spec.split})...")
    dataset = load_dataset(spec.repo, spec.config, split=spec.split)
    if with_audio and "audio" in dataset.column_names:
        dataset = dataset.cast_column("audio", Audio(sampling_rate=AUDIO_SAMPLE_RATE))

    output_file = output_dir / f"{dataset_name}.jsonl"
    audio_dir = output_dir / "data" / dataset_name
    if with_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(dataset)} samples from {dataset_name}...")
    count = 0
    with open(output_file, "w", encoding="utf-8") as fout:
        for entry in tqdm(dataset, desc=dataset_name):
            formatted = format_entry(entry, dataset_name, audio_dir, spec, with_audio)
            if formatted is None:
                continue
            fout.write(json.dumps(formatted) + "\n")
            count += 1

    print(f"Saved {count} samples to {output_file}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Prepare ASR Leaderboard datasets for evaluation")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=list(DATASET_CONFIGS.keys()) + ["all"],
        help="Datasets to prepare. 'all' (default) = the full Open ASR Leaderboard set.",
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
    else:
        datasets_to_prepare = args.datasets

    total_samples = 0
    for dataset_name in datasets_to_prepare:
        total_samples += prepare_dataset(dataset_name, output_dir, with_audio=with_audio)

    # Combine only the datasets prepared in this run so stale JSONLs from earlier
    # runs cannot leak into the average.
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
