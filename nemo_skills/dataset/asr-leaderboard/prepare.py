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

Datasets are sourced from the repositories used by the HuggingFace Open ASR
Leaderboard, using the leaderboard's own config names and session-level scoring
protocol so the prepared data matches the leaderboard exactly. In particular the
2026 protocol updates are reflected by consuming the pre-cleaned configs directly:

    - ami          -> config "ami_cleaned"          (sub-1s clips removed upstream)
    - gigaspeech   -> config "gigaspeech_cleaned"   (sub-1s clips removed upstream)
    - voxpopuli_aa -> config "voxpopuli_cleaned_aa" (Artificial Analysis cleaned refs)
    - monsoon_en   -> pinned VoiceArena English test set
    - earnings22_aa -> ArtificialAnalysis/Earnings22-Cleaned-AA-chunked

AA-Earnings22 contains 341 VAD-derived audio chunks but only six parent-call
references. Every prepared chunk carries parent_id and chunk_index;
OpenASRLeaderboardMetrics concatenates predictions in chunk order and scores one
transcript per call, matching open_asr_leaderboard PR #202.

Audio paths in JSONL: {NEMO_SKILLS_AUDIO_ROOT}/asr-leaderboard/data/{dataset}/{sample_id}.flac

Usage:
    ns prepare_data asr-leaderboard                          # full leaderboard set
    ns prepare_data asr-leaderboard --datasets librispeech_clean ami
    ns prepare_data asr-leaderboard --no-audio
"""

import argparse
import importlib
import io
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from huggingface_hub import snapshot_download
from tqdm import tqdm

from nemo_skills.evaluation.evaluator.open_asr_normalizer import OpenASREnglishTextNormalizer

LEADERBOARD_REPO = "hf-audio/open-asr-leaderboard"
LEADERBOARD_REVISION = "b6bdcd0beb34f8975dc659796176d88f43aff502"
_monsoon_prepare = importlib.import_module("nemo_skills.dataset.voicearena-monsoon.prepare")
MONSOON_ENGLISH_CONTRACT = _monsoon_prepare.SOURCE_CONTRACTS["en-in.public"]
SYSTEM_MESSAGE = "You are a helpful assistant. /no_think"
AUDIO_SAMPLE_RATE = 16000
AUDIO_ROOT = os.environ.get("NEMO_SKILLS_AUDIO_ROOT", "/data").rstrip("/")


@dataclass
class DatasetSpec:
    """A leaderboard dataset: which HF config/split to read and its fields."""

    config: str | None
    split: str
    text_field: str = "text"
    id_field: str = "id"
    repo: str = LEADERBOARD_REPO
    revision: str = LEADERBOARD_REVISION
    parent_revision: str | None = None
    parent_repo: str | None = None
    parent_text_field: str = "transcript"
    parent_id_field: str = "parent_id"
    audio_dir: str | None = None
    audio_filename_field: str = "file_name"
    chunk_index_field: str = "chunk_index"

    @property
    def is_chunked(self) -> bool:
        return self.parent_repo is not None


# Friendly subset keys map to the official 2026 public leaderboard sources.
DATASET_CONFIGS = {
    "librispeech_clean": DatasetSpec("librispeech", "test.clean"),
    "librispeech_other": DatasetSpec("librispeech", "test.other"),
    "voxpopuli_aa": DatasetSpec("voxpopuli_cleaned_aa", "test"),
    "gigaspeech": DatasetSpec("gigaspeech_cleaned", "test"),
    "spgispeech": DatasetSpec("spgispeech", "test"),
    "earnings22_aa": DatasetSpec(
        None,
        "test",
        repo="ArtificialAnalysis/Earnings22-Cleaned-AA-chunked",
        revision="27f7d0056d593231974a752ce03c1e965da69b3f",
        parent_repo="ArtificialAnalysis/Earnings22-Cleaned-AA",
        parent_revision="5999936ee3d8d3cde160e3c65a8cc6ff5b7c5276",
        audio_dir="audio",
    ),
    "ami": DatasetSpec("ami_cleaned", "test"),
    "monsoon_en": DatasetSpec(
        None,
        "test",
        repo=MONSOON_ENGLISH_CONTRACT.repo,
        revision=MONSOON_ENGLISH_CONTRACT.revision,
    ),
}

_NORMALIZER = OpenASREnglishTextNormalizer()


def is_target_text_in_range(text: str) -> bool:
    """Apply the current Open ASR Leaderboard reference filter."""
    if text.strip() == "ignore time segment in scoring":
        return False
    return bool(_NORMALIZER(text).strip())


def extract_audio(audio_info):
    """Extract (audio_array, sampling_rate) from a HF audio value.

    Prefers the undecoded ``{"bytes", "path"}`` form (from ``Audio(decode=False)``)
    read with soundfile, so no torchcodec/ffmpeg dependency is required. Also
    accepts the already-decoded ``{"array", "sampling_rate"}`` form. Leaderboard
    configs are 16 kHz, so the stored bytes need no resampling.
    """
    if audio_info is None:
        return None, None
    if isinstance(audio_info, (str, Path)):
        audio_array, sampling_rate = sf.read(str(audio_info), dtype="float32")
        if getattr(audio_array, "ndim", 1) > 1:
            audio_array = audio_array.mean(axis=1)
        return np.asarray(audio_array), int(sampling_rate)
    if isinstance(audio_info, dict) and audio_info.get("bytes") is not None:
        audio_array, sampling_rate = sf.read(io.BytesIO(audio_info["bytes"]), dtype="float32")
        if getattr(audio_array, "ndim", 1) > 1:
            audio_array = audio_array.mean(axis=1)
        return np.asarray(audio_array), int(sampling_rate)
    if isinstance(audio_info, dict) and audio_info.get("path"):
        audio_array, sampling_rate = sf.read(str(audio_info["path"]), dtype="float32")
        if getattr(audio_array, "ndim", 1) > 1:
            audio_array = audio_array.mean(axis=1)
        return np.asarray(audio_array), int(sampling_rate)
    try:
        return np.array(audio_info["array"]), int(audio_info["sampling_rate"])
    except (KeyError, TypeError, IndexError):
        return None, None


def format_entry(entry, dataset_name, audio_dir, spec, with_audio, reference_text=None, audio_info=None):
    """Format a dataset entry into JSONL and optionally save the audio file."""
    text = (entry[spec.text_field] if reference_text is None else reference_text).strip()
    if not is_target_text_in_range(text):
        return None

    sample_id = str(entry[spec.id_field]).replace("/", "_").replace(":", "_")
    audio_filename = f"{Path(sample_id).stem}{'.wav' if dataset_name == 'monsoon_en' else '.flac'}"

    audio_array, sampling_rate = extract_audio(entry.get("audio") if audio_info is None else audio_info)
    if dataset_name == "monsoon_en" and with_audio:
        audio_array, sampling_rate = _monsoon_prepare._decode_audio(entry.get("audio"))
    duration = None
    if audio_array is not None and sampling_rate is not None:
        if sampling_rate != AUDIO_SAMPLE_RATE:
            raise ValueError(
                f"Expected {AUDIO_SAMPLE_RATE} Hz leaderboard audio, got {sampling_rate} Hz for {sample_id}"
            )
        duration = len(audio_array) / sampling_rate
        if with_audio:
            sf.write(str(audio_dir / audio_filename), audio_array, sampling_rate)
    elif entry.get("audio_length_s", entry.get("duration")) is not None:
        duration = float(entry.get("audio_length_s", entry.get("duration")))

    prompt = "Transcribe the following audio."
    if dataset_name == "monsoon_en":
        prompt += " " + _monsoon_prepare.script_instruction("en-in")
    user_message = {"role": "user", "content": prompt}
    audio_meta = {"path": f"{AUDIO_ROOT}/asr-leaderboard/data/{dataset_name}/{audio_filename}"}
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
    if spec.is_chunked:
        formatted["parent_id"] = entry[spec.parent_id_field]
        formatted["chunk_index"] = int(entry[spec.chunk_index_field])

    return formatted


def load_source_dataset(spec: DatasetSpec):
    """Load a regular or chunked leaderboard dataset without decoding audio."""
    if spec.config is None:
        return load_dataset(spec.repo, split=spec.split, revision=spec.revision)
    return load_dataset(spec.repo, spec.config, split=spec.split, revision=spec.revision)


def load_parent_references(spec: DatasetSpec) -> dict[str, str]:
    """Load one transcript per parent session for a chunked dataset."""
    if not spec.is_chunked:
        return {}
    parent_dataset = load_dataset(spec.parent_repo, split=spec.split, revision=spec.parent_revision)
    return {str(entry[spec.id_field]): entry[spec.parent_text_field] for entry in parent_dataset}


def prepare_dataset(dataset_name, output_dir, with_audio=True):
    """Download, decode, and write a single ASR dataset to JSONL + audio files."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")

    spec = DATASET_CONFIGS[dataset_name]

    print(f"Loading {dataset_name} from {spec.repo} (config={spec.config}, split={spec.split})...")
    dataset = load_source_dataset(spec)
    if with_audio and "audio" in dataset.column_names:
        # decode=False -> raw encoded bytes, read with soundfile (no torchcodec).
        dataset = dataset.cast_column("audio", Audio(decode=False))

    if dataset_name == "monsoon_en":
        _monsoon_prepare._validate_metadata(
            [dict(row) for row in dataset.remove_columns("audio")], MONSOON_ENGLISH_CONTRACT
        )

    parent_references = load_parent_references(spec)
    if spec.is_chunked:
        missing_parent_ids = sorted({str(entry[spec.parent_id_field]) for entry in dataset} - set(parent_references))
        if missing_parent_ids:
            raise ValueError(
                f"No transcript found in {spec.parent_repo} for parent id(s) {missing_parent_ids}. "
                "The chunked and parent datasets are out of sync."
            )

    audio_root = None
    if with_audio and spec.audio_dir is not None:
        audio_root = Path(
            snapshot_download(
                repo_id=spec.repo,
                repo_type="dataset",
                revision=spec.revision,
                allow_patterns=[f"{spec.audio_dir}/*"],
            )
        )

    output_file = output_dir / f"{dataset_name}.jsonl"
    audio_dir = output_dir / "data" / dataset_name
    if with_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(dataset)} samples from {dataset_name}...")
    count = 0
    with open(output_file, "w", encoding="utf-8") as fout:
        for entry in tqdm(dataset, desc=dataset_name):
            reference_text = None
            audio_info = None
            if spec.is_chunked:
                reference_text = parent_references[str(entry[spec.parent_id_field])]
                if audio_root is not None:
                    audio_info = audio_root / spec.audio_dir / entry[spec.audio_filename_field]
            formatted = format_entry(
                entry,
                dataset_name,
                audio_dir,
                spec,
                with_audio,
                reference_text=reference_text,
                audio_info=audio_info,
            )
            if formatted is None:
                continue
            fout.write(json.dumps(formatted) + "\n")
            count += 1

    print(f"Saved {count} samples to {output_file}")
    # Dotted evaluation targets resolve to
    # ``asr-leaderboard/<subset>/test.jsonl``. Publish the exact same rows there
    # so a subset can be regenerated with its own prompt without rerunning the
    # other six leaderboard datasets.
    leaf_dir = output_dir / dataset_name
    leaf_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output_file, leaf_dir / "test.jsonl")
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
