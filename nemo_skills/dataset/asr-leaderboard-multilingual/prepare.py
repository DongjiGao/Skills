# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Prepare the HF Open ASR multilingual leaderboard datasets for evaluation.

Data is sourced from the official HuggingFace multilingual benchmark aggregate
(`hf-audio/open-asr-leaderboard-multilingual-datasets`). Config names match the
leaderboard scripts exactly: ``{dataset}_{language}``.

Audio paths in JSONL:
    /dataset/asr-leaderboard-multilingual/data/{dataset}/{sample_id}.flac

Usage:
    ns prepare_data asr-leaderboard-multilingual
    ns prepare_data asr-leaderboard-multilingual --datasets fleurs_de mcv_de
    ns prepare_data asr-leaderboard-multilingual --no-audio
"""

import argparse
import importlib
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm

LEADERBOARD_REPO = "hf-audio/open-asr-leaderboard-multilingual-datasets"
LEADERBOARD_REVISION = "b791fc9151221e5b7e59c6c2dfa4dee09dda3cb7"
SYSTEM_MESSAGE = "You are a helpful assistant. /no_think"
AUDIO_ROOT = os.environ.get("NEMO_SKILLS_AUDIO_ROOT", "/dataset").rstrip("/")
TEXT_FIELDS = ("text", "sentence", "normalized_text", "transcript", "transcription")
ID_FIELDS = ("id", "audio_filepath", "path", "file", "file_name", "audio_id")


@dataclass(frozen=True)
class DatasetSpec:
    """A multilingual leaderboard dataset config and its scoring language."""

    config: str | None
    language: str
    split: str = "test"
    repo: str = LEADERBOARD_REPO
    revision: str | None = LEADERBOARD_REVISION


_monsoon_prepare = importlib.import_module("nemo_skills.dataset.voicearena-monsoon.prepare")
MONSOON_HINDI_CONTRACT = _monsoon_prepare.SOURCE_CONTRACTS["hi-in.public"]


# Matches the official public multilingual evaluation matrix.
DATASET_CONFIGS = {
    "fleurs_de": DatasetSpec("fleurs_de", "de"),
    "fleurs_fr": DatasetSpec("fleurs_fr", "fr"),
    "fleurs_it": DatasetSpec("fleurs_it", "it"),
    "fleurs_es": DatasetSpec("fleurs_es", "es"),
    "fleurs_pt": DatasetSpec("fleurs_pt", "pt"),
    "fleurs_nl": DatasetSpec("fleurs_nl", "nl"),
    "mcv_de": DatasetSpec("mcv_de", "de"),
    "mcv_es": DatasetSpec("mcv_es", "es"),
    "mcv_fr": DatasetSpec("mcv_fr", "fr"),
    "mcv_it": DatasetSpec("mcv_it", "it"),
    "mcv_nl": DatasetSpec("mcv_nl", "nl"),
    "mls_es": DatasetSpec("mls_es", "es"),
    "mls_fr": DatasetSpec("mls_fr", "fr"),
    "mls_it": DatasetSpec("mls_it", "it"),
    "mls_pt": DatasetSpec("mls_pt", "pt"),
    "mls_nl": DatasetSpec("mls_nl", "nl"),
    "monsoon_hi": DatasetSpec(
        None,
        "hi",
        repo=MONSOON_HINDI_CONTRACT.repo,
        revision=MONSOON_HINDI_CONTRACT.revision,
    ),
}

LANGUAGE_DATASETS = {
    "de": ("fleurs_de", "mcv_de"),
    "fr": ("fleurs_fr", "mcv_fr", "mls_fr"),
    "it": ("fleurs_it", "mcv_it", "mls_it"),
    "es": ("fleurs_es", "mcv_es", "mls_es"),
    "pt": ("fleurs_pt", "mls_pt"),
    "nl": ("fleurs_nl", "mcv_nl", "mls_nl"),
    "hi": ("monsoon_hi",),
}


def extract_audio(audio_info):
    """Extract (audio_array, sampling_rate) from a HF audio value."""
    if audio_info is None:
        return None, None
    if isinstance(audio_info, dict) and audio_info.get("bytes") is not None:
        audio_array, sampling_rate = sf.read(io.BytesIO(audio_info["bytes"]), dtype="float32")
        if getattr(audio_array, "ndim", 1) > 1:
            audio_array = audio_array.mean(axis=1)
        return np.asarray(audio_array), int(sampling_rate)
    try:
        return np.array(audio_info["array"]), int(audio_info["sampling_rate"])
    except (KeyError, TypeError, IndexError):
        return None, None


def get_text(entry) -> str:
    """Return the transcript field using the official fallback order."""
    for field in TEXT_FIELDS:
        if field in entry and entry[field] is not None:
            return str(entry[field])
    raise ValueError(f"Expected one of transcript fields {TEXT_FIELDS}. Got fields: {sorted(entry.keys())}")


def get_sample_id(entry, dataset_name: str, index: int) -> str:
    """Return a stable sample id from common HF dataset metadata fields."""
    for field in ID_FIELDS:
        value = entry.get(field)
        if value is not None and str(value).strip():
            return str(value)
    audio = entry.get("audio")
    if isinstance(audio, dict) and audio.get("path"):
        return str(audio["path"])
    return f"{dataset_name}-{index}"


def format_entry(entry, index, dataset_name, audio_dir, spec, with_audio):
    """Format one dataset entry into the Skills audio JSONL schema."""
    reference_lattice = entry.get("lattice") if dataset_name == "monsoon_hi" else None
    text = (
        " ".join(slot[0] for slot in reference_lattice) if reference_lattice is not None else get_text(entry).strip()
    )
    if not text:
        return None

    sample_id = get_sample_id(entry, dataset_name, index).replace("/", "_").replace(":", "_")
    audio_filename = f"{Path(sample_id).stem}{'.wav' if dataset_name == 'monsoon_hi' else '.flac'}"

    if dataset_name == "monsoon_hi" and with_audio:
        audio_array, sampling_rate = _monsoon_prepare._decode_audio(entry.get("audio"))
    else:
        audio_array, sampling_rate = extract_audio(entry.get("audio"))
    duration = None
    if audio_array is not None and sampling_rate is not None:
        duration = len(audio_array) / sampling_rate
        if with_audio:
            if dataset_name == "monsoon_hi":
                sf.write(
                    str(audio_dir / audio_filename),
                    audio_array,
                    sampling_rate,
                    format="WAV",
                    subtype=_monsoon_prepare.EXPECTED_SUBTYPE,
                )
                _monsoon_prepare._validate_wav(audio_dir / audio_filename, duration)
            else:
                sf.write(str(audio_dir / audio_filename), audio_array, sampling_rate)
    elif entry.get("audio_length_s") is not None:
        duration = float(entry["audio_length_s"])

    user_message = {"role": "user", "content": "Transcribe the following audio."}
    audio_meta = {"path": f"{AUDIO_ROOT}/asr-leaderboard-multilingual/data/{dataset_name}/{audio_filename}"}
    if duration is not None:
        audio_meta["duration"] = float(duration)
    user_message["audio"] = audio_meta

    formatted = {
        "task_type": "Multilingual-ASR",
        "expected_answer": text,
        "messages": [{"role": "system", "content": SYSTEM_MESSAGE}, user_message],
        "subset_for_metrics": dataset_name,
        "id": sample_id,
        "extra_fields": {
            "src_lang": spec.language,
            "remove_diacritics": False,
            "normalize_compound": True,
            "merge_compounds": True,
            "use_cer": False,
        },
    }

    if "speaker_id" in entry:
        formatted["speaker_id"] = entry["speaker_id"]
    if reference_lattice is not None:
        formatted["reference_lattice"] = reference_lattice
        formatted["monsoon_language"] = "hi-in"
        formatted["extra_fields"].update(
            source_repo=spec.repo,
            source_revision=spec.revision,
            source_split=spec.split,
            source_license=_monsoon_prepare.SOURCE_LICENSE,
        )

    return formatted


def prepare_dataset(dataset_name, output_dir, with_audio=True):
    """Download, decode, and write a single multilingual leaderboard config."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")

    spec = DATASET_CONFIGS[dataset_name]

    print(f"Loading {dataset_name} from {spec.repo} (config={spec.config}, split={spec.split})...")
    load_kwargs = {"split": spec.split}
    if spec.revision is not None:
        load_kwargs["revision"] = spec.revision
    if spec.config is None:
        dataset = load_dataset(spec.repo, **load_kwargs)
    else:
        dataset = load_dataset(spec.repo, spec.config, **load_kwargs)
    if (with_audio or dataset_name == "monsoon_hi") and "audio" in dataset.column_names:
        dataset = dataset.cast_column("audio", Audio(decode=False))

    if dataset_name == "monsoon_hi":
        metadata_rows = [dict(row) for row in dataset.remove_columns("audio")]
        validated_rows = _monsoon_prepare._validate_metadata(metadata_rows, MONSOON_HINDI_CONTRACT)
    else:
        validated_rows = None

    output_file = output_dir / f"{dataset_name}.jsonl"
    audio_dir = output_dir / "data" / dataset_name
    if with_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(dataset)} samples from {dataset_name}...")
    count = 0
    with open(output_file, "w", encoding="utf-8") as fout:
        for index, entry in enumerate(tqdm(dataset, desc=dataset_name)):
            if validated_rows is not None:
                entry = {**validated_rows[index], "audio": entry.get("audio")}
            formatted = format_entry(entry, index, dataset_name, audio_dir, spec, with_audio)
            if formatted is None:
                continue
            fout.write(json.dumps(formatted, ensure_ascii=False) + "\n")
            count += 1

    print(f"Saved {count} samples to {output_file}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Prepare Open ASR multilingual leaderboard datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=list(DATASET_CONFIGS.keys()) + ["all"],
        help="Datasets to prepare. 'all' (default) = the full public multilingual leaderboard set.",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip saving audio files (JSONL still includes audio paths)",
    )
    args = parser.parse_args()

    data_dir = Path("/dataset/asr-leaderboard-multilingual")
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
