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

"""Prepare HALAS for NeMo Skills evaluation.

The HALAS metadata is small, but its source audio lives in the Earnings22
``chunked`` split. This script writes NeMo-Skills JSONL manifests for the HALAS
train/test splits and optionally extracts only the referenced Earnings22 clips
to ``<data_dir>/audio``.
"""

import argparse
import json
import os
import re
import wave
from pathlib import Path
from typing import Iterable

import numpy as np

HF_REPO_ID = "MatBar99/HALAS"
EARNINGS22_REPO_ID = "distil-whisper/earnings22"
EARNINGS22_CONFIG = "chunked"
EARNINGS22_SPLIT = "test"
BENCHMARK_NAME = "halas"
PROMPT = (
    "Transcribe the English audio into text. Return only the words spoken in the audio; "
    "do not add explanations or continue beyond the audible speech."
)


def _default_data_dir() -> Path:
    if os.environ.get("NEMO_SKILLS_DATA_DIR"):
        return Path(os.environ["NEMO_SKILLS_DATA_DIR"]) / BENCHMARK_NAME
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent.parent / f"{parent.name}-data" / BENCHMARK_NAME
    return Path.home() / ".cache" / "nemo-skills-data" / BENCHMARK_NAME


def _default_audio_prefix(args_data_dir: str | None, data_dir: Path) -> str:
    if args_data_dir is None and os.environ.get("NEMO_SKILLS_DATA_DIR"):
        return f"/data/{BENCHMARK_NAME}"
    return str(data_dir)


def _write_wav(path: Path, samples: np.ndarray, sampling_rate: int) -> None:
    if samples.ndim > 1:
        samples = samples.mean(axis=tuple(range(1, samples.ndim)))
    if np.issubdtype(samples.dtype, np.floating):
        samples = np.clip(samples, -1.0, 1.0)
        samples_i16 = (samples * 32767.0).astype(np.int16)
    else:
        samples_i16 = samples.astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sampling_rate))
        wf.writeframes(samples_i16.tobytes())


def _as_str(value) -> str:
    if value is None:
        return ""
    return str(value)


def _canonical_audio_id(value: str) -> str:
    return Path(_as_str(value).strip()).name


def _reference_text(row: dict) -> str:
    corrected = _as_str(row.get("corrected_reference_text")).strip()
    if corrected:
        return corrected
    return _as_str(row.get("e22_reference_text")).strip()


def _load_halas_rows(max_samples: int | None = None) -> list[dict]:
    from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

    loaded = load_dataset(HF_REPO_ID)
    if isinstance(loaded, DatasetDict):
        datasets = [ds for ds in loaded.values()]
        ds = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
    elif isinstance(loaded, Dataset):
        ds = loaded
    else:
        ds = load_dataset(HF_REPO_ID, split="train")

    rows = [dict(row) for row in ds]
    if max_samples is not None and max_samples > 0:
        rows = rows[:max_samples]
    return rows


def _earnings22_audio_id(row: dict) -> str:
    segment_id = _as_str(row["segment_id"]).strip()
    file_id = _as_str(row["file_id"]).strip()
    return f"{segment_id}_{file_id}.wav"


def _extract_audio_for_rows(rows: Iterable[dict], data_dir: Path) -> set[str]:
    from datasets import load_dataset

    audio_dir = data_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    target_ids = {_canonical_audio_id(row["audio_id"]) for row in rows}
    missing_ids = set(target_ids)

    print("=" * 70)
    print(f"Loading {EARNINGS22_REPO_ID}/{EARNINGS22_CONFIG} split={EARNINGS22_SPLIT}")
    print(f"Extracting {len(target_ids)} HALAS clips to {audio_dir}")
    print("=" * 70)

    ds = load_dataset(EARNINGS22_REPO_ID, EARNINGS22_CONFIG, split=EARNINGS22_SPLIT)
    written = 0
    for e22_row in ds:
        audio_id = _earnings22_audio_id(e22_row)
        if audio_id not in target_ids:
            continue

        wav_path = audio_dir / audio_id
        if not wav_path.exists():
            audio = e22_row["audio"]
            _write_wav(wav_path, np.asarray(audio["array"]), int(audio["sampling_rate"]))
            written += 1
        missing_ids.discard(audio_id)
        if not missing_ids:
            break

    print(f"Wrote {written} new audio files; {len(missing_ids)} referenced clips were not found in Earnings22.")
    if missing_ids:
        preview = ", ".join(sorted(missing_ids)[:10])
        print(f"Missing HALAS audio IDs (first 10): {preview}")
    return missing_ids


def _model_label_counts(row: dict) -> dict[str, int]:
    counts = {"hallucination_or_looping": 0, "no_hallucination": 0}
    for key, value in row.items():
        if not key.endswith("_label"):
            continue
        label = _as_str(value).strip().lower()
        if not label:
            continue
        if label == "no hallucination":
            counts["no_hallucination"] += 1
        elif "hallucination" in label or "looping" in label:
            counts["hallucination_or_looping"] += 1
        else:
            counts["no_hallucination"] += 1
    return counts


def _build_messages(audio_path: str, duration: float) -> list[dict]:
    return [
        {
            "role": "user",
            "content": PROMPT,
            "audio": {"path": audio_path, "duration": float(duration)},
        }
    ]


def _format_entry(row: dict, audio_prefix: str) -> dict:
    audio_id = _canonical_audio_id(row["audio_id"])
    audio_path = f"{audio_prefix.rstrip('/')}/audio/{audio_id}"
    reference = _reference_text(row)
    label_counts = _model_label_counts(row)
    return {
        "messages": _build_messages(audio_path, float(row.get("audio_duration") or 0.0)),
        "expected_answer": reference,
        "expected_answer_cased": _as_str(row.get("e22_reference_text")).strip(),
        "audio_id": audio_id,
        "audio_duration": float(row.get("audio_duration") or 0.0),
        "audio_filepath": audio_path,
        "subset_for_metrics": _as_str(row.get("split") or "test"),
        "halas_released_system_hallucination_count": label_counts["hallucination_or_looping"],
        "halas_released_system_no_hallucination_count": label_counts["no_hallucination"],
        "task_type": "HALAS-ASR",
    }


def _split_rows(rows: Iterable[dict]) -> dict[str, list[dict]]:
    splits: dict[str, list[dict]] = {"train": [], "test": []}
    for row in rows:
        split = _as_str(row.get("split")).strip().lower() or "test"
        split = re.sub(r"[^a-z0-9_]+", "_", split)
        splits.setdefault(split, []).append(row)
    return {split: split_rows for split, split_rows in splits.items() if split_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HALAS manifests for NeMo Skills")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory for extracted Earnings22 audio")
    parser.add_argument("--audio-prefix", type=str, default=None, help="Audio path prefix written into JSONL")
    parser.add_argument("--no-audio", action="store_true", help="Skip Earnings22 audio extraction")
    parser.add_argument("--max-samples", type=int, default=-1, help="Limit metadata rows for smoke preparation")
    args = parser.parse_args()

    output_dir = Path(__file__).parent
    data_dir = Path(args.data_dir) if args.data_dir else _default_data_dir()
    audio_prefix = args.audio_prefix if args.audio_prefix else _default_audio_prefix(args.data_dir, data_dir)

    rows = _load_halas_rows(max_samples=args.max_samples if args.max_samples > 0 else None)
    if not rows:
        raise ValueError("No HALAS rows were loaded from HuggingFace.")

    if not args.no_audio:
        _extract_audio_for_rows(rows, data_dir)

    split_rows = _split_rows(rows)
    print("\nWriting HALAS JSONL manifests...")
    for split, rows_for_split in split_rows.items():
        output_path = output_dir / f"{split}.jsonl"
        with open(output_path, "w", encoding="utf-8") as fout:
            for row in rows_for_split:
                fout.write(json.dumps(_format_entry(row, audio_prefix), ensure_ascii=False) + "\n")
        print(f"  {split}: wrote {len(rows_for_split)} rows to {output_path}")


if __name__ == "__main__":
    main()
