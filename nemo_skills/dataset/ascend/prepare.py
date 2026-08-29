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

"""Prepare the official ASCEND test split for NeMo Skills."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

HF_REPO_ID = "CAiRE/ASCEND"
HF_REVISION = "737e9800ae31be9932ba8464c80366559bd28424"
OFFICIAL_SPLIT = "test"
OFFICIAL_TEST_SIZE = 1315
PROMPT = "Transcribe the following audio."


def _container_audio_path(sample_id: str) -> str:
    audio_root = os.getenv("NEMO_SKILLS_AUDIO_ROOT", "/data").rstrip("/")
    return f"{audio_root}/ascend/audio/{sample_id}.wav"


def _decode_audio(audio: dict) -> tuple[np.ndarray, int]:
    if audio.get("bytes") is not None:
        samples, sampling_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
    elif audio.get("path"):
        samples, sampling_rate = sf.read(audio["path"], dtype="float32")
    else:
        raise ValueError("ASCEND audio row has neither bytes nor a path")
    samples = np.asarray(samples)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples, int(sampling_rate)


def _safe_id(value: object, index: int) -> str:
    sample_id = str(value if value is not None else f"ascend-test-{index}").strip()
    return sample_id.replace("/", "_").replace(":", "_")


def _format_entry(row: dict, index: int) -> dict:
    sample_id = _safe_id(row.get("id"), index)
    duration = float(row.get("duration") or 0.0)
    audio_path = _container_audio_path(sample_id)
    audio = {"path": audio_path, "duration": duration}
    return {
        "id": sample_id,
        "expected_answer": str(row["transcription"]),
        "audio_path": audio_path,
        "duration": duration,
        "messages": [{"role": "user", "content": PROMPT, "audio": audio}],
        "subset_for_metrics": OFFICIAL_SPLIT,
        "task_type": "Code-Switched-ASR",
        "extra_fields": {
            "language": row.get("language"),
            "speaker_id": row.get("original_speaker_id"),
            "session_id": row.get("session_id"),
            "topic": row.get("topic"),
            "source_repo": HF_REPO_ID,
            "source_revision": HF_REVISION,
            "source_split": OFFICIAL_SPLIT,
        },
    }


def prepare_ascend(data_dir: Path, no_audio: bool = False, max_samples: int | None = None) -> Path:
    """Download and format ASCEND's official test split.

    max_samples exists only for smoke preparation. Complete preparation is
    checked against the official 1,315-row test-set cardinality.
    """
    from datasets import Audio, load_dataset

    dataset = load_dataset(HF_REPO_ID, split=OFFICIAL_SPLIT, revision=HF_REVISION)
    if max_samples is None and len(dataset) != OFFICIAL_TEST_SIZE:
        raise ValueError(f"Expected {OFFICIAL_TEST_SIZE} ASCEND test rows, found {len(dataset)}")
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    if "audio" not in dataset.column_names:
        raise ValueError(f"ASCEND dataset has no audio column: {dataset.column_names}")
    dataset = dataset.cast_column("audio", Audio(decode=False))

    data_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = data_dir / "audio"
    if not no_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / "test.jsonl"
    with open(output_path, "w", encoding="utf-8") as output:
        for index, row in enumerate(dataset):
            record = _format_entry(dict(row), index)
            if not no_audio:
                samples, sampling_rate = _decode_audio(row["audio"])
                sf.write(audio_dir / f"{record['id']}.wav", samples, sampling_rate)
                if record["duration"] <= 0:
                    record["duration"] = len(samples) / sampling_rate
                    record["messages"][0]["audio"]["duration"] = record["duration"]
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the official ASCEND test split")
    parser.add_argument("--data_dir", default=os.getenv("NEMO_SKILLS_DATA_DIR"))
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--max-samples", type=int, default=-1, help="Smoke-test row limit")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) / "ascend" if args.data_dir else Path(__file__).parent
    output_path = prepare_ascend(
        data_dir,
        no_audio=args.no_audio,
        max_samples=args.max_samples if args.max_samples > 0 else None,
    )
    print(f"ASCEND prepared at {output_path}")


if __name__ == "__main__":
    main()
