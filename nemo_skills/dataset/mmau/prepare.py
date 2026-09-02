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

"""Prepare public MMAU test-mini rows for the speech-and-sound profile.

Official sources:
  Metadata/scorer: https://github.com/Sakshi113/MMAU
  Public audio:    https://huggingface.co/datasets/gamma-lab-umd/MMAU-test-mini

The canonical GitHub metadata is used because it contains the exact clean
choice strings expected by the official set-token scorer. Audio is joined by
the immutable sample ID from the pinned Hugging Face release.
"""

import argparse
import json
import os
import shutil
import urllib.request
import wave
from collections.abc import Iterable
from pathlib import Path

HF_REPO_ID = "gamma-lab-umd/MMAU-test-mini"
HF_REVISION = "ccd9696c0111ea7060827598f310558df0b71b0a"
OFFICIAL_REPO_REVISION = "110127f54c0dfba3faa5ec9feee4a7e4148679c5"
METADATA_URL = f"https://raw.githubusercontent.com/Sakshi113/MMAU/{OFFICIAL_REPO_REVISION}/mmau-test-mini.json"
BENCHMARK_NAME = "mmau"
FILTER_TASKS = frozenset({"speech", "sound"})
EXPECTED_TEST_ROWS = 666
ANSWER_INSTRUCTION = "Respond with only the text of the correct option."


def _default_data_dir() -> Path:
    if os.environ.get("NEMO_SKILLS_DATA_DIR"):
        return Path(os.environ["NEMO_SKILLS_DATA_DIR"]) / BENCHMARK_NAME
    return Path(__file__).resolve().parent


def _default_audio_prefix(args_data_dir: str | None, data_dir: Path) -> str:
    if args_data_dir is None and os.environ.get("NEMO_SKILLS_DATA_DIR"):
        return f"/data/{BENCHMARK_NAME}"
    return str(data_dir)


def _download_metadata(data_dir: Path) -> Path:
    source_dir = data_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = source_dir / "mmau-test-mini.json"
    if metadata_path.is_file():
        return metadata_path

    temporary_path = metadata_path.with_suffix(".json.tmp")
    try:
        urllib.request.urlretrieve(METADATA_URL, temporary_path)
        os.replace(temporary_path, metadata_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return metadata_path


def _load_rows(data_dir: Path) -> list[dict]:
    with open(_download_metadata(data_dir), "rt", encoding="utf-8") as fin:
        rows = json.load(fin)
    if not isinstance(rows, list):
        raise TypeError(f"Expected a JSON list in MMAU test-mini metadata, got {type(rows).__name__}")
    return rows


def select_rows(rows: Iterable[dict], max_samples: int | None = None) -> list[dict]:
    """Select exactly public test-mini rows whose official task is speech or sound."""
    selected = [dict(row) for row in rows if row.get("split") == "test-mini" and row.get("task") in FILTER_TASKS]
    if max_samples is None and len(selected) != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"Pinned MMAU test-mini yielded {len(selected)} speech/sound rows; expected {EXPECTED_TEST_ROWS}."
        )
    if max_samples is not None:
        selected = selected[:max_samples]
    return selected


def _audio_filename(row: dict) -> str:
    name = Path(str(row["audio_id"])).name
    if not name or name in {".", ".."}:
        raise ValueError(f"Invalid MMAU audio_id: {row['audio_id']!r}")
    return name


def _hf_sample_id(row: dict) -> str:
    attributes = row.get("other_attributes")
    if isinstance(attributes, str):
        attributes = json.loads(attributes)
    if not isinstance(attributes, dict) or not attributes.get("id"):
        raise ValueError("MMAU Hugging Face row has no sample ID in other_attributes")
    return str(attributes["id"])


def _write_hf_audio(context: dict, destination: Path) -> None:
    if not isinstance(context, dict):
        raise TypeError(f"Expected decoded-disabled Audio mapping, got {type(context).__name__}")
    audio_bytes = context.get("bytes")
    source_path = context.get("path")
    if audio_bytes is not None:
        with open(destination, "wb") as fout:
            fout.write(audio_bytes)
    elif source_path:
        shutil.copyfile(source_path, destination)
    else:
        raise ValueError("MMAU audio row contains neither bytes nor a path")


def _materialize_audio(rows: Iterable[dict], data_dir: Path) -> None:
    from datasets import Audio, load_dataset

    audio_dir = data_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    row_by_id = {str(row["id"]): row for row in rows}
    missing_ids = {
        sample_id for sample_id, row in row_by_id.items() if not (audio_dir / _audio_filename(row)).is_file()
    }
    if not missing_ids:
        return

    dataset = load_dataset(HF_REPO_ID, split="test", revision=HF_REVISION)
    dataset = dataset.cast_column("context", Audio(decode=False))
    found: set[str] = set()
    for hf_row in dataset:
        sample_id = _hf_sample_id(hf_row)
        if sample_id not in missing_ids:
            continue
        destination = audio_dir / _audio_filename(row_by_id[sample_id])
        _write_hf_audio(hf_row["context"], destination)
        found.add(sample_id)
        if found == missing_ids:
            break

    unresolved = missing_ids - found
    if unresolved:
        preview = ", ".join(sorted(unresolved)[:10])
        raise FileNotFoundError(f"MMAU Hugging Face release is missing {len(unresolved)} selected IDs: {preview}")


def _wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wav_file:
            return wav_file.getnframes() / wav_file.getframerate()
    except (EOFError, OSError, wave.Error, ZeroDivisionError):
        return None


def _prompt(question: str, choices: list[str]) -> str:
    options = "\n".join(f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(choices))
    return (
        "Listen to the audio and answer the multiple-choice question.\n\n"
        f"Question: {question}\n\nOptions:\n{options}\n\n{ANSWER_INSTRUCTION}"
    )


def format_entry(row: dict, audio_prefix: str, local_audio_dir: Path | None = None) -> dict:
    """Preserve official question/choices/answer and add a neutral audio prompt."""
    choices = list(row["choices"])
    filename = _audio_filename(row)
    audio_path = f"{audio_prefix.rstrip('/')}/audio/{filename}"
    audio = {"path": audio_path}
    if local_audio_dir is not None:
        duration = _wav_duration(local_audio_dir / filename)
        if duration is not None:
            audio["duration"] = duration

    formatted = dict(row)
    formatted.update(
        {
            "source_audio_path": row["audio_id"],
            "audio_path": audio_path,
            "audio_filepath": audio_path,
            "messages": [{"role": "user", "content": _prompt(row["question"], choices), "audio": audio}],
            "expected_answer": row["answer"],
            "choices": choices,
            "subset_for_metrics": row["task"],
            "dataset_revision": HF_REVISION,
            "official_repo_revision": OFFICIAL_REPO_REVISION,
        }
    )
    return formatted


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare public MMAU test-mini speech/sound rows")
    parser.add_argument("--data-dir", type=str, default=None, help="Durable directory for selected MMAU audio")
    parser.add_argument("--audio-prefix", type=str, default=None, help="Audio prefix written into test.jsonl")
    parser.add_argument("--no-audio", action="store_true", help="Write manifest without materializing audio")
    parser.add_argument("--max-samples", type=int, default=-1, help="Limit rows for a preparation smoke test")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else _default_data_dir()
    audio_prefix = args.audio_prefix or _default_audio_prefix(args.data_dir, data_dir)
    max_samples = args.max_samples if args.max_samples > 0 else None
    rows = select_rows(_load_rows(data_dir), max_samples=max_samples)

    if not args.no_audio:
        _materialize_audio(rows, data_dir)
    local_audio_dir = None if args.no_audio else data_dir / "audio"

    output_path = Path(__file__).resolve().parent / "test.jsonl"
    with open(output_path, "wt", encoding="utf-8") as fout:
        fout.writelines(
            json.dumps(format_entry(row, audio_prefix, local_audio_dir), ensure_ascii=False) + "\n" for row in rows
        )
    print(f"Wrote {len(rows)} MMAU test-mini speech/sound rows to {output_path}")
    print(f"Hugging Face audio revision: {HF_REVISION}")
    print(f"Official metadata/scorer revision: {OFFICIAL_REPO_REVISION}")


if __name__ == "__main__":
    main()
