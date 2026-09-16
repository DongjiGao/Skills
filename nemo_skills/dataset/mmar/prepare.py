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

"""Prepare the official MMAR test rows whose modality is exactly ``speech``.

Official sources:
  Dataset: https://huggingface.co/datasets/BoJack/MMAR
  Scorer:  https://github.com/ddlBoJack/MMAR/blob/main/code/evaluation.py

The Hugging Face revision is pinned because MMAR publishes all test labels and
may update the single public test set in place. Only the 294 pure-speech audio
members are extracted from the full archive.
"""

import argparse
import json
import os
import shutil
import tarfile
from collections.abc import Iterable
from pathlib import Path

HF_REPO_ID = "BoJack/MMAR"
HF_REVISION = "3bd051123480e80d273ae9e8e9f1653f49010ac7"
OFFICIAL_SCORER_REVISION = "3bce090a967db576c8ad433b290a8da582d6d2a0"
METADATA_FILENAME = "MMAR-meta.json"
AUDIO_ARCHIVE_FILENAME = "mmar-audio.tar.gz"
BENCHMARK_NAME = "mmar"
FILTER_MODALITY = "speech"
EXPECTED_TEST_ROWS = 294

ANSWER_INSTRUCTION = "Respond with only the text of the correct option."


def _default_data_dir() -> Path:
    if os.environ.get("NEMO_SKILLS_DATA_DIR"):
        return Path(os.environ["NEMO_SKILLS_DATA_DIR"]) / BENCHMARK_NAME
    return Path(__file__).resolve().parent


def _default_audio_prefix(args_data_dir: str | None, data_dir: Path) -> str:
    if args_data_dir is None and os.environ.get("NEMO_SKILLS_DATA_DIR"):
        return f"/data/{BENCHMARK_NAME}"
    return str(data_dir)


def _load_rows() -> list[dict]:
    from huggingface_hub import hf_hub_download

    metadata_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=METADATA_FILENAME,
        repo_type="dataset",
        revision=HF_REVISION,
    )
    with open(metadata_path, "rt", encoding="utf-8") as fin:
        rows = json.load(fin)
    if not isinstance(rows, list):
        raise TypeError(f"Expected a JSON list in {METADATA_FILENAME}, got {type(rows).__name__}")
    return rows


def select_rows(rows: Iterable[dict], max_samples: int | None = None) -> list[dict]:
    """Select the exact PRD profile: official test rows with modality == speech."""
    selected = [dict(row) for row in rows if row.get("modality") == FILTER_MODALITY]
    if max_samples is None and len(selected) != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"Pinned MMAR revision yielded {len(selected)} pure-speech rows; expected {EXPECTED_TEST_ROWS}."
        )
    if max_samples is not None:
        selected = selected[:max_samples]
    return selected


def _audio_filename(row: dict) -> str:
    name = Path(str(row["audio_path"])).name
    if not name or name in {".", ".."}:
        raise ValueError(f"Invalid MMAR audio_path: {row['audio_path']!r}")
    return name


def _extract_selected_audio(rows: Iterable[dict], data_dir: Path) -> None:
    """Safely materialize only selected files; never extract archive paths directly."""
    from huggingface_hub import hf_hub_download

    audio_dir = data_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    wanted = {_audio_filename(row) for row in rows}
    missing = {name for name in wanted if not (audio_dir / name).is_file()}
    if not missing:
        return

    archive_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=AUDIO_ARCHIVE_FILENAME,
        repo_type="dataset",
        revision=HF_REVISION,
    )
    found: set[str] = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            basename = Path(member.name).name
            if basename not in missing or basename in found:
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            destination = audio_dir / basename
            with source, open(destination, "wb") as fout:
                shutil.copyfileobj(source, fout)
            found.add(basename)
            if found == missing:
                break

    unresolved = missing - found
    if unresolved:
        preview = ", ".join(sorted(unresolved)[:10])
        raise FileNotFoundError(f"MMAR archive is missing {len(unresolved)} selected audio files: {preview}")


def _parse_hms(value: str) -> float:
    hours, minutes, seconds = value.strip().split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _duration_seconds(timestamp: object) -> float | None:
    try:
        start, end = str(timestamp).split(",", maxsplit=1)
        duration = _parse_hms(end) - _parse_hms(start)
    except (TypeError, ValueError):
        return None
    return duration if duration > 0 else None


def _prompt(question: str, choices: list[str]) -> str:
    options = "\n".join(f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(choices))
    return (
        "Listen to the audio and answer the multiple-choice question.\n\n"
        f"Question: {question}\n\nOptions:\n{options}\n\n{ANSWER_INSTRUCTION}"
    )


def format_entry(row: dict, audio_prefix: str) -> dict:
    """Preserve official question/choices/answer and add a neutral audio prompt."""
    choices = list(row["choices"])
    audio_path = f"{audio_prefix.rstrip('/')}/audio/{_audio_filename(row)}"
    audio = {"path": audio_path}
    duration = _duration_seconds(row.get("timestamp"))
    if duration is not None:
        audio["duration"] = duration

    formatted = dict(row)
    formatted.update(
        {
            "source_audio_path": row["audio_path"],
            "audio_path": audio_path,
            "audio_filepath": audio_path,
            "messages": [{"role": "user", "content": _prompt(row["question"], choices), "audio": audio}],
            "expected_answer": row["answer"],
            "choices": choices,
            "subset_for_metrics": row["category"],
            "dataset_revision": HF_REVISION,
            "official_scorer_revision": OFFICIAL_SCORER_REVISION,
        }
    )
    return formatted


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the public MMAR pure-speech test profile")
    parser.add_argument("--data-dir", type=str, default=None, help="Durable directory for selected MMAR audio")
    parser.add_argument("--audio-prefix", type=str, default=None, help="Audio prefix written into test.jsonl")
    parser.add_argument("--no-audio", action="store_true", help="Write manifest without downloading audio")
    parser.add_argument("--max-samples", type=int, default=-1, help="Limit rows for a preparation smoke test")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else _default_data_dir()
    audio_prefix = args.audio_prefix or _default_audio_prefix(args.data_dir, data_dir)
    max_samples = args.max_samples if args.max_samples > 0 else None
    rows = select_rows(_load_rows(), max_samples=max_samples)

    if not args.no_audio:
        _extract_selected_audio(rows, data_dir)

    output_path = Path(__file__).resolve().parent / "test.jsonl"
    with open(output_path, "wt", encoding="utf-8") as fout:
        fout.writelines(json.dumps(format_entry(row, audio_prefix), ensure_ascii=False) + "\n" for row in rows)
    print(f"Wrote {len(rows)} MMAR pure-speech rows to {output_path}")
    print(f"Dataset revision: {HF_REVISION}")
    print(f"Official scorer revision: {OFFICIAL_SCORER_REVISION}")


if __name__ == "__main__":
    main()
