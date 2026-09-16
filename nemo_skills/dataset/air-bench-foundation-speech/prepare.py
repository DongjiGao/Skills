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

"""Prepare the public AIR-Bench Foundation speech-category benchmark.

The adapter follows the prompt construction in ``Inference_Foundation.py``
and preserves the exact task and dataset strings used by
``score_foundation.py``. In particular, two misspellings in the public release
(``Recontion`` and ``Reconition``) are protocol identifiers and must not be
corrected.
"""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download

HF_REPO_ID = "qyang1021/AIR-Bench-Dataset"
HF_REVISION = "8fa643aefd1d7248fd55fcf3e7ad22c81bbac1fa"
FOUNDATION_METADATA_PATH = "Foundation/Foundation_meta.json"
FOUNDATION_METADATA_SHA256 = "e59da5c972e30f03797517a009105428060ae34e3b6c112635db8f0bebc312fa"
EXPECTED_FOUNDATION_ROWS = 24683
OFFICIAL_REPO = "OFA-Sys/AIR-Bench"
OFFICIAL_REPO_REVISION = "983247666e2a25519b5c122810c2bcd91f26fd8d"
OFFICIAL_INFERENCE_SHA256 = "607ece9a9a78b9994905824b9e3f74397f0ef575518d160ecfd0222a15247448"
OFFICIAL_SCORER_SHA256 = "1b6380648d0a716a2650cf258b0d76bdf25290cd52493891f163b07a0d18272d"
BENCHMARK_NAME = "air-bench-foundation-speech"
PROVENANCE_FILENAME = "AIR_BENCH_FOUNDATION_SPEECH_PROVENANCE.json"

OFFICIAL_QUESTION_PROMPT = (
    "Choose the most suitable answer from options A, B, C, and D to respond the question in next line, "
    "you may only choose A or B or C or D."
)

# This is the speech category in the AIR-Bench Foundation taxonomy. Vocal
# sound classification belongs to the sound category and is intentionally not
# included. Keep the public spelling of every identifier verbatim.
SPEECH_TASKS = frozenset(
    {
        "Speech_Grounding",
        "Spoken_Language_Identification",
        "Speaker_Gender_Recognition",
        "Speaker_Emotion_Recontion",
        "Speaker_Age_Prediction",
        "Speech_Entity_Reconition",
        "Speaker_Intent_Classification",
        "Speaker_Number_Verification",
        "Synthesized_Voice_Detection",
    }
)
EXPECTED_LEAF_COUNTS = {
    "Speaker_Age_Prediction_common_voice_13.0_en": 1000,
    "Speaker_Emotion_Recontion_iemocap": 1000,
    "Speaker_Emotion_Recontion_meld": 1000,
    "Speaker_Gender_Recognition_common_voice_13_en": 992,
    "Speaker_Gender_Recognition_meld": 1000,
    "Speaker_Intent_Classification_slurp": 1000,
    "Speaker_Number_Verification_voxceleb1": 1000,
    "Speech_Entity_Reconition_slurp": 1000,
    "Speech_Grounding_librispeech": 981,
    "Spoken_Language_Identification_covost2": 1000,
    "Synthesized_Voice_Detection_fake_or_real": 1000,
}
EXPECTED_TOTAL_ROWS = 10973
EXPECTED_AUDIO_INVENTORY_SHA256 = "89571f90615c54d317a178c59691a24dd0223815499ab96c1e0df489d2eef873"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _leaf_id(row: dict[str, Any]) -> str:
    return f"{row['task_name']}_{row['dataset_name']}"


def _audio_repo_path(row: dict[str, Any]) -> str:
    source_path = PurePosixPath(str(row["path"]))
    if source_path.is_absolute() or not source_path.parts or ".." in source_path.parts:
        raise ValueError(f"Unsafe AIR-Bench audio path: {row['path']!r}")
    return str(PurePosixPath("Foundation") / _leaf_id(row) / source_path)


def _inventory_sha256(rows: list[dict[str, Any]]) -> str:
    inventory = "".join(f"{path}\n" for path in sorted(_audio_repo_path(row) for row in rows))
    return hashlib.sha256(inventory.encode("utf-8")).hexdigest()


def _validate_source_row(row: dict[str, Any], index: int) -> None:
    required_strings = (
        "path",
        "question",
        "choice_a",
        "choice_b",
        "answer_gt",
        "task_name",
        "dataset_name",
    )
    for key in required_strings:
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(f"AIR-Bench Foundation row {index} has invalid {key!r}")
    for key in ("choice_c", "choice_d"):
        if row.get(key) is not None and not isinstance(row[key], str):
            raise ValueError(f"AIR-Bench Foundation row {index} has invalid {key!r}")
    if not isinstance(row.get("uniq_id"), int):
        raise TypeError(f"AIR-Bench Foundation row {index} has invalid 'uniq_id'")
    choices = [row.get(f"choice_{letter}") for letter in "abcd"]
    if row["answer_gt"] not in choices:
        raise ValueError(f"AIR-Bench Foundation row {index} answer_gt does not match a choice")
    _audio_repo_path(row)


def validate_pinned_speech_census(rows: list[dict[str, Any]]) -> None:
    """Fail closed if the pinned public speech slice changes."""
    for index, row in enumerate(rows):
        _validate_source_row(row, index)
        if row["task_name"] not in SPEECH_TASKS:
            raise ValueError(f"Non-speech AIR-Bench task in selected rows: {row['task_name']!r}")

    leaf_counts = Counter(_leaf_id(row) for row in rows)
    if leaf_counts != Counter(EXPECTED_LEAF_COUNTS):
        raise ValueError(
            "AIR-Bench Foundation speech leaf-count drift: "
            f"expected={dict(EXPECTED_LEAF_COUNTS)}, observed={dict(sorted(leaf_counts.items()))}"
        )
    unique_ids = {row["uniq_id"] for row in rows}
    if len(rows) != EXPECTED_TOTAL_ROWS or len(unique_ids) != EXPECTED_TOTAL_ROWS:
        raise ValueError(
            f"AIR-Bench Foundation speech row drift: expected {EXPECTED_TOTAL_ROWS}; "
            f"rows={len(rows)}, unique_ids={len(unique_ids)}"
        )
    inventory_sha256 = _inventory_sha256(rows)
    if inventory_sha256 != EXPECTED_AUDIO_INVENTORY_SHA256:
        raise ValueError(
            "AIR-Bench Foundation speech audio-inventory drift: "
            f"expected {EXPECTED_AUDIO_INVENTORY_SHA256}, observed {inventory_sha256}"
        )


def _download_metadata(data_dir: Path) -> Path:
    downloaded = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            revision=HF_REVISION,
            filename=FOUNDATION_METADATA_PATH,
            local_dir=str(data_dir),
        )
    )
    metadata_path = data_dir / FOUNDATION_METADATA_PATH
    if downloaded.resolve() != metadata_path.resolve() and not metadata_path.is_file():
        raise FileNotFoundError(f"Hugging Face download did not materialize {metadata_path}")
    observed_sha256 = _sha256_file(metadata_path)
    if observed_sha256 != FOUNDATION_METADATA_SHA256:
        raise ValueError(
            f"AIR-Bench Foundation metadata hash drift: expected {FOUNDATION_METADATA_SHA256}, "
            f"observed {observed_sha256}"
        )
    return metadata_path


def load_pinned_speech_rows(data_dir: Path) -> list[dict[str, Any]]:
    metadata_path = _download_metadata(data_dir)
    with metadata_path.open("r", encoding="utf-8") as source:
        foundation_rows = json.load(source)
    if not isinstance(foundation_rows, list) or len(foundation_rows) != EXPECTED_FOUNDATION_ROWS:
        observed = len(foundation_rows) if isinstance(foundation_rows, list) else type(foundation_rows).__name__
        raise ValueError(
            f"Expected {EXPECTED_FOUNDATION_ROWS} rows in pinned Foundation metadata, observed {observed}"
        )
    rows = [dict(row) for row in foundation_rows if row.get("task_name") in SPEECH_TASKS]
    validate_pinned_speech_census(rows)
    return rows


def _download_audio(rows: list[dict[str, Any]], data_dir: Path, full_slice: bool) -> None:
    missing_paths = [path for path in (_audio_repo_path(row) for row in rows) if not (data_dir / path).is_file()]
    if not missing_paths:
        return
    if full_slice:
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            revision=HF_REVISION,
            local_dir=str(data_dir),
            allow_patterns=[f"Foundation/{leaf}/**" for leaf in sorted(EXPECTED_LEAF_COUNTS)],
        )
    else:
        for filename in missing_paths:
            hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                revision=HF_REVISION,
                filename=filename,
                local_dir=str(data_dir),
            )


def _validate_audio(rows: list[dict[str, Any]], data_dir: Path) -> None:
    missing = []
    empty = []
    for row in rows:
        relative_path = _audio_repo_path(row)
        path = data_dir / relative_path
        if not path.is_file():
            missing.append(relative_path)
        elif path.stat().st_size <= 0:
            empty.append(relative_path)
    if missing or empty:
        details = []
        if missing:
            details.append(f"missing={len(missing)} ({', '.join(missing[:3])})")
        if empty:
            details.append(f"empty={len(empty)} ({', '.join(empty[:3])})")
        raise FileNotFoundError("Invalid AIR-Bench Foundation speech audio inventory: " + "; ".join(details))


def official_prompt(row: dict[str, Any]) -> str:
    """Construct the exact public ``Inference_Foundation.py`` instruction."""
    choices = (
        f"A. {row['choice_a']}\nB. {row['choice_b']}\nC. {row.get('choice_c', None)}\nD. {row.get('choice_d', None)}"
    )
    return f"{OFFICIAL_QUESTION_PROMPT}\n{row['question']}\n{choices}"


def format_entry(row: dict[str, Any], audio_prefix: str) -> dict[str, Any]:
    relative_path = _audio_repo_path(row)
    audio_path = f"{audio_prefix.rstrip('/')}/{relative_path}"
    return {
        "messages": [{"role": "user", "content": official_prompt(row), "audio": {"path": audio_path}}],
        "expected_answer": row["answer_gt"],
        "audio_path": audio_path,
        "audio_filepath": audio_path,
        "source_audio_path": relative_path,
        "question": row["question"],
        "choice_a": row["choice_a"],
        "choice_b": row["choice_b"],
        "choice_c": row.get("choice_c"),
        "choice_d": row.get("choice_d"),
        "answer_gt": row["answer_gt"],
        "task_name": row["task_name"],
        "dataset_name": row["dataset_name"],
        "uniq_id": row["uniq_id"],
        "subset_for_metrics": _leaf_id(row),
        "air_bench_foundation_leaf": _leaf_id(row),
        "air_bench_dataset_revision": HF_REVISION,
        "air_bench_official_repo_revision": OFFICIAL_REPO_REVISION,
    }


def prepare_air_bench_foundation_speech(
    data_dir: Path,
    *,
    audio_prefix: str = f"/data/{BENCHMARK_NAME}",
    no_audio: bool = False,
    max_samples: int | None = None,
) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    all_rows = load_pinned_speech_rows(data_dir)
    rows = all_rows if max_samples is None else all_rows[:max_samples]
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")

    if not no_audio:
        _download_audio(rows, data_dir, full_slice=max_samples is None)
    _validate_audio(rows, data_dir)

    output_path = data_dir / "test.jsonl"
    temporary_output = output_path.with_name(f".{output_path.name}.tmp")
    provenance_path = data_dir / PROVENANCE_FILENAME
    try:
        with temporary_output.open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(format_entry(row, audio_prefix), ensure_ascii=False) + "\n")
        manifest_sha256 = _sha256_file(temporary_output)
        selected_counts = Counter(_leaf_id(row) for row in rows)
        receipt = {
            "schema_version": 1,
            "dataset": BENCHMARK_NAME,
            "license": "CC-BY-NC-4.0",
            "manifest": output_path.name,
            "manifest_sha256": manifest_sha256,
            "source_repo": HF_REPO_ID,
            "source_revision": HF_REVISION,
            "source_metadata": FOUNDATION_METADATA_PATH,
            "source_metadata_sha256": FOUNDATION_METADATA_SHA256,
            "official_protocol_repo": OFFICIAL_REPO,
            "official_protocol_revision": OFFICIAL_REPO_REVISION,
            "official_inference_sha256": OFFICIAL_INFERENCE_SHA256,
            "official_scorer_sha256": OFFICIAL_SCORER_SHA256,
            "selected_rows": len(rows),
            "selected_leaf_counts": dict(sorted(selected_counts.items())),
            "full_slice_rows": EXPECTED_TOTAL_ROWS,
            "full_slice_leaf_counts": EXPECTED_LEAF_COUNTS,
            "full_slice_audio_inventory_sha256": EXPECTED_AUDIO_INVENTORY_SHA256,
            "audio_mode": "validated-existing" if no_audio else "materialized",
            "scoring": "official per-task_name_dataset_name accuracy; no aggregate accuracy",
        }
        os.replace(temporary_output, output_path)
        _write_json_atomic(provenance_path, receipt)
    finally:
        temporary_output.unlink(missing_ok=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare AIR-Bench Foundation public speech tasks")
    parser.add_argument("--data_dir", default=os.getenv("NEMO_SKILLS_DATA_DIR"))
    parser.add_argument("--audio-prefix", default=f"/data/{BENCHMARK_NAME}")
    parser.add_argument("--no-audio", action="store_true", help="Validate existing pinned audio without downloading")
    parser.add_argument("--max-samples", type=int, default=-1, help="Optional preparation smoke-test limit")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) / BENCHMARK_NAME if args.data_dir else Path(__file__).parent
    max_samples = args.max_samples if args.max_samples > 0 else None
    output_path = prepare_air_bench_foundation_speech(
        data_dir,
        audio_prefix=args.audio_prefix,
        no_audio=args.no_audio,
        max_samples=max_samples,
    )
    print(
        f"AIR-Bench Foundation speech: wrote {sum(1 for _ in output_path.open(encoding='utf-8'))} rows "
        f"to {output_path}; dataset revision={HF_REVISION}; protocol revision={OFFICIAL_REPO_REVISION}"
    )


if __name__ == "__main__":
    main()
