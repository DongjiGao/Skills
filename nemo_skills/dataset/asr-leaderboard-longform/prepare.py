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

"""Prepare the official Open ASR Leaderboard English long-form track.

Protocol provenance is frozen to ``huggingface/open_asr_leaderboard`` commit
``8f37837``.  That revision's NeMo and Transformers long-form runners evaluate
Earnings-21, Earnings-22, TED-LIUM, and all eight CORAAL subsets.  The public
leaderboard's ``Average`` is the macro mean of the first three WERs and the
eight-subset CORAAL macro.

The current head of ``hf-audio/asr-leaderboard-longform`` removed TED-LIUM.
The source is therefore pinned to the final immutable revision that contains
all three official leaves.  This is intentional backward protocol pinning, not
a silent substitution.

This benchmark is not AudioBench Earnings.  AudioBench uses nine randomized
ASR instructions and a different vendored normalization contract.  Here the
prompt is a single deterministic SpeechLM adapter prompt; scoring follows the
official Open ASR English normalizer, compound-aware corpus WER, and long-form
macro hierarchy.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm

from nemo_skills.evaluation.evaluator.open_asr_normalizer import OpenASREnglishTextNormalizer

BENCHMARK_NAME = "asr-leaderboard-longform"
OFFICIAL_PROTOCOL_REPO = "https://github.com/huggingface/open_asr_leaderboard"
OFFICIAL_PROTOCOL_REVISION = "8f37837f988e3d641e726b61087b9c761c060013"
OFFICIAL_NEMO_RUNNER_SHA256 = "dcd7f91d13ef194f997d9e11b444689001ad925527172132b7797cc76df71c79"
OFFICIAL_NEMO_EVAL_SHA256 = "61d9d7d4a88d2eaa6f41c92a2312aea724b976f7474ab844995426153999686e"
OFFICIAL_NORMALIZER_SHA256 = "5cfcaab0e772ac331e9904495ad7c1e07e3eace072f7399d68369303d6927cc4"
OFFICIAL_ABBREVIATIONS_SHA256 = "d3b2c326bdb700b954b2c7125cd0c0b502deb7e22dbb57f707f914646098eaab"
OFFICIAL_EVAL_UTILS_SHA256 = "c8178e859e730ad6d8434ec673b0c196fedf011066447ca89e829f1a556b3a95"

LONGFORM_REPO = "hf-audio/asr-leaderboard-longform"
LONGFORM_REVISION = "4b158b35bb19ed58280bd09da84cdbbcbfa2be8e"
CORAAL_REPO = "bezzam/coraal"
CORAAL_REVISION = "76417d96197aa8bfa419eb5cc7637dab2f75cba0"

AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_FORMAT = "FLAC"
AUDIO_SUBTYPE = "PCM_16"
AUDIO_ROOT = os.environ.get("NEMO_SKILLS_AUDIO_ROOT", "/data").rstrip("/")
SYSTEM_MESSAGE = "You are a helpful assistant. /no_think"
USER_MESSAGE = "Transcribe the following audio."
PROVENANCE_FILENAME = "provenance.json"

CORAAL_CONFIGS = ("ATL", "DCA", "DCB", "DTA", "LES", "PRV", "ROC", "VLD")
PRIMARY_SUBSETS = ("earnings21", "earnings22", "tedlium")
CORAAL_SUBSETS = tuple(f"coraal_{config}" for config in CORAAL_CONFIGS)
OFFICIAL_SUBSETS = PRIMARY_SUBSETS + CORAAL_SUBSETS


@dataclass(frozen=True)
class SourceSpec:
    leaf: str
    repo: str
    revision: str
    config: str
    expected_rows: int
    exact_columns: frozenset[str] | None = None
    required_columns: frozenset[str] = frozenset({"audio", "text"})
    id_field: str | None = None
    license: str = ""
    split: str = "test"


SOURCE_SPECS = (
    SourceSpec(
        "earnings21",
        LONGFORM_REPO,
        LONGFORM_REVISION,
        "earnings21",
        44,
        exact_columns=frozenset(
            {
                "audio",
                "audio_length",
                "original_sample_rate",
                "company_name",
                "financial_quarter",
                "sector",
                "speaker_switches",
                "unique_speakers",
                "curator_id",
                "text",
            }
        ),
        license="CC-BY-SA-4.0",
    ),
    SourceSpec(
        "earnings22",
        LONGFORM_REPO,
        LONGFORM_REVISION,
        "earnings22",
        125,
        exact_columns=frozenset({"audio", "text"}),
        license="CC-BY-SA-4.0",
    ),
    SourceSpec(
        "tedlium",
        LONGFORM_REPO,
        LONGFORM_REVISION,
        "tedlium",
        11,
        exact_columns=frozenset({"audio", "text", "speaker_id"}),
        license="CC-BY-NC-ND-3.0",
    ),
    *(
        SourceSpec(
            f"coraal_{config}",
            CORAAL_REPO,
            CORAAL_REVISION,
            config,
            expected_rows,
            required_columns=frozenset({"audio", "text", "file_id", "CORAAL.Sub"}),
            id_field="file_id",
            license="CC-BY-NC-SA-4.0",
        )
        for config, expected_rows in zip(CORAAL_CONFIGS, (14, 74, 63, 40, 15, 32, 19, 14), strict=True)
    ),
)
EXPECTED_ROWS = {spec.leaf: spec.expected_rows for spec in SOURCE_SPECS}
EXPECTED_TOTAL_ROWS = sum(EXPECTED_ROWS.values())

_NORMALIZER = OpenASREnglishTextNormalizer()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source_dataset(dataset: Any, spec: SourceSpec) -> None:
    """Fail closed on pinned row and schema drift before decoding audio."""
    if len(dataset) != spec.expected_rows:
        raise ValueError(
            f"Pinned row contract failed for {spec.leaf}: expected {spec.expected_rows}, found {len(dataset)}"
        )
    columns = set(dataset.column_names)
    if spec.exact_columns is not None and columns != set(spec.exact_columns):
        raise ValueError(
            f"Pinned column contract failed for {spec.leaf}: "
            f"expected {sorted(spec.exact_columns)}, found {sorted(columns)}"
        )
    missing_columns = set(spec.required_columns) - columns
    if missing_columns:
        raise ValueError(f"Pinned column contract failed for {spec.leaf}: missing {sorted(missing_columns)}")


def load_pinned_source(spec: SourceSpec):
    """Load one immutable source without invoking Hugging Face's TorchCodec decoder."""
    dataset = load_dataset(
        spec.repo,
        spec.config,
        split=spec.split,
        revision=spec.revision,
    )
    _validate_source_dataset(dataset, spec)
    return dataset.cast_column("audio", Audio(decode=False))


def _decode_audio(audio: Any) -> tuple[np.ndarray, int]:
    """Decode one source with libsndfile and enforce the pinned mono/16-kHz contract."""
    if isinstance(audio, dict):
        if audio.get("array") is not None and audio.get("sampling_rate") is not None:
            array = audio["array"]
            sample_rate = audio["sampling_rate"]
        else:
            encoded = audio.get("bytes")
            source_path = audio.get("path")
            if encoded is not None:
                source: Any = io.BytesIO(encoded)
            elif isinstance(source_path, (str, os.PathLike)) and str(source_path):
                source = source_path
            else:
                raise ValueError("Raw Hugging Face audio dict lacks bytes/path")
            try:
                array, sample_rate = sf.read(source, dtype="float32", always_2d=True)
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                raise ValueError("Could not decode pinned source audio with libsndfile") from error
            if array.shape[1] != AUDIO_CHANNELS:
                raise ValueError(f"Pinned source audio has {array.shape[1]} channels, expected {AUDIO_CHANNELS}")
            array = array[:, 0]
    elif hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        array = samples.data
        sample_rate = samples.sample_rate
    else:
        raise TypeError(f"Unsupported decoded Hugging Face audio value: {type(audio).__name__}")

    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    array = np.asarray(array, dtype=np.float32)
    array = np.squeeze(array)
    if array.ndim != 1:
        raise ValueError(f"Official mono Audio contract produced shape {array.shape}, expected one channel")
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Official audio is empty or contains non-finite samples")
    sample_rate = int(sample_rate)
    if sample_rate != AUDIO_SAMPLE_RATE:
        raise ValueError(f"Official resampling contract expected {AUDIO_SAMPLE_RATE} Hz, got {sample_rate} Hz")
    return array, sample_rate


def _safe_identifier(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not normalized:
        raise ValueError(f"Source identifier cannot be converted to a safe path component: {value!r}")
    return normalized


def _source_id(entry: dict[str, Any], spec: SourceSpec, index: int) -> str:
    if spec.id_field is None:
        return f"{spec.leaf}-{index:04d}"
    value = entry.get(spec.id_field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{spec.leaf} row {index} has invalid {spec.id_field}: {value!r}")
    return value.strip()


def _validate_source_row(entry: dict[str, Any], spec: SourceSpec, index: int) -> tuple[str, str]:
    text = entry.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{spec.leaf} row {index} has an empty/non-string reference")
    if not _NORMALIZER(text).strip():
        raise ValueError(f"{spec.leaf} row {index} has an empty reference after official normalization")
    if spec.repo == CORAAL_REPO and entry.get("CORAAL.Sub") != spec.config:
        raise ValueError(
            f"{spec.leaf} row {index} source subset drift: expected {spec.config!r}, got {entry.get('CORAAL.Sub')!r}"
        )
    return _source_id(entry, spec, index), text.strip()


def _validate_materialized_audio(path: Path, expected_frames: int) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing/empty prepared audio: {path}")
    info = sf.info(str(path))
    observed = (info.format, info.subtype, info.samplerate, info.channels, info.frames)
    expected = (AUDIO_FORMAT, AUDIO_SUBTYPE, AUDIO_SAMPLE_RATE, AUDIO_CHANNELS, expected_frames)
    if observed != expected:
        raise ValueError(f"Prepared audio contract drift for {path}: expected {expected}, got {observed}")


def _write_audio_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.partial.flac")
    temporary.unlink(missing_ok=True)
    try:
        sf.write(
            str(temporary),
            array,
            AUDIO_SAMPLE_RATE,
            format=AUDIO_FORMAT,
            subtype=AUDIO_SUBTYPE,
        )
        _validate_materialized_audio(temporary, len(array))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def format_entry(
    entry: dict[str, Any],
    *,
    spec: SourceSpec,
    index: int,
    audio_path: str,
    duration: float,
    source_id: str,
    reference: str,
) -> dict[str, Any]:
    """Format one official long-form test item for SpeechLM generation."""
    return {
        "task_type": "ASR",
        "id": source_id,
        "expected_answer": reference,
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": USER_MESSAGE,
                "audio": {"path": audio_path, "duration": duration},
            },
        ],
        "subset_for_metrics": spec.leaf,
        "audio_path": audio_path,
        "audio_filepath": audio_path,
        "source_repo": spec.repo,
        "source_revision": spec.revision,
        "source_config": spec.config,
        "source_split": spec.split,
        "official_protocol_repo": OFFICIAL_PROTOCOL_REPO,
        "official_protocol_revision": OFFICIAL_PROTOCOL_REVISION,
        "extra_fields": {
            "open_asr_longform_leaf": spec.leaf,
            "source_id": source_id,
            "source_row_index": index,
            "official_normalization": "OpenASREnglishTextNormalizer@8f37837",
            "official_aggregation": "macro(earnings21,earnings22,tedlium,macro(CORAAL[8]))",
        },
    }


def prepare_source(spec: SourceSpec, output_dir: Path, *, no_audio: bool = False) -> list[dict[str, Any]]:
    """Prepare and validate every row from one pinned source config."""
    dataset = load_pinned_source(spec)
    audio_dir = output_dir / "data" / spec.leaf
    seen_ids: set[str] = set()
    records: list[dict[str, Any]] = []

    for index, entry in enumerate(tqdm(dataset, desc=spec.leaf)):
        source_id, reference = _validate_source_row(entry, spec, index)
        if source_id in seen_ids:
            raise ValueError(f"Duplicate source id in {spec.leaf}: {source_id}")
        seen_ids.add(source_id)

        array, sample_rate = _decode_audio(entry.get("audio"))
        if sample_rate != AUDIO_SAMPLE_RATE:  # guarded in _decode_audio; explicit contract here too
            raise ValueError(f"Unexpected sample rate for {spec.leaf}/{source_id}: {sample_rate}")
        audio_filename = f"{index:04d}-{_safe_identifier(source_id)}.flac"
        local_audio_path = audio_dir / audio_filename
        if no_audio:
            _validate_materialized_audio(local_audio_path, len(array))
        else:
            _write_audio_atomic(local_audio_path, array)

        duration = len(array) / AUDIO_SAMPLE_RATE
        served_audio_path = f"{AUDIO_ROOT}/{BENCHMARK_NAME}/data/{spec.leaf}/{audio_filename}"
        records.append(
            format_entry(
                entry,
                spec=spec,
                index=index,
                audio_path=served_audio_path,
                duration=duration,
                source_id=source_id,
                reference=reference,
            )
        )

    if len(records) != spec.expected_rows or len(seen_ids) != spec.expected_rows:
        raise ValueError(
            f"Prepared row contract failed for {spec.leaf}: expected {spec.expected_rows}, "
            f"records={len(records)}, unique_ids={len(seen_ids)}"
        )
    return records


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def prepare_open_asr_longform(output_dir: Path, *, no_audio: bool = False) -> Path:
    """Prepare the complete official 11-leaf long-form composite atomically."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records_by_leaf: dict[str, list[dict[str, Any]]] = {}
    for spec in SOURCE_SPECS:
        records_by_leaf[spec.leaf] = prepare_source(spec, output_dir, no_audio=no_audio)

    observed_counts = {leaf: len(rows) for leaf, rows in records_by_leaf.items()}
    if observed_counts != EXPECTED_ROWS:
        raise ValueError(f"Official long-form census drift: expected {EXPECTED_ROWS}, got {observed_counts}")
    all_rows = [row for leaf in OFFICIAL_SUBSETS for row in records_by_leaf[leaf]]
    if len(all_rows) != EXPECTED_TOTAL_ROWS:
        raise ValueError(f"Official long-form total drift: expected {EXPECTED_TOTAL_ROWS}, got {len(all_rows)}")

    temporary_paths: dict[str, Path] = {}
    manifest_hashes: dict[str, str] = {}
    try:
        for leaf in OFFICIAL_SUBSETS:
            temporary = output_dir / f".{leaf}.jsonl.tmp"
            _write_jsonl(temporary, records_by_leaf[leaf])
            temporary_paths[f"{leaf}.jsonl"] = temporary
            manifest_hashes[f"{leaf}.jsonl"] = sha256_file(temporary)

        combined_temporary = output_dir / ".test.jsonl.tmp"
        _write_jsonl(combined_temporary, all_rows)
        temporary_paths["test.jsonl"] = combined_temporary
        manifest_hashes["test.jsonl"] = sha256_file(combined_temporary)

        receipt = {
            "schema_version": 1,
            "dataset": BENCHMARK_NAME,
            "official_protocol_repo": OFFICIAL_PROTOCOL_REPO,
            "official_protocol_revision": OFFICIAL_PROTOCOL_REVISION,
            "official_artifact_sha256": {
                "nemo_asr/run_longform.sh": OFFICIAL_NEMO_RUNNER_SHA256,
                "nemo_asr/run_eval_long.py": OFFICIAL_NEMO_EVAL_SHA256,
                "normalizer/normalizer.py": OFFICIAL_NORMALIZER_SHA256,
                "normalizer/english_abbreviations.py": OFFICIAL_ABBREVIATIONS_SHA256,
                "normalizer/eval_utils.py": OFFICIAL_EVAL_UTILS_SHA256,
            },
            "sources": [
                {
                    "leaf": spec.leaf,
                    "repo": spec.repo,
                    "revision": spec.revision,
                    "config": spec.config,
                    "split": spec.split,
                    "rows": spec.expected_rows,
                    "license": spec.license,
                }
                for spec in SOURCE_SPECS
            ],
            "row_counts": observed_counts,
            "total_rows": len(all_rows),
            "manifests_sha256": manifest_hashes,
            "audio_mode": "validated-existing" if no_audio else "materialized-and-validated",
            "audio_contract": {
                "official_decode": "datasets.Audio(sampling_rate=16000, mono=True)",
                "served_format": AUDIO_FORMAT,
                "served_subtype": AUDIO_SUBTYPE,
                "sample_rate": AUDIO_SAMPLE_RATE,
                "channels": AUDIO_CHANNELS,
                "one_complete_recording_per_row": True,
            },
            "prompt_contract": {
                "adapter_prompt": USER_MESSAGE,
                "randomized": False,
                "audio_bench_earnings_compatible": False,
            },
            "scoring_contract": {
                "normalizer": "official Open ASR English normalizer at the pinned protocol revision",
                "leaf_metric": "compound-aware corpus WER",
                "coraal_macro": list(CORAAL_SUBSETS),
                "official_average": [*PRIMARY_SUBSETS, "coraal_avg"],
                "avg_without_coraal": list(PRIMARY_SUBSETS),
            },
        }
        receipt_temporary = output_dir / f".{PROVENANCE_FILENAME}.tmp"
        with receipt_temporary.open("w", encoding="utf-8") as output:
            json.dump(receipt, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary_paths[PROVENANCE_FILENAME] = receipt_temporary

        # Leaf manifests are diagnostic; test.jsonl is the evaluation contract,
        # and provenance is published last as the completion marker.
        for filename in [*(f"{leaf}.jsonl" for leaf in OFFICIAL_SUBSETS), "test.jsonl", PROVENANCE_FILENAME]:
            os.replace(temporary_paths.pop(filename), output_dir / filename)
    finally:
        for temporary in temporary_paths.values():
            temporary.unlink(missing_ok=True)

    return output_dir / "test.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the official Open ASR Leaderboard long-form composite")
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Do not rewrite audio; require and validate the complete already-materialized inventory.",
    )
    args = parser.parse_args()

    mounted_data_dir = Path(f"/dataset/{BENCHMARK_NAME}")
    output_dir = mounted_data_dir if mounted_data_dir.exists() else Path(__file__).parent
    output_path = prepare_open_asr_longform(output_dir, no_audio=args.no_audio)
    print(f"Prepared {EXPECTED_TOTAL_ROWS} official long-form rows at {output_path}")


if __name__ == "__main__":
    main()
