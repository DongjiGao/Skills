# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Prepare the DavidAI RT Sound Expert evaluation manifests for NeMo-Skills."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Iterable

BENCHMARK_NAME = "rich_transcription_asr"
DEFAULT_MANIFEST_ROOT = Path(
    "/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/"
    "data/ASR/davidAI/manifests/RT_dev_eval"
)
BENCHMARK_MANIFESTS = {
    "val.d12a": ("d12a_val.json", "validation"),
    "val.d8": ("d8_val_noflat.json", "validation"),
    "test.d12a": ("d12a_test.json", "test"),
    "test.d8": ("d8_test_noflat.json", "test"),
}
DEFAULT_PROMPT = "Transcribe the following audio and include audible sound-event and speaking-style tags inline."
SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")
MIN_AUDIO_DURATION_SECONDS = 0.1


class InvalidAudioSegmentError(ValueError):
    """Raised when a manifest row cannot produce a non-empty audio segment."""


def _default_data_dir() -> Path:
    data_root = os.getenv("NEMO_SKILLS_DATA_DIR")
    if data_root:
        return Path(data_root).expanduser().resolve() / BENCHMARK_NAME

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent.parent / f"{parent.name}-data" / BENCHMARK_NAME
    return Path.home() / ".cache" / "nemo-skills-data" / BENCHMARK_NAME


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("rt", encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc


def _resolve_sets(values: list[str] | None) -> list[str]:
    if not values or values == ["all"]:
        return list(BENCHMARK_MANIFESTS)

    selected = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if part == "all":
                selected.extend(BENCHMARK_MANIFESTS)
            elif part in ("val", "test"):
                selected.extend(
                    name for name in BENCHMARK_MANIFESTS if name.startswith(f"{part}.")
                )
            elif part not in BENCHMARK_MANIFESTS:
                raise ValueError(
                    f"Unsupported RT set {part!r}; expected one of {sorted(BENCHMARK_MANIFESTS)}"
                )
            else:
                selected.append(part)
    return list(dict.fromkeys(selected))


def _safe_id(value: str) -> str:
    return SAFE_ID_RE.sub("_", value).strip("_") or "sample"


def _sample_id(benchmark: str, index: int, audio_path: Path) -> str:
    return _safe_id(f"{benchmark}-{index:05d}-{audio_path.stem}")


def _materialize_audio_segment(
    source_path: Path,
    output_path: Path,
    offset: float,
    duration: float,
    *,
    overwrite: bool = False,
) -> float:
    """Write one manifest segment as a standalone WAV and return its duration.

    Very short reference segments are right-padded with silence. The Sound
    Expert encoder cannot transcribe a standalone 10 ms clip, while 100 ms is
    the model backend's established warm-up/input floor.
    """
    import numpy as np
    import soundfile as sf

    if not source_path.is_file():
        raise FileNotFoundError(f"Source audio does not exist: {source_path}")

    if output_path.is_file() and not overwrite:
        info = sf.info(str(output_path))
        minimum_frames = math.ceil(MIN_AUDIO_DURATION_SECONDS * info.samplerate)
        if info.frames >= minimum_frames and info.samplerate > 0:
            return float(info.frames / info.samplerate)

    info = sf.info(str(source_path))
    start_frame = max(0, int(round(offset * info.samplerate)))
    requested_frames = max(0, int(round(duration * info.samplerate)))
    available_frames = max(0, int(info.frames) - start_frame)
    if requested_frames <= 0 or available_frames <= 0:
        raise InvalidAudioSegmentError(
            f"Empty segment: source={source_path} offset={offset} duration={duration}"
        )
    if requested_frames > available_frames:
        available_duration = available_frames / info.samplerate
        raise InvalidAudioSegmentError(
            "Segment extends beyond source audio: "
            f"source={source_path} offset={offset} requested_duration={duration} "
            f"available_duration={available_duration}"
        )

    with sf.SoundFile(str(source_path), mode="r") as fin:
        fin.seek(start_frame)
        audio = fin.read(frames=requested_frames, dtype="float32", always_2d=True)
    if len(audio) != requested_frames:
        raise InvalidAudioSegmentError(
            f"Decoded {len(audio)} of {requested_frames} requested samples: "
            f"source={source_path} offset={offset} duration={duration}"
        )

    minimum_frames = math.ceil(MIN_AUDIO_DURATION_SECONDS * info.samplerate)
    if len(audio) < minimum_frames:
        audio = np.pad(audio, ((0, minimum_frames - len(audio)), (0, 0)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp.wav")
    sf.write(
        str(temporary_path), audio, info.samplerate, format="WAV", subtype="PCM_16"
    )
    os.replace(temporary_path, output_path)
    return float(len(audio) / info.samplerate)


def _build_record(
    row: dict,
    *,
    benchmark: str,
    index: int,
    data_dir: Path,
    audio_prefix: Path,
    no_audio: bool,
    overwrite: bool,
) -> dict:
    required = ("audio_filepath", "duration", "offset", "text")
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(
            f"{benchmark} row {index} is missing required fields: {missing}"
        )

    source_audio = Path(str(row["audio_filepath"])).expanduser()
    if not source_audio.is_absolute():
        raise ValueError(
            f"{benchmark} row {index} has a non-absolute audio path: {source_audio}"
        )

    reference = str(row["text"] or "").strip()
    if not reference:
        raise ValueError(f"{benchmark} row {index} has an empty reference transcript")

    offset = float(row["offset"] or 0.0)
    duration = float(row["duration"])
    if offset < 0 or duration <= 0:
        raise ValueError(
            f"{benchmark} row {index} has invalid offset/duration: {offset}/{duration}"
        )

    sample_id = _sample_id(benchmark, index, source_audio)
    benchmark_path = Path(*benchmark.split("."))
    local_audio_path = data_dir / "audio" / benchmark_path / f"{sample_id}.wav"
    if no_audio:
        materialized_duration = duration
    else:
        materialized_duration = _materialize_audio_segment(
            source_audio,
            local_audio_path,
            offset,
            duration,
            overwrite=overwrite,
        )

    public_audio_path = audio_prefix / benchmark_path / local_audio_path.name
    duration_seconds = round(materialized_duration, 6)
    record = {
        "id": sample_id,
        "sample_id": sample_id,
        "dataset_id": f"{BENCHMARK_NAME}:{benchmark}",
        "subset_for_metrics": benchmark,
        "task_type": "Sound-ASR",
        "category": "nonjudge",
        "audio_filepath": str(public_audio_path),
        "source_audio_filepath": str(source_audio),
        "offset": offset,
        "duration": duration_seconds,
        "text": reference,
        "expected_answer": reference,
        "messages": [
            {
                "role": "user",
                "content": DEFAULT_PROMPT,
                "audio": {"path": str(public_audio_path), "duration": duration_seconds},
            }
        ],
    }
    scenario = str(row.get("scenario") or "")
    if scenario:
        record["extra_fields"] = {"scenario": scenario}
    return record


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    count = 0
    with temporary_path.open("wt", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    os.replace(temporary_path, path)
    return count


def prepare_manifest_set(
    benchmark: str,
    *,
    manifest_root: Path,
    output_dir: Path,
    data_dir: Path,
    audio_prefix: Path,
    no_audio: bool,
    overwrite: bool,
    max_samples: int | None,
) -> int:
    manifest_filename, output_split = BENCHMARK_MANIFESTS[benchmark]
    manifest_path = manifest_root / manifest_filename
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing exact RT manifest for {benchmark}: {manifest_path}"
        )

    rows = []
    for index, row in enumerate(_iter_jsonl(manifest_path)):
        if max_samples is not None and len(rows) >= max_samples:
            break
        rows.append(
            _build_record(
                row,
                benchmark=benchmark,
                index=index,
                data_dir=data_dir,
                audio_prefix=audio_prefix,
                no_audio=no_audio,
                overwrite=overwrite,
            )
        )
        if (index + 1) % 250 == 0:
            print(f"{benchmark}: prepared {index + 1} rows", flush=True)

    return _write_jsonl(
        output_dir / Path(*benchmark.split(".")) / f"{output_split}.jsonl", rows
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument(
        "--data_dir", "--dataset-root", dest="data_dir", type=Path, default=None
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument(
        "--audio-prefix",
        type=Path,
        default=None,
        help="Audio prefix embedded in JSONL. Defaults to <data_dir>/audio.",
    )
    parser.add_argument("--sets", nargs="+", default=["all"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Prepare manifests without writing clipped WAVs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already-materialized WAV segments.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    # Keep the user-selected Lustre spelling in diagnostics and provenance. On the
    # development hosts this mount may resolve through /pool, but it is still the
    # exact shared manifest root selected above.
    manifest_root = args.manifest_root.expanduser()
    data_dir = (
        args.data_dir.expanduser().resolve() if args.data_dir else _default_data_dir()
    )
    output_dir = args.output_dir.expanduser().resolve()
    selected_sets = _resolve_sets(args.sets)

    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exact RT manifest root: {manifest_root}")
    print(f"Data/artifact root: {data_dir}")
    for benchmark in selected_sets:
        count = prepare_manifest_set(
            benchmark,
            manifest_root=manifest_root,
            output_dir=output_dir,
            data_dir=data_dir,
            audio_prefix=(
                args.audio_prefix.expanduser()
                if args.audio_prefix
                else data_dir / "audio"
            ),
            no_audio=args.no_audio,
            overwrite=args.overwrite,
            max_samples=args.max_samples,
        )
        print(f"Prepared {benchmark}: {count}")


if __name__ == "__main__":
    main()
