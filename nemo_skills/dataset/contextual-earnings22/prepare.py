# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Prepare Contextual Earnings-22 dataset for NeMo Skills evaluation.

Downloads the Argmax/Contextual Earnings-22 ``test`` split from HuggingFace
(``argmaxinc/contextual-earnings22``, ~404 MB total), extracts each clip's
audio array to a 16 kHz WAV file under ``<data_dir>/audio/``, and produces
three JSONL split files - one per evaluation mode (contextless, local, global).

If ``--data_dir`` is provided and already contains the prepared audio, the
download/extract step is skipped and the existing data is used directly.

Dataset: https://huggingface.co/datasets/argmaxinc/contextual-earnings22
Paper:   Contextual Earnings-22 (Argmax, 2025) -- https://arxiv.org/abs/2604.07354

Usage:
    # Auto-download to a sibling directory of the repo
    ns prepare_data contextual-earnings22

    # Use pre-downloaded / cached data
    ns prepare_data contextual-earnings22 --data_dir=/path/to/contextual-earnings22

    # Override audio path prefix written into JSONL (e.g. for container mounts)
    ns prepare_data contextual-earnings22 --data_dir=/path --audio-prefix /data/ce22
"""

import argparse
import hashlib
import io
import json
import os
import tempfile
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

HF_REPO_ID = "argmaxinc/contextual-earnings22"
HF_REVISION = "59d7f94771cc197984d8038e494c83abeff87282"
BENCHMARK_NAME = "contextual-earnings22"
SPLIT = "test"
SAMPLING_RATE = 16000
EXPECTED_ROWS = 772
MINIMUM_AUDIO_SAMPLES = 320
PREPARATION_SCHEMA_VERSION = 2
DURATION_POLICY = "decoded_wav_frames_divided_by_16000"
PROVENANCE_FILENAME = "provenance.json"

PROMPT_CONTEXTLESS = "Transcribe the English audio into text, ensuring all punctuation marks are included."
PROMPT_LOCAL = (
    "This audio may contain the following words or phrases: {keyword_list}. "
    "Transcribe the English audio into text, ensuring all punctuation marks are included."
)
PROMPT_GLOBAL = (
    "This audio may contain the following words or phrases (some may not be present in the clip): "
    "{keyword_list}. "
    "Transcribe the English audio into text, ensuring all punctuation marks are included."
)


def _default_data_dir() -> Path:
    """Pick a dataset download location that sits OUTSIDE the repo.

    Honors ``NEMO_SKILLS_DATA_DIR`` when invoked by ``ns prepare_data``.
    Walks up from this file until it finds the repo root (identified by
    ``pyproject.toml``) and returns ``<repo_root>.parent / "<repo_name>-data" /
    <BENCHMARK_NAME>``. Falls back to ``~/.cache/nemo-skills-data/<bench>`` if
    no repo root is found. Never returns a path inside the repo tree.
    """
    if root := os.environ.get("NEMO_SKILLS_DATA_DIR"):
        return Path(root) / BENCHMARK_NAME
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent.parent / f"{parent.name}-data" / BENCHMARK_NAME
    return Path.home() / ".cache" / "nemo-skills-data" / BENCHMARK_NAME


def _audio_subdir(data_dir: Path) -> Path:
    return data_dir / "audio"


def _audio_filename(file_id: str, audio_start: float, idx: int) -> str:
    """Stable, unique audio filename for a clip.

    ``file_id + audio_start`` uniquely identifies a clip in the source dataset
    (multiple clips can share a ``file_id`` since they are 15 s windows from
    the same call). We include ``idx`` only as a deterministic tiebreaker for
    duplicates at the same start position.
    """
    return f"{file_id}__{audio_start:09.2f}__{idx}.wav"


def _decode_audio(value: object) -> tuple[np.ndarray, int]:
    """Decode raw Hugging Face audio with libsndfile, without TorchCodec."""
    if not isinstance(value, dict):
        raise TypeError(f"Contextual Earnings22 raw audio must be a mapping, found {type(value).__name__}")
    encoded = value.get("bytes")
    path = value.get("path")
    if encoded is not None:
        source = io.BytesIO(encoded)
    elif path:
        source = str(path)
    else:
        raise ValueError("Contextual Earnings22 raw audio has neither bytes nor path")

    try:
        samples, sampling_rate = sf.read(source, dtype="float32", always_2d=False)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError("Contextual Earnings22 raw audio could not be decoded by libsndfile") from error
    samples = np.asarray(samples)
    if samples.ndim == 2:
        if samples.shape[1] != 1:
            raise ValueError(f"Contextual Earnings22 source audio must be mono; got shape={samples.shape}")
        samples = samples[:, 0]
    if samples.ndim != 1 or samples.size < MINIMUM_AUDIO_SAMPLES:
        raise ValueError(f"Contextual Earnings22 source audio is empty, short, or malformed: shape={samples.shape}")
    if not np.isfinite(samples).all():
        raise ValueError("Contextual Earnings22 source audio contains non-finite samples")
    sampling_rate = int(sampling_rate)
    if sampling_rate != SAMPLING_RATE:
        raise ValueError(
            f"Contextual Earnings22 source sample-rate drift: expected {SAMPLING_RATE}, found {sampling_rate}"
        )
    return samples.astype(np.float32, copy=False), sampling_rate


def _write_wav(path: Path, samples: np.ndarray, sampling_rate: int) -> None:
    """Atomically write a mono int16 WAV at ``sampling_rate`` Hz."""
    samples = np.asarray(samples)
    if samples.ndim != 1 or samples.size < MINIMUM_AUDIO_SAMPLES:
        raise ValueError(f"Contextual Earnings22 output audio must be nonempty mono; got shape={samples.shape}")
    if not np.isfinite(samples).all():
        raise ValueError("Contextual Earnings22 output audio contains non-finite samples")
    if sampling_rate != SAMPLING_RATE:
        raise ValueError(f"Contextual Earnings22 output sample-rate must be {SAMPLING_RATE}, got {sampling_rate}")
    if np.issubdtype(samples.dtype, np.floating):
        samples = np.clip(samples, -1.0, 1.0)
        samples_i16 = (samples * 32767.0).astype(np.int16)
    else:
        samples_i16 = samples.astype(np.int16)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.unlink(missing_ok=True)
    try:
        with wave.open(str(temporary), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sampling_rate)
            wf.writeframes(samples_i16.tobytes())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    _validate_wav(path)


def _validate_wav(path: Path) -> int:
    if not path.is_file() or path.stat().st_size <= 44:
        raise FileNotFoundError(f"Contextual Earnings22 audio is missing or empty: {path}")
    try:
        with wave.open(str(path), "rb") as stream:
            header = (
                stream.getnchannels(),
                stream.getsampwidth(),
                stream.getframerate(),
                stream.getnframes(),
            )
    except (EOFError, wave.Error) as error:
        raise ValueError(f"Contextual Earnings22 audio has an invalid WAV header: {path}") from error
    if header[:3] != (1, 2, SAMPLING_RATE) or header[3] < MINIMUM_AUDIO_SAMPLES:
        raise ValueError(f"Contextual Earnings22 audio contract failed for {path}: {header}")
    return int(header[3])


def download_and_extract_audio(data_dir: Path) -> int:
    """Download the HF parquet split and write per-clip WAV files.

    Returns the number of clips written. Skips clips that already exist on disk.
    """
    from datasets import Audio, load_dataset

    audio_dir = _audio_subdir(data_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Loading {HF_REPO_ID} (split={SPLIT}) from HuggingFace")
    print(f"Destination: {data_dir}")
    print("Total download size: ~404 MB (parquet with 16 kHz embedded audio)")
    print("=" * 70)

    ds = load_dataset(HF_REPO_ID, split=SPLIT, revision=HF_REVISION)
    if len(ds) != EXPECTED_ROWS:
        raise ValueError(f"Contextual Earnings22 row-count drift: expected {EXPECTED_ROWS}, found {len(ds)}")
    ds = ds.cast_column("audio", Audio(decode=False))

    print(f"Loaded {len(ds)} samples. Extracting audio to {audio_dir} ...")

    samples = []
    written = 0
    for idx, row in enumerate(ds):
        audio = row["audio"]
        file_id = row["file_id"]
        audio_start = float(row["audio_start"])
        wav_name = _audio_filename(file_id, audio_start, idx)
        wav_path = audio_dir / wav_name

        if not wav_path.exists():
            arr, sr = _decode_audio(audio)
            _write_wav(wav_path, arr, sr)
            written += 1
        frames = _validate_wav(wav_path)
        decoded_duration = frames / SAMPLING_RATE
        source_duration = float(row["duration"])
        if not np.isfinite(source_duration) or source_duration <= 0:
            raise ValueError(f"Contextual Earnings22 sample {idx} has invalid source duration metadata")

        samples.append(
            {
                "file_id": file_id,
                "audio_filename": wav_name,
                "audio_start": audio_start,
                "audio_end": float(row["audio_end"]),
                # Inference consumes the materialized WAV, so request duration
                # describes its decoded frame closure.  The pinned source's
                # nominal window duration remains explicit provenance below.
                "duration": decoded_duration,
                "source_duration": source_duration,
                "duration_policy": DURATION_POLICY,
                "preparation_schema_version": PREPARATION_SCHEMA_VERSION,
                "text": row["text"],
                "norm_text": row["norm_text"],
                "transcript_tokens": list(row["transcript"]),
                "local_keywords": list(row["dictionary"]),
                "global_keywords": list(row["keywords"]),
                "is_flagged": bool(row["is_flagged"]),
                "source_row_id": f"{file_id}:{audio_start:.2f}:{idx}",
                "source_repo": HF_REPO_ID,
                "source_revision": HF_REVISION,
                "source_split": SPLIT,
                "expected_rows": EXPECTED_ROWS,
            }
        )

    print(f"Wrote {written} new audio files (skipped {len(samples) - written} already present).")

    metadata_path = data_dir / "samples.jsonl"
    temporary = metadata_path.with_name(f".{metadata_path.name}.partial")
    temporary.unlink(missing_ok=True)
    try:
        with open(temporary, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(s, ensure_ascii=False) + "\n" for s in samples)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, metadata_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Wrote sample metadata: {metadata_path}")

    return len(samples)


def load_samples(data_dir: Path) -> list[dict]:
    """Load the cached sample metadata produced by ``download_and_extract_audio``."""
    metadata_path = data_dir / "samples.jsonl"
    samples: list[dict] = []
    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _validate_samples(samples: list[dict], data_dir: Path, *, require_audio: bool) -> None:
    if len(samples) != EXPECTED_ROWS:
        raise ValueError(
            f"Contextual Earnings22 metadata row-count drift: expected {EXPECTED_ROWS}, found {len(samples)}"
        )

    required = {
        "audio_filename",
        "duration",
        "duration_policy",
        "expected_rows",
        "preparation_schema_version",
        "source_repo",
        "source_revision",
        "source_row_id",
        "source_split",
        "source_duration",
    }
    seen_audio: set[str] = set()
    seen_ids: set[str] = set()
    for index, sample in enumerate(samples):
        missing = required - sample.keys()
        if missing:
            raise ValueError(f"Contextual Earnings22 row {index} is missing fields: {sorted(missing)}")
        if (
            sample["source_repo"] != HF_REPO_ID
            or sample["source_revision"] != HF_REVISION
            or sample["source_split"] != SPLIT
            or sample["expected_rows"] != EXPECTED_ROWS
            or sample["preparation_schema_version"] != PREPARATION_SCHEMA_VERSION
            or sample["duration_policy"] != DURATION_POLICY
        ):
            raise ValueError(f"Contextual Earnings22 row {index} does not match the pinned source contract")
        audio_filename = str(sample["audio_filename"])
        source_row_id = str(sample["source_row_id"])
        if audio_filename in seen_audio or source_row_id in seen_ids:
            raise ValueError(f"Contextual Earnings22 duplicate row/audio identity at row {index}")
        seen_audio.add(audio_filename)
        seen_ids.add(source_row_id)
        duration = float(sample["duration"])
        source_duration = float(sample["source_duration"])
        if not np.isfinite(duration) or duration <= 0 or not np.isfinite(source_duration) or source_duration <= 0:
            raise ValueError(f"Contextual Earnings22 row {index} has invalid duration")

        if not require_audio:
            continue
        audio_path = data_dir / "audio" / audio_filename
        frames = _validate_wav(audio_path)
        if abs(duration * SAMPLING_RATE - frames) > 0.5:
            raise ValueError(f"Contextual Earnings22 row {index} duration does not match decoded WAV frames")


def build_messages(prompt_text: str, audio_path: str, duration: float) -> list[dict]:
    """Build OpenAI-format messages with audio metadata."""
    return [
        {
            "role": "user",
            "content": prompt_text,
            "audio": {
                "path": audio_path,
                "duration": float(duration),
            },
        }
    ]


def format_entry(sample: dict, mode: str, audio_prefix: str) -> dict:
    """Format a single dataset sample into a JSONL record for a given mode."""
    audio_path = f"{audio_prefix}/audio/{sample['audio_filename']}"
    local_keywords = sample["local_keywords"]
    global_keywords = sample["global_keywords"]

    if mode == "contextless":
        prompt = PROMPT_CONTEXTLESS
        eval_keywords = local_keywords
    elif mode == "local":
        prompt = PROMPT_LOCAL.format(keyword_list=", ".join(local_keywords))
        eval_keywords = local_keywords
    elif mode == "global":
        prompt = PROMPT_GLOBAL.format(keyword_list=", ".join(global_keywords))
        eval_keywords = local_keywords
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return {
        "id": f"{BENCHMARK_NAME}-{mode}-{sample['source_row_id']}",
        "messages": build_messages(prompt, audio_path, sample["duration"]),
        "expected_answer": sample["norm_text"],
        "expected_answer_cased": sample["text"],
        "keyword_list": eval_keywords,
        "local_keywords": local_keywords,
        "global_keywords": global_keywords,
        "subset_for_metrics": sample["file_id"],
        "file_id": sample["file_id"],
        "audio_start": sample["audio_start"],
        "duration": float(sample["duration"]),
        "source_duration": float(sample["source_duration"]),
        "duration_policy": sample["duration_policy"],
        "audio_filepath": audio_path,
        "source_provenance": {
            "repo_id": HF_REPO_ID,
            "revision": HF_REVISION,
            "split": SPLIT,
            "expected_rows": EXPECTED_ROWS,
            "preparation_schema_version": PREPARATION_SCHEMA_VERSION,
            "duration_policy": DURATION_POLICY,
        },
    }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_publish_exact(path: Path, payload: bytes) -> None:
    """Publish ``payload`` once and reject any non-identical prior artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"Contextual Earnings22 existing artifact does not match pinned output: {path}")
        return

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".partial",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # Linking within the destination directory creates the final name
            # atomically without ever overwriting a concurrent publisher.
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != payload:
                raise ValueError(
                    f"Contextual Earnings22 concurrent artifact does not match pinned output: {path}"
                ) from None
        if path.read_bytes() != payload:
            raise ValueError(f"Contextual Earnings22 published artifact failed revalidation: {path}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def publish_manifests(samples: list[dict], output_dir: Path, audio_prefix: str) -> dict[str, Path]:
    """Atomically publish and strictly revalidate the three CE22 profiles.

    The deterministic provenance receipt is published last. A killed run can
    safely resume from any byte-identical subset, while stale or mixed-source
    artifacts fail closed instead of being overwritten.
    """
    if len(samples) != EXPECTED_ROWS:
        raise ValueError(
            f"Contextual Earnings22 publication row-count drift: expected {EXPECTED_ROWS}, found {len(samples)}"
        )
    output_dir = Path(output_dir)
    modes = ("contextless", "local", "global")
    outputs: dict[str, Path] = {}
    manifest_receipts: list[dict] = []
    for mode_name in modes:
        output_path = output_dir / mode_name / "test.jsonl"
        payload = b"".join(
            (json.dumps(format_entry(sample, mode_name, audio_prefix), ensure_ascii=False) + "\n").encode("utf-8")
            for sample in samples
        )
        _atomic_publish_exact(output_path, payload)
        outputs[mode_name] = output_path
        manifest_receipts.append(
            {
                "mode": mode_name,
                "path": f"{mode_name}/test.jsonl",
                "rows": len(samples),
                "sha256": _sha256_bytes(payload),
            }
        )

    manifest_set_payload = "".join(f"{item['path']}\0{item['sha256']}\n" for item in manifest_receipts).encode("utf-8")
    receipt = {
        "audio_prefix": audio_prefix,
        "dataset": BENCHMARK_NAME,
        "expected_rows_per_mode": EXPECTED_ROWS,
        "manifest_set_sha256": _sha256_bytes(manifest_set_payload),
        "manifests": manifest_receipts,
        "preparation_schema_version": PREPARATION_SCHEMA_VERSION,
        "schema_version": 1,
        "source_repo": HF_REPO_ID,
        "source_revision": HF_REVISION,
        "source_split": SPLIT,
        "status": "ready",
    }
    receipt_payload = (json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_publish_exact(output_dir / PROVENANCE_FILENAME, receipt_payload)
    return outputs


def main() -> None:
    """Parse arguments, download/extract data if needed, and write per-mode JSONL splits."""
    parser = argparse.ArgumentParser(description="Prepare Contextual Earnings-22 for NeMo Skills")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help=(
            "Path to Contextual-Earnings-22 dataset root. If the directory already "
            "contains audio/ and samples.jsonl, that data is used directly. If the "
            "directory is empty or incomplete, data will be downloaded there "
            "automatically from HuggingFace. If not provided, defaults to a sibling "
            "directory of the repo (never inside the repo tree)."
        ),
    )
    parser.add_argument(
        "--audio-prefix",
        type=str,
        default=None,
        help=(
            "Override audio path prefix written into JSONL files. Defaults to the "
            "data_dir value. Useful for container mount points "
            "(e.g., --audio-prefix /data/contextual-earnings22)."
        ),
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip downloading/verifying audio files (not recommended for actual evaluation).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else _default_data_dir()
    metadata_path = data_dir / "samples.jsonl"

    if metadata_path.exists():
        print(f"Using pre-downloaded data from {data_dir}")
    elif args.no_audio:
        raise FileNotFoundError(
            f"Sample metadata not found at {metadata_path}.\n"
            f"Cannot use --no-audio when data has not been downloaded yet. "
            f"Either run without --no-audio first to download, or point --data_dir "
            f"to a directory that already contains samples.jsonl."
        )
    else:
        print(f"Data not found at {data_dir}. Downloading there...")
        download_and_extract_audio(data_dir)

    audio_prefix = args.audio_prefix if args.audio_prefix else str(data_dir)
    audio_prefix = audio_prefix.rstrip("/")

    print(f"\nReading sample metadata from {metadata_path}")
    samples = load_samples(data_dir)
    print(f"Loaded {len(samples)} samples")
    _validate_samples(samples, data_dir, require_audio=not args.no_audio)
    if not samples:
        raise ValueError(
            f"No samples found in {metadata_path}. Re-run without --no-audio to regenerate dataset metadata."
        )

    if not args.no_audio:
        print(f"Verified {len(samples)} pinned 16 kHz mono WAV files")

    print("\nWriting JSONL splits...")
    outputs = publish_manifests(samples, data_dir, audio_prefix)
    for mode_name, output_path in outputs.items():
        print(f"  {mode_name}: published and revalidated {len(samples)} samples at {output_path}")

    print(f"\nDone. Total: {len(samples)} samples x 3 modes = {len(samples) * 3} records; READY provenance published")


if __name__ == "__main__":
    main()
