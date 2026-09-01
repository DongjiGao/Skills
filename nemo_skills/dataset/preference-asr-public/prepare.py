# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Prepare the pinned 2,822-row public Preference-ASR release.

The paper describes an earlier seven-corpus 3,210-preference/335-standard set.
The public repository intentionally replaced that draft with this licensable
four-corpus release. This adapter evaluates the released artifact and records
both identities; it never labels the 2,822-row release as the paper census.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import tempfile
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

BENCHMARK_NAME = "preference-asr-public"
GITHUB_REPO = "nithinraok/preference-asr-bench"
GITHUB_REVISION = "efa5cae9aecb56ce28aeea6105425ded654acbdd"
MANIFEST_PATH = "data/preference_asr_testset.jsonl"
MANIFEST_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_REVISION}/{MANIFEST_PATH}"
MANIFEST_SHA256 = "134214055ef2b6f4ff88b5526072036be17a59e6a238472fd1f71d9b0c81c2d9"
HF_RELEASE_REPO = "nvidia/preference-asr-bench"
HF_RELEASE_REVISION = "038bd44722b25a17abc9ad53265c6a96be8195ab"
ESB_REPO = "hf-audio/open-asr-leaderboard"
ESB_REVISION = "b6bdcd0beb34f8975dc659796176d88f43aff502"
VOX_REPO = "ArtificialAnalysis/VoxPopuli-Cleaned-AA"
VOX_REVISION = "07adf4a242e2e5dd955a230909c177573c978ee3"
ANNOTATION_LICENSE = "CC-BY-NC-4.0"
CODE_LICENSE = "Apache-2.0"
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SUBTYPE = "PCM_16"
MINIMUM_AUDIO_SAMPLES = 320
DEFAULT_PROMPT = "Transcribe the English audio into text"
PROFILES = ("default", "instructed")
PROVENANCE_FILENAME = "provenance.json"
_SAFE_FILENAME = re.compile(r"[A-Za-z0-9._-]+")


@dataclass(frozen=True)
class ManifestContract:
    rows: int
    unique_audio: int
    row_duration_seconds: float
    empty_preference_text: int
    corpus_rows: tuple[tuple[str, int], ...]
    category_rows: tuple[tuple[str, int], ...]
    direction_rows: tuple[tuple[str | None, int], ...]


MANIFEST_CONTRACT = ManifestContract(
    rows=2822,
    unique_audio=1997,
    row_duration_seconds=21600.068,
    empty_preference_text=5,
    corpus_rows=(("ami", 632), ("common_voice", 477), ("librispeech", 749), ("voxpopuli", 964)),
    category_rows=(("case", 1113), ("disfluencies", 516), ("entities", 906), ("normalization", 228), ("standard", 59)),
    direction_rows=((None, 2078), ("itn", 114), ("keep", 258), ("remove", 258), ("tn", 114)),
)

ESB_SPLITS = {
    "ami": ("test",),
    "common_voice": ("test",),
    "librispeech": ("test.clean", "test.other"),
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_data_root() -> Path:
    if os.environ.get("NEMO_SKILLS_DATA_DIR"):
        return Path(os.environ["NEMO_SKILLS_DATA_DIR"]) / BENCHMARK_NAME
    return Path.home() / ".cache" / "nemo-skills-data" / BENCHMARK_NAME


def _profile_dir(profile: str) -> Path:
    if profile not in PROFILES:
        raise ValueError(f"Unsupported Preference-ASR profile: {profile!r}")
    return Path(__file__).parent / profile


def _audio_prefix() -> str:
    root = os.environ.get("NEMO_SKILLS_AUDIO_ROOT", "/data").rstrip("/")
    return f"{root}/{BENCHMARK_NAME}/audio"


def _source_audio_parts(row: dict) -> tuple[str, str]:
    path = PurePosixPath(str(row.get("audio_filepath", "")))
    if path.is_absolute() or len(path.parts) != 3 or path.parts[0] != "data":
        raise ValueError(f"Unsafe Preference-ASR audio path: {path}")
    corpus, filename = path.parts[1:]
    if corpus != row.get("dataset") or _SAFE_FILENAME.fullmatch(filename) is None:
        raise ValueError(f"Preference-ASR audio path/corpus mismatch: {path}, dataset={row.get('dataset')!r}")
    return corpus, filename


def _target_filename(source_filename: str) -> str:
    return f"{Path(source_filename).stem}.wav"


def _validate_manifest(rows: list[dict], contract: ManifestContract = MANIFEST_CONTRACT) -> list[dict]:
    if len(rows) != contract.rows:
        raise ValueError(f"Preference-ASR row-count drift: expected {contract.rows}, found {len(rows)}")
    required = {
        "audio_filepath",
        "duration",
        "text",
        "preference_text",
        "instruction",
        "preference_type",
        "sub_category",
        "dataset",
    }
    unique_audio: dict[str, float] = {}
    corpus_rows: Counter[str] = Counter()
    category_rows: Counter[str] = Counter()
    direction_rows: Counter[str | None] = Counter()
    empty_preference_text = 0
    total_duration = 0.0
    validated: list[dict] = []
    for row_number, source in enumerate(rows, start=1):
        row = dict(source)
        missing = required.difference(row)
        if missing:
            raise ValueError(f"Preference-ASR row {row_number} is missing fields: {sorted(missing)}")
        corpus, _ = _source_audio_parts(row)
        duration = row["duration"]
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Preference-ASR row {row_number} has invalid duration: {duration!r}")
        for field in ("text", "instruction"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"Preference-ASR row {row_number} has empty/non-string {field}")
        preference_text = row["preference_text"]
        if not isinstance(preference_text, str):
            raise TypeError(f"Preference-ASR row {row_number} has non-string preference_text")
        if not preference_text.strip():
            if row["preference_type"] != "disfluencies" or row.get("direction") != "remove":
                raise ValueError(f"Preference-ASR row {row_number} has unexpected empty preference_text")
            empty_preference_text += 1
        if not isinstance(row["sub_category"], list) or any(
            not isinstance(value, str) or not value for value in row["sub_category"]
        ):
            raise ValueError(f"Preference-ASR row {row_number} has invalid sub_category")
        path = row["audio_filepath"]
        previous_duration = unique_audio.setdefault(path, float(duration))
        if abs(previous_duration - float(duration)) > 1e-9:
            raise ValueError(f"Preference-ASR reused audio has conflicting durations: {path}")
        total_duration += float(duration)
        corpus_rows[corpus] += 1
        category_rows[row["preference_type"]] += 1
        direction_rows[row.get("direction")] += 1
        validated.append(row)
    observed = {
        "unique_audio": len(unique_audio),
        "corpus_rows": tuple(sorted(corpus_rows.items())),
        "category_rows": tuple(sorted(category_rows.items())),
        "direction_rows": tuple(sorted(direction_rows.items(), key=lambda item: str(item[0]))),
        "empty_preference_text": empty_preference_text,
    }
    expected = {
        "unique_audio": contract.unique_audio,
        "corpus_rows": tuple(sorted(contract.corpus_rows)),
        "category_rows": tuple(sorted(contract.category_rows)),
        "direction_rows": tuple(sorted(contract.direction_rows, key=lambda item: str(item[0]))),
        "empty_preference_text": contract.empty_preference_text,
    }
    if observed != expected:
        raise ValueError(f"Preference-ASR released-census drift: expected {expected}, found {observed}")
    if abs(total_duration - contract.row_duration_seconds) > 1e-6:
        raise ValueError(
            f"Preference-ASR duration drift: expected {contract.row_duration_seconds}, found {total_duration}"
        )
    return validated


def load_pinned_manifest(path: Path | None = None) -> tuple[list[dict], bytes]:
    if path is None:
        request = urllib.request.Request(MANIFEST_URL, headers={"User-Agent": "nemo-skills-preference-asr/1"})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
    else:
        payload = path.read_bytes()
    actual_sha = sha256_bytes(payload)
    if actual_sha != MANIFEST_SHA256:
        raise ValueError(f"Preference-ASR manifest SHA drift: expected {MANIFEST_SHA256}, found {actual_sha}")
    try:
        rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Preference-ASR manifest is not valid UTF-8 JSONL") from error
    return _validate_manifest(rows), payload


def _decode_audio(value: object) -> tuple[np.ndarray, int]:
    if isinstance(value, dict):
        encoded, path = value.get("bytes"), value.get("path")
        source = io.BytesIO(encoded) if encoded is not None else path
    else:
        source = value
    if not source:
        raise ValueError("Preference-ASR source audio has neither bytes nor path")
    try:
        samples, sample_rate = sf.read(source, dtype="float32", always_2d=False)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError("Preference-ASR source audio could not be decoded by libsndfile") from error
    samples = np.asarray(samples)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError(f"Preference-ASR source audio is invalid: shape={samples.shape}")
    return samples.astype(np.float32, copy=False), int(sample_rate)


def _canonicalize_audio(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate <= 0:
        raise ValueError(f"Preference-ASR source sample rate is invalid: {sample_rate}")
    if sample_rate != TARGET_SAMPLE_RATE:
        divisor = math.gcd(sample_rate, TARGET_SAMPLE_RATE)
        samples = resample_poly(samples, TARGET_SAMPLE_RATE // divisor, sample_rate // divisor)
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size < MINIMUM_AUDIO_SAMPLES or not np.isfinite(samples).all():
        raise ValueError(f"Preference-ASR canonical audio is invalid: shape={samples.shape}")
    return samples


def _validate_wav(path: Path, expected_duration: float) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Preference-ASR audio is missing: {path}")
    try:
        with sf.SoundFile(str(path)) as stream:
            observed = (stream.samplerate, stream.channels, stream.subtype)
            expected = (TARGET_SAMPLE_RATE, TARGET_CHANNELS, TARGET_SUBTYPE)
            if observed != expected or stream.frames < MINIMUM_AUDIO_SAMPLES:
                raise ValueError(f"Preference-ASR audio contract failed for {path}: {observed}, frames={stream.frames}")
            duration = stream.frames / stream.samplerate
    except RuntimeError as error:
        raise ValueError(f"Preference-ASR audio has an unreadable header: {path}") from error
    if abs(duration - expected_duration) > 0.11:
        raise ValueError(f"Preference-ASR audio duration drift for {path}: metadata={expected_duration}, WAV={duration}")


def _atomic_wav(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.unlink(missing_ok=True)
    try:
        sf.write(temporary, samples, TARGET_SAMPLE_RATE, format="WAV", subtype=TARGET_SUBTYPE)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _needed_audio(rows: list[dict], data_root: Path) -> dict[str, dict[str, tuple[Path, float]]]:
    needed: dict[str, dict[str, tuple[Path, float]]] = defaultdict(dict)
    for row in rows:
        corpus, source_filename = _source_audio_parts(row)
        target = data_root / "audio" / corpus / _target_filename(source_filename)
        needed[corpus][source_filename] = (target, float(row["duration"]))
    return needed


def _id_to_flac_name(sample_id: str) -> str:
    safe = sample_id.replace("/", "_").replace("\\", "_")
    for extension in (".wav", ".flac", ".mp3", ".ogg", ".opus"):
        if safe.lower().endswith(extension):
            safe = safe[: -len(extension)]
            break
    return f"{safe}.flac"


def _materialize_esb(corpus: str, needed: dict[str, tuple[Path, float]], token: str | None) -> None:
    from datasets import Audio, concatenate_datasets, load_dataset

    missing = {name for name, (path, _) in needed.items() if not path.is_file()}
    if missing:
        parts = [
            load_dataset(ESB_REPO, corpus, split=split, revision=ESB_REVISION, token=token)
            for split in ESB_SPLITS[corpus]
        ]
        dataset = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        dataset = dataset.cast_column("audio", Audio(decode=False))
        required = {"id", "audio"}
        if not required.issubset(dataset.column_names):
            raise ValueError(f"Preference-ASR {corpus} source fields missing: {sorted(required-set(dataset.column_names))}")
        for sample in dataset:
            name = _id_to_flac_name(str(sample["id"]))
            if name not in missing:
                continue
            target, _ = needed[name]
            samples, sample_rate = _decode_audio(sample["audio"])
            _atomic_wav(target, _canonicalize_audio(samples, sample_rate))
            missing.remove(name)
            if not missing:
                break
    if missing:
        raise FileNotFoundError(f"Preference-ASR {corpus} source is missing {len(missing)} required clips")
    for target, duration in needed.values():
        _validate_wav(target, duration)


def _materialize_voxpopuli(needed: dict[str, tuple[Path, float]], token: str | None) -> None:
    from huggingface_hub import hf_hub_download

    for name, (target, duration) in sorted(needed.items()):
        if not target.is_file():
            cached = hf_hub_download(
                repo_id=VOX_REPO,
                filename=f"audio/{name}",
                repo_type="dataset",
                revision=VOX_REVISION,
                token=token,
            )
            samples, sample_rate = _decode_audio(cached)
            _atomic_wav(target, _canonicalize_audio(samples, sample_rate))
        _validate_wav(target, duration)


def materialize_audio(rows: list[dict], data_root: Path, token: str | None = None) -> None:
    needed = _needed_audio(rows, data_root)
    if set(needed) != set(ESB_SPLITS) | {"voxpopuli"}:
        raise ValueError(f"Unexpected Preference-ASR source corpora: {sorted(needed)}")
    for corpus in ESB_SPLITS:
        _materialize_esb(corpus, needed[corpus], token)
    _materialize_voxpopuli(needed["voxpopuli"], token)


def _build_record(row: dict, row_number: int, profile: str) -> dict:
    corpus, source_filename = _source_audio_parts(row)
    audio_path = f"{_audio_prefix()}/{corpus}/{_target_filename(source_filename)}"
    prompt = DEFAULT_PROMPT if profile == "default" else row["instruction"].strip()
    target_case = ""
    if row["preference_type"] == "case":
        matches = [
            value
            for value in row["sub_category"]
            if value in {"lower_case", "upper_case", "proper_punctuated_capitalized_format"}
        ]
        if len(matches) != 1:
            raise ValueError(f"Preference-ASR case row {row_number} has ambiguous target: {row['sub_category']!r}")
        target_case = matches[0]
    return {
        "id": f"{BENCHMARK_NAME}-{profile}-{row_number:04d}",
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "audio": {"path": audio_path, "duration": float(row["duration"])},
            }
        ],
        "expected_answer": row["preference_text"].strip(),
        "standard_reference": row["text"].strip(),
        "audio_filepath": audio_path,
        "audio_duration": float(row["duration"]),
        "preference_type": row["preference_type"],
        "sub_category": list(row["sub_category"]),
        "direction": row.get("direction", ""),
        "target_case": target_case,
        "preference_source_dataset": row["dataset"],
        "preference_prompt_variant": profile,
        "subset_for_metrics": row["preference_type"],
        "extra_fields": {
            "source_repo": GITHUB_REPO,
            "source_revision": GITHUB_REVISION,
            "source_manifest_sha256": MANIFEST_SHA256,
            "source_audio_filepath": row["audio_filepath"],
            "annotation_license": ANNOTATION_LICENSE,
            "release_census": MANIFEST_CONTRACT.rows,
            "paper_census_note": "paper describes an earlier seven-corpus 3210 preference + 335 standard draft",
        },
    }


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as output:
            temporary = Path(output.name)
            json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_manifests(rows: list[dict], profiles: tuple[str, ...] = PROFILES) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for profile in profiles:
        manifest_dir = _profile_dir(profile)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        output_path = manifest_dir / "test.jsonl"
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=manifest_dir, prefix=".test.", suffix=".jsonl.tmp", delete=False
            ) as output:
                temporary = Path(output.name)
                for row_number, row in enumerate(rows, start=1):
                    output.write(json.dumps(_build_record(row, row_number, profile), ensure_ascii=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, output_path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        receipt = {
            "schema_version": 1,
            "dataset": BENCHMARK_NAME,
            "profile": profile,
            "manifest": output_path.name,
            "manifest_sha256": sha256_file(output_path),
            "manifest_rows": len(rows),
            "released_manifest_contract": asdict(MANIFEST_CONTRACT),
            "released_source": {
                "github_repo": GITHUB_REPO,
                "github_revision": GITHUB_REVISION,
                "manifest_path": MANIFEST_PATH,
                "manifest_sha256": MANIFEST_SHA256,
                "hf_release_repo": HF_RELEASE_REPO,
                "hf_release_revision": HF_RELEASE_REVISION,
                "annotation_license": ANNOTATION_LICENSE,
                "code_license": CODE_LICENSE,
            },
            "audio_sources": {
                "esb_repo": ESB_REPO,
                "esb_revision": ESB_REVISION,
                "voxpopuli_repo": VOX_REPO,
                "voxpopuli_revision": VOX_REVISION,
                "sample_rate": TARGET_SAMPLE_RATE,
                "channels": TARGET_CHANNELS,
                "subtype": TARGET_SUBTYPE,
                "complete_closure": True,
            },
            "protocol_notes": {
                "condition": profile,
                "default_prompt": DEFAULT_PROMPT,
                "reference": "preference_text",
                "metrics": ["corpus_micro_standard_wer", "corpus_micro_preference_aware_wer"],
                "case_release_contract_repair": "target_case recovered from the sole case sub_category",
                "paper_release_difference": (
                    "The paper describes an earlier seven-corpus 3210-preference/335-standard draft; "
                    "this adapter pins the current four-corpus 2822-row public release."
                ),
            },
        }
        _atomic_json(manifest_dir / PROVENANCE_FILENAME, receipt)
        outputs[profile] = output_path
    return outputs


def prepare(manifest: Path | None, data_root: Path, profiles: tuple[str, ...], token: str | None) -> dict[str, Path]:
    rows, _ = load_pinned_manifest(manifest)
    materialize_audio(rows, data_root, token)
    return write_manifests(rows, profiles)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=None, help="Optional local copy; exact released SHA is required")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--profile", action="append", choices=PROFILES)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()
    profiles = tuple(args.profile or PROFILES)
    outputs = prepare(args.manifest, args.data_root or _default_data_root(), profiles, args.hf_token)
    for profile, output in outputs.items():
        print(f"Wrote pinned public Preference-ASR {profile} manifest to {output}")


if __name__ == "__main__":
    main()
