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

"""Prepare the frozen LibriSpeech test-clean + MUSAN FreeSound 0 dB profile.

Protocol ``ls-test-clean-musan-freesound-0db-v1``:

* clean speech is the complete official LibriSpeech ``test-clean`` archive;
* noise is the 728-file ``noise/free-sound`` inventory exposed by the pinned
  FluidInference MUSAN mirror revision below;
* SHA-256 of ``profile_version|seed|utterance_id`` chooses the sorted noise
  path (first 64 bits) and circular-crop offset (next 64 bits);
* both inputs must be 16 kHz.  Multi-channel inputs are averaged to mono in
  float64.  Noise shorter than the utterance is circularly repeated;
* whole-utterance (not VAD/active-speech) RMS is used.  Noise is scaled to the
  clean RMS, giving exactly 0 dB before summation;
* if the mixture peak exceeds 0.999, one common gain is applied to the mixture.
  This cannot change SNR.  IEEE float32 WAV avoids PCM clipping/quantization.

The HF repository is only a transport mirror.  The audio is the FreeSound
subset distributed in MUSAN/OpenSLR-17.  Pinning its immutable revision and
inventory digest makes this profile reproducible despite the mirror omitting
some files from the complete OpenSLR corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

BENCHMARK_NAME = "librispeech-test-clean-musan-freesound-0db"
PROFILE_VERSION = "ls-test-clean-musan-freesound-0db-v1"
PROFILE_SEED = 20260829
TARGET_SNR_DB = 0.0
SAMPLE_RATE = 16000
PEAK_LIMIT = 0.999
PROMPT = "Produce a verbatim transcript of the audio."

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
LIBRISPEECH_MD5 = "32fa31d27d2e1cad72775fee3f4849a9"
LIBRISPEECH_EXPECTED_UTTERANCES = 2620
MUSAN_SOURCE_URL = "https://www.openslr.org/resources/17/musan.tar.gz"
MUSAN_HF_REPO_ID = "FluidInference/musan"
MUSAN_HF_REVISION = "3edcfdf89b56dbe6a395ff29f9c29489e03d1321"
MUSAN_FREESOUND_GLOB = "noise/free-sound/*.wav"
MUSAN_FREESOUND_EXPECTED_FILES = 728
MUSAN_FREESOUND_INVENTORY_SHA256 = "9b6738e18db769fd747cf172da9c51ac09a07e0bc9ee7cb5eadf7e6c709ff694"


def _default_data_dir() -> Path:
    root = os.environ.get("NEMO_SKILLS_DATA_DIR")
    if root:
        return Path(root) / BENCHMARK_NAME
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent.parent / f"{parent.name}-data" / BENCHMARK_NAME
    return Path.home() / ".cache" / "nemo-skills-data" / BENCHMARK_NAME


def _file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with open(path, "rb") as fin:
        for block in iter(lambda: fin.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    destination_resolved = destination.resolve()
    for member in tar.getmembers():
        target = (destination / member.name).resolve()
        if destination_resolved not in target.parents and target != destination_resolved:
            raise ValueError(f"Unsafe archive member: {member.name}")
    if sys.version_info >= (3, 11, 4):
        tar.extractall(destination, filter="data")
    else:
        tar.extractall(destination)


def _download_librispeech(cache_dir: Path) -> Path:
    archive = cache_dir / "test-clean.tar.gz"
    extracted = cache_dir / "LibriSpeech" / "test-clean"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        print(f"Downloading {LIBRISPEECH_URL}")
        urllib.request.urlretrieve(LIBRISPEECH_URL, archive)
    actual_md5 = _file_digest(archive, "md5")
    if actual_md5 != LIBRISPEECH_MD5:
        raise ValueError(f"LibriSpeech archive MD5 mismatch: expected {LIBRISPEECH_MD5}, got {actual_md5}")
    if not extracted.exists():
        with tarfile.open(archive, "r:gz") as tar:
            _safe_extract(tar, cache_dir)
    return extracted


def inventory_digest(paths: Iterable[str]) -> str:
    """Return the digest of canonical sorted POSIX paths, including final LF."""
    payload = "".join(f"{path}\n" for path in sorted(paths))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_noise_inventory(relative_paths: Iterable[str]) -> list[str]:
    paths = sorted(Path(path).as_posix() for path in relative_paths)
    if len(paths) != MUSAN_FREESOUND_EXPECTED_FILES:
        raise ValueError(
            f"Pinned MUSAN FreeSound inventory has {len(paths)} files; expected {MUSAN_FREESOUND_EXPECTED_FILES}"
        )
    actual_digest = inventory_digest(paths)
    if actual_digest != MUSAN_FREESOUND_INVENTORY_SHA256:
        raise ValueError(
            "Pinned MUSAN FreeSound inventory digest mismatch: "
            f"expected {MUSAN_FREESOUND_INVENTORY_SHA256}, got {actual_digest}"
        )
    return paths


def _download_musan_freesound(cache_dir: Path) -> tuple[Path, list[str]]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ImportError("huggingface_hub is required to prepare this benchmark") from error

    snapshot = Path(
        snapshot_download(
            repo_id=MUSAN_HF_REPO_ID,
            repo_type="dataset",
            revision=MUSAN_HF_REVISION,
            allow_patterns=[MUSAN_FREESOUND_GLOB],
            local_dir=cache_dir / "musan-freesound",
        )
    )
    relative_paths = [path.relative_to(snapshot).as_posix() for path in snapshot.glob(MUSAN_FREESOUND_GLOB)]
    return snapshot, validate_noise_inventory(relative_paths)


def _read_librispeech_rows(test_clean_dir: Path) -> list[tuple[str, str, Path]]:
    transcripts: dict[str, str] = {}
    for transcript_path in sorted(test_clean_dir.glob("*/*/*.trans.txt")):
        with open(transcript_path, encoding="utf-8") as fin:
            for line in fin:
                utterance_id, text = line.rstrip("\n").split(" ", 1)
                if utterance_id in transcripts:
                    raise ValueError(f"Duplicate LibriSpeech utterance ID: {utterance_id}")
                transcripts[utterance_id] = text

    rows = []
    for utterance_id, text in sorted(transcripts.items()):
        speaker, chapter, _ = utterance_id.split("-", 2)
        audio_path = test_clean_dir / speaker / chapter / f"{utterance_id}.flac"
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        rows.append((utterance_id, text, audio_path))
    if len(rows) != LIBRISPEECH_EXPECTED_UTTERANCES:
        raise ValueError(f"LibriSpeech test-clean has {len(rows)} utterances; expected 2620")
    return rows


def selection_digest(utterance_id: str) -> bytes:
    payload = f"{PROFILE_VERSION}|{PROFILE_SEED}|{utterance_id}".encode()
    return hashlib.sha256(payload).digest()


def select_noise(utterance_id: str, noise_paths: list[str], noise_num_samples: int) -> tuple[str, int, str]:
    """Choose noise path and circular offset without any process-global RNG."""
    noise_paths = sorted(noise_paths)
    if not noise_paths:
        raise ValueError("Noise inventory is empty")
    if noise_num_samples <= 0:
        raise ValueError("Noise file is empty")
    digest = selection_digest(utterance_id)
    noise_path = noise_paths[int.from_bytes(digest[:8], "big") % len(noise_paths)]
    crop_offset = int.from_bytes(digest[8:16], "big") % noise_num_samples
    return noise_path, crop_offset, digest.hex()


def _as_mono(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 2:
        samples = samples.mean(axis=1, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError(f"Expected mono or channels-last audio, got shape {samples.shape}")
    return samples


def circular_crop(noise: np.ndarray, num_samples: int, offset: int) -> np.ndarray:
    noise = _as_mono(noise)
    if noise.size == 0 or num_samples <= 0:
        raise ValueError("Audio cannot be empty")
    indices = (np.arange(num_samples, dtype=np.int64) + int(offset)) % noise.size
    return noise[indices]


def rms(samples: np.ndarray) -> float:
    samples = np.asarray(samples, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))


def mix_at_zero_db(clean: np.ndarray, noise: np.ndarray, crop_offset: int) -> tuple[np.ndarray, dict[str, float]]:
    clean = _as_mono(clean)
    if clean.size == 0:
        raise ValueError("Clean utterance is empty")
    noise_crop = circular_crop(noise, clean.size, crop_offset)
    clean_rms = rms(clean)
    noise_rms = rms(noise_crop)
    if clean_rms <= np.finfo(np.float64).eps or noise_rms <= np.finfo(np.float64).eps:
        raise ValueError("Cannot construct 0 dB mixture from silent clean/noise input")
    noise_scale = clean_rms / noise_rms
    scaled_noise = noise_crop * noise_scale
    measured_snr_db = 20.0 * np.log10(clean_rms / rms(scaled_noise))
    mixture = clean + scaled_noise
    peak = float(np.max(np.abs(mixture)))
    peak_gain = min(1.0, PEAK_LIMIT / peak) if peak > 0.0 else 1.0
    mixture = mixture * peak_gain
    return mixture.astype(np.float32), {
        "speech_rms": clean_rms,
        "noise_crop_rms": noise_rms,
        "noise_scale": float(noise_scale),
        "pre_peak_abs_max": peak,
        "peak_gain": float(peak_gain),
        "measured_snr_db": float(measured_snr_db),
    }


def create_manifest_entry(
    utterance_id: str,
    transcript: str,
    duration: float,
    audio_path: str,
    noise_path: str,
    crop_offset: int,
    digest_hex: str,
    mix_metadata: dict[str, float],
) -> dict:
    audio = {"path": audio_path, "duration": float(duration)}
    return {
        "audio_path": [audio_path],
        "messages": [{"role": "user", "content": PROMPT, "audio": audio, "audios": [audio]}],
        "expected_answer": transcript,
        "dataset": BENCHMARK_NAME,
        "subset_for_metrics": "test-clean_musan-freesound_0db",
        "sample_id": utterance_id,
        "task_type": "ASR",
        "audio_duration": float(duration),
        "question": PROMPT,
        "profile_version": PROFILE_VERSION,
        "profile_seed": PROFILE_SEED,
        "target_snr_db": TARGET_SNR_DB,
        "clean_source_url": LIBRISPEECH_URL,
        "clean_source_md5": LIBRISPEECH_MD5,
        "noise_corpus_source_url": MUSAN_SOURCE_URL,
        "noise_transport_repo": MUSAN_HF_REPO_ID,
        "noise_transport_revision": MUSAN_HF_REVISION,
        "noise_inventory_sha256": MUSAN_FREESOUND_INVENTORY_SHA256,
        "noise_path": noise_path,
        "noise_crop_offset_samples": int(crop_offset),
        "selection_sha256": digest_hex,
        **mix_metadata,
    }


def prepare(data_dir: Path, max_samples: int | None = None) -> Path:
    cache_dir = data_dir / "sources"
    test_clean_dir = _download_librispeech(cache_dir)
    noise_root, noise_paths = _download_musan_freesound(cache_dir)
    clean_rows = _read_librispeech_rows(test_clean_dir)
    if max_samples is not None and max_samples > 0:
        clean_rows = clean_rows[:max_samples]

    audio_dir = data_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_prefix = f"/data/{BENCHMARK_NAME}/audio"
    manifest_path = data_dir / "test.jsonl"
    entries = []
    for utterance_id, transcript, clean_path in tqdm(clean_rows, desc="Mixing 0 dB profile"):
        clean, clean_sr = sf.read(clean_path, dtype="float64", always_2d=False)
        if clean_sr != SAMPLE_RATE:
            raise ValueError(f"{clean_path}: expected {SAMPLE_RATE} Hz, got {clean_sr}")

        digest = selection_digest(utterance_id)
        selected_rel = noise_paths[int.from_bytes(digest[:8], "big") % len(noise_paths)]
        noise, noise_sr = sf.read(noise_root / selected_rel, dtype="float64", always_2d=False)
        if noise_sr != SAMPLE_RATE:
            raise ValueError(f"{selected_rel}: expected {SAMPLE_RATE} Hz, got {noise_sr}")
        noise = _as_mono(noise)
        _, crop_offset, digest_hex = select_noise(utterance_id, noise_paths, noise.size)
        mixture, metadata = mix_at_zero_db(clean, noise, crop_offset)

        output_path = audio_dir / f"{utterance_id}.wav"
        sf.write(output_path, mixture, SAMPLE_RATE, subtype="FLOAT")
        entries.append(
            create_manifest_entry(
                utterance_id=utterance_id,
                transcript=transcript,
                duration=mixture.size / SAMPLE_RATE,
                audio_path=f"{audio_prefix}/{output_path.name}",
                noise_path=selected_rel,
                crop_offset=crop_offset,
                digest_hex=digest_hex,
                mix_metadata=metadata,
            )
        )

    with open(manifest_path, "w", encoding="utf-8") as fout:
        fout.writelines(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries)
    print(f"Wrote {len(entries)} examples to {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Developer smoke-test subset; omit for profile")
    args = parser.parse_args()
    prepare(args.data_dir or _default_data_dir(), args.max_samples)


if __name__ == "__main__":
    main()
