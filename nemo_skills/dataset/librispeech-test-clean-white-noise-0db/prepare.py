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

"""Prepare LibriSpeech test-clean with white noise at 0 dB SNR.

Protocol ``whisper-fig5-ls-test-clean-adt-white-0db-v1`` follows Section 3.7
and Figure 5 of *Robust Speech Recognition via Large-Scale Weak Supervision*:

* the complete official LibriSpeech ``test-clean`` split (2,620 utterances);
* independent uniform white noise, matching Audio Degradation Toolbox (ADT)
  ``rand(size(audio)) - 0.5``;
* noise scaled from each utterance's total signal/noise power to 0 dB SNR;
* the summed waveform peak-normalized to 0.999, matching ADT's default;
* WER after the standard Whisper text normalizer.

The paper and ADT do not publish the random seed used for Figure 5.  This
profile therefore freezes a deterministic per-utterance PCG64 realization.
That removes ordering/process-count dependence and records the seed derivation
in every row.  It matches the published perturbation distribution and exact
power/normalization equations, but does not claim bit identity with OpenAI's
unreleased random draw.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

BENCHMARK_NAME = "librispeech-test-clean-white-noise-0db"
PROFILE_VERSION = "whisper-fig5-ls-test-clean-adt-white-0db-v1"
PROFILE_SEED = 20260829
TARGET_SNR_DB = 0.0
SAMPLE_RATE = 16000
PEAK_AMPLITUDE = 0.999
PROMPT = "Transcribe the following audio."

WHISPER_PAPER_URL = "https://cdn.openai.com/papers/whisper.pdf"
WHISPER_PROTOCOL_SECTION = "3.7 / Figure 5"
ADT_REPO_URL = "https://github.com/hagenw/audio-degradation-toolbox"
ADT_REVISION = "6f06bcc4ad9c4234f4a9ec655814889ba712c8e7"
ADT_SOURCE_SHA256 = {
    "degradationUnit_addNoise.m": "541185314431adbb2faffe9c92296d0d8712024fcaa81b96ea273dc923148937",
    "degradationUnit_addSound.m": "29c0f3be473fd73dbf6b30630dbb8e114ee50e5b1fa25359eb30746de96a7cea",
    "adthelper_normalizeAudio.m": "77931d8384206f6dfbc573dd92af7d7461e36632e2b373597d83410f116459d7",
}
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
LIBRISPEECH_MD5 = "32fa31d27d2e1cad72775fee3f4849a9"
LIBRISPEECH_EXPECTED_UTTERANCES = 2620


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
    root = destination.resolve()
    for member in tar.getmembers():
        target = (destination / member.name).resolve()
        if root not in target.parents and target != root:
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
        raise ValueError(
            f"LibriSpeech test-clean has {len(rows)} utterances; expected {LIBRISPEECH_EXPECTED_UTTERANCES}"
        )
    return rows


def realization_digest(utterance_id: str) -> str:
    payload = f"{PROFILE_VERSION}|{PROFILE_SEED}|{utterance_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def generate_white_noise(utterance_id: str, num_samples: int) -> tuple[np.ndarray, str, int]:
    """Return frozen ADT-distributed ``rand()-0.5`` noise for one utterance."""
    if num_samples <= 0:
        raise ValueError("Audio cannot be empty")
    digest_hex = realization_digest(utterance_id)
    seed = int.from_bytes(bytes.fromhex(digest_hex)[:16], "big")
    rng = np.random.Generator(np.random.PCG64(seed))
    noise = rng.random(num_samples, dtype=np.float64) - 0.5
    return noise, digest_hex, seed


def _as_mono(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 2:
        samples = samples.mean(axis=1, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError(f"Expected mono or channels-last audio, got shape {samples.shape}")
    return samples


def mix_adt_white_noise(clean: np.ndarray, noise: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Literal NumPy port of ADT white-noise scaling and normalization."""
    clean = _as_mono(clean)
    noise = _as_mono(noise)
    if clean.size == 0 or noise.size != clean.size:
        raise ValueError("Clean and noise must have the same positive length")
    speech_power = float(np.sum(np.square(clean), dtype=np.float64))
    noise_power = float(np.sum(np.square(noise), dtype=np.float64))
    if speech_power <= np.finfo(np.float64).eps or noise_power <= np.finfo(np.float64).eps:
        raise ValueError("Cannot construct target-SNR mixture from silent input")

    noise_scale = float(np.sqrt(speech_power / (noise_power * 10.0 ** (TARGET_SNR_DB / 10.0))))
    scaled_noise = noise * noise_scale
    mixture = clean + scaled_noise
    pre_peak = float(np.max(np.abs(mixture)))
    normalization_gain = PEAK_AMPLITUDE / max(pre_peak, np.finfo(np.float64).eps)
    mixture *= normalization_gain
    measured_snr_db = 10.0 * np.log10(speech_power / float(np.sum(np.square(scaled_noise))))
    return mixture.astype(np.float32), {
        "speech_power": speech_power,
        "raw_noise_power": noise_power,
        "noise_scale": noise_scale,
        "pre_normalization_peak": pre_peak,
        "normalization_gain": float(normalization_gain),
        "measured_snr_db": float(measured_snr_db),
    }


def create_manifest_entry(
    *,
    utterance_id: str,
    transcript: str,
    duration: float,
    audio_path: str,
    realization_sha256: str,
    realization_seed: int,
    mix_metadata: dict[str, float],
) -> dict:
    audio = {"path": audio_path, "duration": float(duration)}
    return {
        "audio_path": [audio_path],
        "messages": [{"role": "user", "content": PROMPT, "audio": audio, "audios": [audio]}],
        "expected_answer": transcript,
        "dataset": BENCHMARK_NAME,
        "subset_for_metrics": "test-clean_white-noise_0db",
        "sample_id": utterance_id,
        "task_type": "ASR",
        "audio_duration": float(duration),
        "question": PROMPT,
        "profile_version": PROFILE_VERSION,
        "profile_seed": PROFILE_SEED,
        "target_snr_db": TARGET_SNR_DB,
        "noise_distribution": "uniform[-0.5,0.5)",
        "noise_prng": "numpy.PCG64",
        "realization_sha256": realization_sha256,
        "realization_seed_uint128": str(realization_seed),
        "clean_source_url": LIBRISPEECH_URL,
        "clean_source_md5": LIBRISPEECH_MD5,
        "protocol_source_url": WHISPER_PAPER_URL,
        "protocol_source_section": WHISPER_PROTOCOL_SECTION,
        "adt_source_repo": ADT_REPO_URL,
        "adt_source_revision": ADT_REVISION,
        "adt_source_sha256": ADT_SOURCE_SHA256,
        **mix_metadata,
    }


def prepare(data_dir: Path, max_samples: int | None = None) -> Path:
    test_clean_dir = _download_librispeech(data_dir / "sources")
    clean_rows = _read_librispeech_rows(test_clean_dir)
    if max_samples is not None and max_samples > 0:
        clean_rows = clean_rows[:max_samples]

    audio_dir = data_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_prefix = f"/data/{BENCHMARK_NAME}/audio"
    entries = []
    for utterance_id, transcript, clean_path in tqdm(clean_rows, desc="Adding ADT white noise at 0 dB"):
        clean, sample_rate = sf.read(clean_path, dtype="float64", always_2d=False)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"{clean_path}: expected {SAMPLE_RATE} Hz, got {sample_rate}")
        clean = _as_mono(clean)
        noise, realization_sha256, realization_seed = generate_white_noise(utterance_id, clean.size)
        mixture, metadata = mix_adt_white_noise(clean, noise)

        output_path = audio_dir / f"{utterance_id}.wav"
        sf.write(output_path, mixture, SAMPLE_RATE, subtype="FLOAT")
        entries.append(
            create_manifest_entry(
                utterance_id=utterance_id,
                transcript=transcript,
                duration=mixture.size / SAMPLE_RATE,
                audio_path=f"{audio_prefix}/{output_path.name}",
                realization_sha256=realization_sha256,
                realization_seed=realization_seed,
                mix_metadata=metadata,
            )
        )

    manifest_path = data_dir / "test.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as fout:
        fout.writelines(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries)
    print(f"Wrote {len(entries)} examples to {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Developer smoke subset; omit for benchmark")
    args = parser.parse_args()
    prepare(args.data_dir or _default_data_dir(), args.max_samples)


if __name__ == "__main__":
    main()
