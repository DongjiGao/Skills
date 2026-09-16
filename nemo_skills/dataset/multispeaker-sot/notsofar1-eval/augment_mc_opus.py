#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0

"""Apply independent RIR, MUSAN noise, and upper-band LPF augmentation.

The NOTSOFAR1 MC delivery contains separate mono files named ``*_CH1.opus``
through ``*_CH7.opus``. Each file receives a deterministic but independently
sampled small-room RIR, MUSAN noise recording, SNR, and low-pass cutoff.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal


DATA_ROOT = Path(
    "/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr/data"
)
NOTSOFAR1_ROOT = (
    DATA_ROOT
    / "multi_speaker_asr/diarization_e2e/above4spk_finetune_datasets/notsofar1"
)
DEFAULT_INPUT = NOTSOFAR1_ROOT / "audio_mc_opus"
DEFAULT_OUTPUT = NOTSOFAR1_ROOT / "audio_mc_aug_opus"
DEFAULT_MUSAN = DATA_ROOT / "musan"
DEFAULT_RIRS = DATA_ROOT / "RIRS_NOISES"
_WORKER_CONFIG: dict[str, Any] = {}


@dataclass(frozen=True)
class AugmentationRecord:
    source: str
    output: str
    rir: str
    noise: str
    source_sample_rate: int
    sample_rate: int
    frames: int
    dry_weight: float
    wet_weight: float
    noise_snr_db: float
    lowpass_hz: float
    peak_gain: float


def _seed_for(path: Path, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{path.as_posix()}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] != 1:
        raise ValueError(f"Expected mono audio, found {audio.shape[1]} channels: {path}")
    return audio[:, 0], sample_rate


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    divisor = math.gcd(source_rate, target_rate)
    return signal.resample_poly(audio, target_rate // divisor, source_rate // divisor).astype(
        np.float32, copy=False
    )


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + 1e-12))


def _fit_noise(noise: np.ndarray, frames: int, rng: np.random.Generator) -> np.ndarray:
    if not noise.size:
        raise ValueError("Noise recording is empty")
    if noise.size < frames:
        repeats = math.ceil(frames / noise.size)
        noise = np.tile(noise, repeats)
    start = int(rng.integers(0, noise.size - frames + 1))
    return noise[start : start + frames]


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def augment_one(
    source: Path,
    rir_path: Path,
    *,
    input_root: Path,
    output_root: Path,
    noises: tuple[Path, ...],
    seed: int,
    output_sample_rate: int,
    wet_min: float,
    wet_max: float,
    snr_min_db: float,
    snr_max_db: float,
    cutoff_min_hz: float,
    cutoff_max_hz: float,
    overwrite: bool,
) -> AugmentationRecord:
    relative = source.relative_to(input_root)
    destination = output_root / relative
    rng = np.random.default_rng(_seed_for(relative, seed))
    noise_path = noises[int(rng.integers(len(noises)))]
    snr_db = float(rng.uniform(snr_min_db, snr_max_db))
    wet_weight = float(rng.uniform(wet_min, wet_max))
    dry_weight = 1.0 - wet_weight

    dry, source_sample_rate = _read_mono(source)
    if source_sample_rate != 16000:
        raise ValueError(
            f"Expected 16 kHz NOTSOFAR1 signal, found {source_sample_rate}: {source}"
        )
    dry = _resample(dry, source_sample_rate, output_sample_rate)
    sample_rate = output_sample_rate
    nyquist = sample_rate / 2.0
    effective_cutoff_max = min(cutoff_max_hz, nyquist * 0.975)
    if not 0.0 < cutoff_min_hz < effective_cutoff_max:
        raise ValueError(
            f"LPF range [{cutoff_min_hz}, {cutoff_max_hz}] is invalid for {sample_rate} Hz audio"
        )
    cutoff_hz = float(rng.uniform(cutoff_min_hz, effective_cutoff_max))

    rir, rir_rate = _read_mono(rir_path)
    rir = _resample(rir, rir_rate, sample_rate)
    rir_energy = float(np.linalg.norm(rir.astype(np.float64)))
    if rir_energy <= 1e-8:
        raise ValueError(f"RIR has no energy: {rir_path}")
    rir = rir / rir_energy
    wet = signal.fftconvolve(dry, rir, mode="full")[: dry.size].astype(np.float32)
    dry_rms = _rms(dry)
    wet_rms = _rms(wet)
    if wet_rms > 1e-8:
        wet *= dry_rms / wet_rms
    augmented = dry_weight * dry + wet_weight * wet

    noise, noise_rate = _read_mono(noise_path)
    noise = _resample(noise, noise_rate, sample_rate)
    noise = _fit_noise(noise, augmented.size, rng)
    noise_rms = _rms(noise)
    target_noise_rms = _rms(augmented) / (10.0 ** (snr_db / 20.0))
    if noise_rms > 1e-8:
        augmented += noise * (target_noise_rms / noise_rms)

    lowpass = signal.butter(6, cutoff_hz, btype="lowpass", fs=sample_rate, output="sos")
    augmented = signal.sosfiltfilt(lowpass, augmented).astype(np.float32)
    peak = float(np.max(np.abs(augmented), initial=0.0))
    peak_gain = min(1.0, 0.98 / peak) if peak > 0.0 else 1.0
    augmented *= peak_gain

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        info = sf.info(destination)
        if info.samplerate == sample_rate and info.channels == 1:
            return AugmentationRecord(
                source=str(source),
                output=str(destination),
                rir=str(rir_path),
                noise=str(noise_path),
                source_sample_rate=source_sample_rate,
                sample_rate=sample_rate,
                frames=len(dry),
                dry_weight=dry_weight,
                wet_weight=wet_weight,
                noise_snr_db=snr_db,
                lowpass_hz=cutoff_hz,
                peak_gain=peak_gain,
            )
        raise ValueError(f"Existing output is invalid: {destination}")

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".opus", dir=destination.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        sf.write(temporary_path, augmented, sample_rate, format="OGG", subtype="OPUS")
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)

    return AugmentationRecord(
        source=str(source),
        output=str(destination),
        rir=str(rir_path),
        noise=str(noise_path),
        source_sample_rate=source_sample_rate,
        sample_rate=sample_rate,
        frames=len(dry),
        dry_weight=dry_weight,
        wet_weight=wet_weight,
        noise_snr_db=snr_db,
        lowpass_hz=cutoff_hz,
        peak_gain=peak_gain,
    )


def _initialize_worker(config: dict[str, Any]) -> None:
    global _WORKER_CONFIG
    _WORKER_CONFIG = config


def _augment_worker(job: tuple[Path, Path]) -> AugmentationRecord:
    source, rir_path = job
    return augment_one(source, rir_path, **_WORKER_CONFIG)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--musan", type=Path, default=DEFAULT_MUSAN)
    parser.add_argument("--rirs", type=Path, default=DEFAULT_RIRS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--output-sample-rate", type=int, default=48000)
    parser.add_argument("--wet-min", type=float, default=0.3)
    parser.add_argument("--wet-max", type=float, default=0.5)
    parser.add_argument("--snr-min-db", type=float, default=10.0)
    parser.add_argument("--snr-max-db", type=float, default=20.0)
    parser.add_argument("--cutoff-min-hz", type=float, default=4500.0)
    parser.add_argument("--cutoff-max-hz", type=float, default=12000.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sources = sorted(args.input.rglob("*.opus"))
    rirs = tuple(sorted(args.rirs.glob("simulated_rirs/smallroom/**/*.wav")))
    noises = tuple(sorted((args.musan / "noise").rglob("*.wav")))
    if not sources:
        raise FileNotFoundError(f"No source Opus files found under {args.input}")
    if not rirs:
        raise FileNotFoundError(f"No small-room RIR files found under {args.rirs}")
    if not noises:
        raise FileNotFoundError(f"No MUSAN noise files found under {args.musan / 'noise'}")
    if len(rirs) < len(sources):
        raise ValueError(f"Need at least one unique RIR per source: {len(rirs)} < {len(sources)}")
    if not 0.0 <= args.wet_min <= args.wet_max < 1.0:
        raise ValueError("Wet range must satisfy 0 <= wet-min <= wet-max < 1")

    rir_rng = np.random.default_rng(args.seed)
    rir_indices = rir_rng.permutation(len(rirs))[: len(sources)]
    jobs = [(source, rirs[int(rir_index)]) for source, rir_index in zip(sources, rir_indices)]

    print(
        f"sources={len(sources)} rirs={len(rirs)} noises={len(noises)} "
        f"workers={args.workers}",
        flush=True,
    )
    records: list[AugmentationRecord] = []
    common = {
        "input_root": args.input,
        "output_root": args.output,
        "noises": noises,
        "seed": args.seed,
        "output_sample_rate": args.output_sample_rate,
        "wet_min": args.wet_min,
        "wet_max": args.wet_max,
        "snr_min_db": args.snr_min_db,
        "snr_max_db": args.snr_max_db,
        "cutoff_min_hz": args.cutoff_min_hz,
        "cutoff_max_hz": args.cutoff_max_hz,
        "overwrite": args.overwrite,
    }
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_initialize_worker,
        initargs=(common,),
    ) as executor:
        futures = [executor.submit(_augment_worker, job) for job in jobs]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            records.append(future.result())
            if index % 25 == 0 or index == len(sources):
                print(f"completed={index}/{len(sources)}", flush=True)

    records.sort(key=lambda record: record.source)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "augmentation_manifest.jsonl"
    temporary_manifest = manifest.with_suffix(".jsonl.tmp")
    with temporary_manifest.open("w", encoding="utf-8") as output:
        for record in records:
            payload = asdict(record)
            payload["source"] = _relative_or_absolute(Path(record.source), args.input)
            payload["output"] = _relative_or_absolute(Path(record.output), args.output)
            payload["rir"] = _relative_or_absolute(Path(record.rir), args.rirs)
            payload["noise"] = _relative_or_absolute(Path(record.noise), args.musan)
            output.write(json.dumps(payload, ensure_ascii=False) + "\n")
    temporary_manifest.replace(manifest)
    print(f"Wrote {len(records)} augmented files and {manifest}", flush=True)


if __name__ == "__main__":
    main()
