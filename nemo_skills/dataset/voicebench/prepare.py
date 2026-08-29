# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Prepare every public VoiceBench task/config/split leaf.

The public transcription is retained solely for scoring.  Runnable examples
contain an empty text content and audio, matching VoiceBench's audio-only model
input.  MTBench is materialized as data-only because the pinned official repo
publishes neither a working two-turn inference path nor a scorer for it.
"""

import argparse
import io
import json
import os
import shutil
from pathlib import Path

HF_REPO_ID = "hlt-lab/voicebench"
HF_REVISION = "b02edcef1330480be3a11bd6f7434ac32f05ad08"
OFFICIAL_CODE_REVISION = "3c3b0d3a7a956f745305eb348f5e03ce7ec73dad"

SPEAKER_SPLITS = (
    "en_AU_Wavenet_A_1.0_0.0_0.0",
    "en_AU_Wavenet_B_1.0_0.0_0.0",
    "en_IN_Wavenet_A_1.0_0.0_0.0",
    "en_IN_Wavenet_B_1.0_0.0_0.0",
    "en_GB_Wavenet_A_1.0_0.0_0.0",
    "en_GB_Wavenet_B_1.0_0.0_0.0",
    "en_US_Wavenet_A_1.0_0.0_0.0",
    "en_US_Wavenet_C_1.0_0.0_0.0",
    "en_US_Wavenet_A_1.5_0.0_0.0",
    "en_US_Wavenet_A_2.0_0.0_0.0",
    "en_US_Wavenet_A_0.5_0.0_0.0",
)
MMSU_SPLITS = (
    "law51",
    "engineering107",
    "other546",
    "biology172",
    "business236",
    "economics280",
    "health406",
    "philosophy305",
    "psychology317",
    "history104",
    "chemistry167",
    "physics383",
)
SD_QA_SPLITS = ("aus", "gbr", "ind_n", "ind_s", "irl", "kenya", "nga", "nzl", "phl", "usa", "zaf")

PROFILES = {
    "advbench": ("advbench", ("test",)),
    "alpacaeval": ("alpacaeval", ("test",)),
    "alpacaeval-full": ("alpacaeval_full", ("test",)),
    "alpacaeval-speaker": ("alpacaeval_speaker", SPEAKER_SPLITS),
    "bbh": ("bbh", ("test",)),
    "commoneval": ("commoneval", ("test",)),
    "ifeval": ("ifeval", ("test",)),
    "mmsu": ("mmsu", MMSU_SPLITS),
    "mtbench": ("mtbench", ("test",)),
    "openbookqa": ("openbookqa", ("test",)),
    "sd-qa": ("sd-qa", SD_QA_SPLITS),
    "wildvoice": ("wildvoice", ("test",)),
}

EXPECTED_ROWS = {
    "advbench/test": 520,
    "alpacaeval/test": 199,
    "alpacaeval_full/test": 636,
    "bbh/test": 1000,
    "commoneval/test": 200,
    "ifeval/test": 345,
    "mtbench/test": 46,
    "openbookqa/test": 455,
    "wildvoice/test": 1000,
    **{f"alpacaeval_speaker/{split}": 636 for split in SPEAKER_SPLITS},
    **{f"mmsu/{split}": int(''.join(c for c in split if c.isdigit())) for split in MMSU_SPLITS},
    **{f"sd-qa/{split}": 553 for split in SD_QA_SPLITS},
}


def _default_data_dir() -> Path:
    root = Path(os.environ.get("NEMO_SKILLS_DATA_DIR", Path(__file__).resolve().parent))
    return root / "voicebench" if os.environ.get("NEMO_SKILLS_DATA_DIR") else root


def _safe(value: str) -> str:
    return value.replace("/", "_").replace(".", "_")


def _write_audio(audio: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if audio.get("bytes") is not None:
        with open(destination, "wb") as fout:
            fout.write(audio["bytes"])
    elif audio.get("path"):
        shutil.copyfile(audio["path"], destination)
    else:
        raise ValueError("VoiceBench audio row contains neither bytes nor a path")


def _audio_destination(profile: str, split: str, index: int, audio: dict, data_dir: Path) -> Path:
    suffix = Path(str(audio.get("path") or "audio.wav")).suffix or ".wav"
    return data_dir / profile / "audio" / _safe(split) / f"{index:06d}{suffix}"


def _audio_duration_seconds(audio: dict, destination: Path) -> float:
    """Read an exact duration from the materialized header or decode-disabled HF bytes."""
    import soundfile

    sources = []
    if destination.is_file():
        sources.append(destination)
    if audio.get("bytes") is not None:
        sources.append(io.BytesIO(audio["bytes"]))
    if audio.get("path"):
        sources.append(audio["path"])
    last_error = None
    for source in sources:
        try:
            info = soundfile.info(source)
            duration = float(info.frames) / float(info.samplerate)
            if duration > 0:
                return duration
        except (OSError, RuntimeError, TypeError, ValueError, ZeroDivisionError) as exc:
            last_error = exc
    raise ValueError("Could not determine a positive VoiceBench audio duration from its header") from last_error


def format_entry(
    row: dict, profile: str, config: str, split: str, audio_paths: list[str], audio_durations: list[float]
) -> dict:
    """Format a row without putting the public transcription into model text."""
    item = dict(row)
    for audio_column in ("audio", "audio1", "audio2"):
        item.pop(audio_column, None)
    item.update(
        {
            "voicebench_task": config,
            "voicebench_profile": profile,
            "dataset_config": config,
            "dataset_split": split,
            "subset_for_metrics": split,
            "dataset_revision": HF_REVISION,
            "official_code_revision": OFFICIAL_CODE_REVISION,
        }
    )
    if profile == "mtbench":
        item["messages"] = []
        item["protocol_supported"] = False
        item["protocol_gap"] = "Official VoiceBench publishes no working MTBench inference path or scorer."
        item["audio1_path"] = audio_paths[0]
        item["audio2_path"] = audio_paths[1]
        item["audio1_duration"] = audio_durations[0]
        item["audio2_duration"] = audio_durations[1]
    else:
        item["messages"] = [
            {"role": "user", "content": "", "audio": {"path": audio_paths[0], "duration": audio_durations[0]}}
        ]
        item["audio_path"] = audio_paths[0]
        item["audio_filepath"] = audio_paths[0]
        item["protocol_supported"] = True
    return item


def _load_leaf(config: str, split: str):
    from datasets import Audio, load_dataset

    dataset = load_dataset(HF_REPO_ID, config, split=split, revision=HF_REVISION)
    for column in ("audio", "audio1", "audio2"):
        if column in dataset.column_names:
            dataset = dataset.cast_column(column, Audio(decode=False))
    return dataset


def prepare_profile(profile: str, data_dir: Path, audio_prefix: str, no_audio: bool, max_samples: int | None) -> int:
    config, splits = PROFILES[profile]
    entries = []
    for split in splits:
        dataset = _load_leaf(config, split)
        expected = EXPECTED_ROWS[f"{config}/{split}"]
        if max_samples is None and len(dataset) != expected:
            raise ValueError(f"Pinned VoiceBench {config}/{split} has {len(dataset)} rows; expected {expected}")
        limit = len(dataset) if max_samples is None else min(max_samples, len(dataset))
        for index, row in enumerate(dataset.select(range(limit))):
            columns = ("audio1", "audio2") if profile == "mtbench" else ("audio",)
            public_paths = []
            durations = []
            for column in columns:
                audio = row[column]
                destination = _audio_destination(profile, split, index, audio, data_dir)
                if not no_audio and not destination.is_file():
                    _write_audio(audio, destination)
                relative = destination.relative_to(data_dir)
                public_paths.append(f"{audio_prefix.rstrip('/')}/{relative.as_posix()}")
                durations.append(_audio_duration_seconds(audio, destination))
            entries.append(format_entry(row, profile, config, split, public_paths, durations))

    output = Path(__file__).resolve().parent / profile / "test.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wt", encoding="utf-8") as fout:
        fout.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in entries)
    return len(entries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare pinned public VoiceBench leaves")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--audio-prefix", default=None)
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--max-samples", type=int, default=-1)
    args = parser.parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else _default_data_dir()
    audio_prefix = args.audio_prefix or ("/data/voicebench" if os.environ.get("NEMO_SKILLS_DATA_DIR") else str(data_dir))
    max_samples = args.max_samples if args.max_samples > 0 else None
    profiles = [value.strip() for value in args.profiles.split(",") if value.strip()]
    unknown = sorted(set(profiles) - set(PROFILES))
    if unknown:
        raise ValueError(f"Unknown VoiceBench profiles: {unknown}")
    for profile in profiles:
        count = prepare_profile(profile, data_dir, audio_prefix, args.no_audio, max_samples)
        print(f"Wrote {count} rows for voicebench.{profile}")
    print(f"Dataset revision: {HF_REVISION}")
    print(f"Official code revision: {OFFICIAL_CODE_REVISION}")


if __name__ == "__main__":
    main()
