# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import gzip
import json
import re
import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path

SYSTEM_MESSAGE = ""
USER_MESSAGE = "Transcribe this recording."
AUDIO_ROOT_ENV = "NEMO_SKILLS_AUDIO_ROOT"
DEFAULT_AUDIO_ROOT = "/data"


def expand_file_pattern(pattern: str) -> list[str]:
    """Expand corpusview-style _OP_N..M_CL_ ranges in a path string."""
    match = re.search(r"_OP_(\d+)\.\.(\d+)_CL_", pattern)
    if match is None:
        return [pattern]

    start_s, end_s = match.groups()
    width = max(len(start_s), len(end_s))
    start, end = int(start_s), int(end_s)
    if end < start:
        raise ValueError(f"Invalid range in path pattern: {pattern}")

    paths = []
    for idx in range(start, end + 1):
        replacement = f"{idx:0{width}d}"
        expanded = pattern[: match.start()] + replacement + pattern[match.end() :]
        paths.extend(expand_file_pattern(expanded))
    return paths


def iter_jsonl(path: Path) -> Iterator[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt" if path.suffix == ".gz" else "r"
    with opener(path, mode, encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                yield json.loads(line)


def coerce_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("text", "expected_answer", "answer", "transcript", "reference"):
            if key in value:
                return coerce_text(value[key])
        return " ".join(coerce_text(item) for item in value.values()).strip()
    if isinstance(value, (list, tuple)):
        return " ".join(coerce_text(item) for item in value).strip()
    return str(value).strip()


def extract_manifest_text(row: dict) -> str:
    for key in ("text", "expected_answer", "answer", "transcript", "reference", "text_pc", "text_origin"):
        text = coerce_text(row.get(key))
        if text:
            return text

    supervisions = row.get("supervisions")
    if isinstance(supervisions, list) and supervisions:
        text = coerce_text(supervisions[0].get("text"))
        if text:
            return text

    return ""


def extract_audio_path(row: dict) -> str:
    for key in ("audio_filepath", "audio_path", "audio", "recording_filepath"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, list) and value and isinstance(value[0], str):
            return value[0]
        if isinstance(value, dict):
            path = value.get("path") or value.get("audio_filepath")
            if isinstance(path, str) and path:
                return path

    recording = row.get("recording")
    if isinstance(recording, dict):
        sources = recording.get("sources")
        if isinstance(sources, list) and sources:
            source = sources[0]
            if isinstance(source, dict) and source.get("source"):
                return source["source"]

    return ""


def extract_duration(row: dict) -> float | None:
    value = row.get("duration")
    if value is None and isinstance(row.get("recording"), dict):
        value = row["recording"].get("duration")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def copy_audio_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    shutil.copy2(src, dst)


def asr_record(
    *,
    benchmark: str,
    sample_id: str,
    audio_path: str,
    expected_answer: str,
    duration: float | None,
    user_prompt: str = USER_MESSAGE,
    entity_list: list[str] | None = None,
    domain_label: str | None = None,
    subset_for_metrics: str | None = None,
    text_pc: str | None = None,
    text_origin: str | None = None,
    extra_fields: dict | None = None,
) -> dict:
    audio_meta = {"path": audio_path}
    if duration is not None:
        audio_meta["duration"] = float(duration)

    record = {
        "audio_filepath": audio_path,
        "audio_path": [audio_path],
        "expected_answer": expected_answer,
        "text": expected_answer,
        "task_type": "ASR",
        "dataset": benchmark,
        "subset_for_metrics": subset_for_metrics or benchmark,
        "sample_id": sample_id,
        "entity_list": entity_list or [],
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": user_prompt,
                "audio": audio_meta,
                "audios": [audio_meta],
            },
        ],
    }
    if domain_label:
        record["domain_label"] = domain_label
    if duration is not None:
        record["audio_duration"] = float(duration)
    if text_pc:
        record["text_pc"] = text_pc
    if text_origin:
        record["text_origin"] = text_origin
    if extra_fields:
        record.update(extra_fields)
    return record


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count
