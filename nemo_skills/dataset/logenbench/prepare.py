# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Materialize the frozen DFW LoGenBench manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_ROWS = 1_162
SELECTION = Path(__file__).with_name("selection.json")


def _data_root(source_manifest: Path) -> Path:
    for parent in source_manifest.parents:
        if parent.name.startswith("balanced-data-") or parent.name.startswith(
            "immutable-hr9b-step4000-pc-spgi-clean-balanced-"
        ):
            return parent
    raise ValueError(f"Cannot infer mounted data root from {source_manifest}")


def _resolve_audio_path(raw_path: str, data_root: Path) -> str:
    path = Path(raw_path)
    if raw_path in {"/data", "/dataset"}:
        candidate = data_root
    elif raw_path.startswith("/data/"):
        candidate = data_root / raw_path.removeprefix("/data/")
    elif raw_path.startswith("/dataset/"):
        candidate = data_root / raw_path.removeprefix("/dataset/")
    elif path.is_absolute():
        candidate = path
    else:
        raise ValueError(f"Audio path is not absolute: {raw_path!r}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"Resolved audio is not a regular file: {resolved}")
    if not str(resolved).startswith("/lustre/"):
        raise ValueError(f"Resolved audio is not on Lustre: {resolved}")
    return str(resolved)


def _audio_paths(row: dict[str, Any]) -> set[str]:
    found = set()
    for key in ("audio_path", "audio_filepath"):
        if isinstance(row.get(key), str):
            found.add(row[key])
    for message in row.get("messages", []):
        audio = message.get("audio")
        if isinstance(audio, str):
            found.add(audio)
        elif isinstance(audio, dict) and isinstance(audio.get("path"), str):
            found.add(audio["path"])
        audios = message.get("audios")
        if isinstance(audios, list):
            for item in audios:
                if isinstance(item, str):
                    found.add(item)
                elif isinstance(item, dict) and isinstance(item.get("path"), str):
                    found.add(item["path"])
    return found


def _replace_paths(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        return replacements.get(value, value)
    if isinstance(value, list):
        return [_replace_paths(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_paths(item, replacements) for key, item in value.items()}
    return value


def _duration_bucket(duration: float) -> str:
    if duration < 60.0:
        return "short-form"
    if duration < 600.0:
        return "mid-form"
    return "long-form"


def build_manifest(selection_path: Path, output_path: Path) -> dict[str, Any]:
    inventory = json.loads(selection_path.read_text())
    rows = []
    audio_cache: dict[tuple[str, str], str] = {}
    for source in inventory:
        source_manifest = Path(source["source_manifest"])
        data_root = Path(source["source_data_root"]) if source.get("source_data_root") else _data_root(source_manifest)
        wanted = set(source["source_indexes"])
        selected = {}
        with source_manifest.open(encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                if index in wanted:
                    selected[index] = json.loads(line)
        missing = wanted - selected.keys()
        if missing:
            raise ValueError(f"Missing indexes in {source_manifest}: {sorted(missing)}")

        for source_index in source["source_indexes"]:
            source_row = selected[source_index]
            raw_paths = _audio_paths(source_row)
            if not raw_paths:
                raise ValueError(f"Selected row has no audio: {source_manifest}:{source_index}")
            replacements = {}
            for raw_path in raw_paths:
                cache_key = (str(data_root), raw_path)
                if cache_key not in audio_cache:
                    audio_cache[cache_key] = _resolve_audio_path(raw_path, data_root)
                replacements[raw_path] = audio_cache[cache_key]
            row = _replace_paths(source_row, replacements)
            duration = float(row.get("duration") or 0.0)
            if duration <= 0.0:
                message_durations = [
                    float(message["audio"]["duration"])
                    for message in row.get("messages", [])
                    if isinstance(message.get("audio"), dict) and message["audio"].get("duration")
                ]
                duration = sum(message_durations)
            if duration <= 0.0:
                raise ValueError(f"Selected row has no positive duration: {source_manifest}:{source_index}")

            original_subset = row.get("subset_for_metrics")
            extra_fields = dict(row.get("extra_fields") or {})
            extra_fields.update(
                {
                    "logenbench_source_run": source["source_run"],
                    "logenbench_source_benchmark": source["source_benchmark"],
                    "logenbench_source_manifest": str(source_manifest),
                    "logenbench_source_index": source_index,
                    "logenbench_source_subset": original_subset,
                }
            )
            row.update(
                {
                    "id": f"logenbench-{len(rows):04d}",
                    "duration": duration,
                    "subset_for_metrics": _duration_bucket(duration),
                    "inference_overrides": {"tokens_to_generate": int(source["tokens_to_generate"])},
                    "extra_fields": extra_fields,
                }
            )
            rows.append(row)

    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(rows)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    payload = output_path.read_bytes()
    return {
        "manifest": str(output_path),
        "rows": len(rows),
        "unique_audio_paths": len({path for row in rows for path in _audio_paths(row)}),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "duration_buckets": {
            bucket: sum(row["subset_for_metrics"] == bucket for row in rows)
            for bucket in ("short-form", "mid-form", "long-form")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    args = parser.parse_args()
    receipt = build_manifest(args.selection, args.output_dir / "logenbench" / "test.jsonl")
    receipt_path = args.output_dir / "logenbench" / "manifest_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
