#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0

"""Migrate Turing, MLC-SLM 2, and NOTSOFAR1 data to parent-plus-leaves.

The migration is intentionally narrow. It only reads the three known parent
manifests and the two known NOTSOFAR1 leaf manifests. Media and sidecars are
moved within the data root, manifests are replaced atomically, and legacy
artifact directories are removed only when empty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

BENCHMARK_NAME = "multispeaker-sot"
TURING_PARENT = "turing-diarization-v1"
TURING_LANGUAGES = ("ar", "el", "en", "es", "fr", "it", "ja", "ko", "pt", "ru")
MLC_PARENT = "mlc-slm-2-dev-13L"
MLC_LOCALES = ("de", "en", "es", "es-MX", "fr", "fr-CA", "it", "ja", "ko", "pt", "pt-BR", "ru", "th")
NOTSOFAR_PARENT = "notsofar1-eval"
NOTSOFAR_LEAVES = ("notsofar1-eval-ihm", "notsofar1-eval-sc")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def rewrite_path_tokens(row: dict, old: str, new: str) -> dict:
    payload = json.dumps(row, ensure_ascii=False).replace(old, new)
    return json.loads(payload)


def move_file(source: Path, destination: Path, *, execute: bool) -> None:
    if destination.exists():
        if source.exists():
            raise FileExistsError(f"Both source and destination exist: {source}, {destination}")
        return
    if not source.is_file():
        raise FileNotFoundError(source)
    if execute:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)


def remove_empty_tree(path: Path, *, execute: bool) -> None:
    if not path.exists():
        return
    remaining = sorted(path.rglob("*"), reverse=True)
    if execute:
        for item in remaining:
            if item.is_dir():
                item.rmdir()
        path.rmdir()


def copy_module(package_root: Path, data_root: Path, alias: str, *, execute: bool) -> None:
    source = package_root / alias / "__init__.py"
    destination = data_root / alias / "__init__.py"
    if not source.is_file():
        raise FileNotFoundError(source)
    if execute:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def migrate_turing(data_root: Path, package_root: Path, *, execute: bool) -> list[dict]:
    rows = read_jsonl(data_root / TURING_PARENT / "test.jsonl")
    grouped = {f"{TURING_PARENT}-{language}": [] for language in TURING_LANGUAGES}
    for row in rows:
        language = str(row.get("source_lang", "")).lower()
        if language not in TURING_LANGUAGES:
            raise ValueError(f"Unexpected Turing source_lang {language!r} in {row.get('sample_id')}")
        leaf = f"{TURING_PARENT}-{language}"
        filename = f"{row['sample_id']}.opus"
        move_file(
            data_root / "audio" / TURING_PARENT / filename,
            data_root / "audio" / leaf / filename,
            execute=execute,
        )

        seglst_name = Path(row["reference_seglst_filepath"]).name
        move_file(
            data_root / "seglst" / TURING_PARENT / seglst_name,
            data_root / "seglst" / leaf / seglst_name,
            execute=execute,
        )
        rttm_name = Path(row["rttm_filepath"]).name
        move_file(
            data_root / "rttm" / TURING_PARENT / rttm_name,
            data_root / "rttm" / leaf / rttm_name,
            execute=execute,
        )

        migrated = rewrite_path_tokens(row, f"/{TURING_PARENT}/", f"/{leaf}/")
        migrated["language"] = language
        migrated["dataset_id"] = leaf
        migrated["subset_for_metrics"] = leaf
        grouped[leaf].append(migrated)

    merged: list[dict] = []
    for leaf, leaf_rows in grouped.items():
        if execute:
            write_jsonl_atomic(data_root / leaf / "test.jsonl", leaf_rows)
            copy_module(package_root, data_root, leaf, execute=True)
        merged.extend(leaf_rows)
    if execute:
        write_jsonl_atomic(data_root / TURING_PARENT / "test.jsonl", merged)
        copy_module(package_root, data_root, TURING_PARENT, execute=True)
        remove_empty_tree(data_root / "audio" / TURING_PARENT, execute=True)
        remove_empty_tree(data_root / "seglst" / TURING_PARENT, execute=True)
        remove_empty_tree(data_root / "rttm" / TURING_PARENT, execute=True)
    return merged


def migrate_mlc(data_root: Path, package_root: Path, *, execute: bool) -> list[dict]:
    rows = read_jsonl(data_root / MLC_PARENT / "test.jsonl")
    grouped = {f"{MLC_PARENT}-{locale}": [] for locale in MLC_LOCALES}
    for row in rows:
        locale = str(row.get("locale", ""))
        if locale not in MLC_LOCALES:
            raise ValueError(f"Unexpected MLC-SLM 2 locale {locale!r} in {row.get('sample_id')}")
        leaf = f"{MLC_PARENT}-{locale}"
        audio_name = Path(row["audio_filepath"]).name
        move_file(
            data_root / "audio" / MLC_PARENT / locale / audio_name,
            data_root / "audio" / leaf / audio_name,
            execute=execute,
        )
        seglst_name = Path(row["reference_seglst_filepath"]).name
        move_file(
            data_root / "seglst" / MLC_PARENT / locale / seglst_name,
            data_root / "seglst" / leaf / seglst_name,
            execute=execute,
        )
        rttm_name = Path(row["rttm_filepath"]).name
        move_file(
            data_root / "rttm" / MLC_PARENT / locale / rttm_name,
            data_root / "rttm" / leaf / rttm_name,
            execute=execute,
        )

        migrated = rewrite_path_tokens(row, f"/{MLC_PARENT}/{locale}/", f"/{leaf}/")
        migrated["dataset_id"] = leaf
        migrated["subset_for_metrics"] = leaf
        grouped[leaf].append(migrated)

    merged: list[dict] = []
    for leaf, leaf_rows in grouped.items():
        if execute:
            write_jsonl_atomic(data_root / leaf / "test.jsonl", leaf_rows)
            copy_module(package_root, data_root, leaf, execute=True)
        merged.extend(leaf_rows)
    if execute:
        write_jsonl_atomic(data_root / MLC_PARENT / "test.jsonl", merged)
        copy_module(package_root, data_root, MLC_PARENT, execute=True)
        remove_empty_tree(data_root / "audio" / MLC_PARENT, execute=True)
        remove_empty_tree(data_root / "seglst" / MLC_PARENT, execute=True)
        remove_empty_tree(data_root / "rttm" / MLC_PARENT, execute=True)
    return merged


def migrate_notsofar(data_root: Path, package_root: Path, *, execute: bool) -> list[dict]:
    merged: list[dict] = []
    for leaf in NOTSOFAR_LEAVES:
        rows = read_jsonl(data_root / leaf / "test.jsonl")
        for row in rows:
            row["dataset_id"] = leaf
            row["subset_for_metrics"] = leaf
        if execute:
            write_jsonl_atomic(data_root / leaf / "test.jsonl", rows)
            copy_module(package_root, data_root, leaf, execute=True)
        merged.extend(rows)
    if execute:
        write_jsonl_atomic(data_root / NOTSOFAR_PARENT / "test.jsonl", merged)
        copy_module(package_root, data_root, NOTSOFAR_PARENT, execute=True)
    return merged


def update_turing_stats(data_root: Path) -> None:
    manifest = data_root / TURING_PARENT / "test.jsonl"
    stats_path = data_root / TURING_PARENT / "nsv_mapping_stats.json"
    if not stats_path.is_file():
        return
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats.update(
        {
            "input": str(manifest),
            "output": str(manifest),
            "input_sha256": digest,
            "output_sha256": digest,
            "rows": sum(1 for line in manifest.open(encoding="utf-8") if line.strip()),
        }
    )
    temporary = stats_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(stats_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Skills data root containing multispeaker-sot")
    parser.add_argument("--execute", action="store_true", help="Apply the migration; otherwise perform checks only")
    args = parser.parse_args()

    data_root = args.data_dir.expanduser().resolve() / BENCHMARK_NAME
    package_root = Path(__file__).resolve().parent
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)

    turing = migrate_turing(data_root, package_root, execute=args.execute)
    mlc = migrate_mlc(data_root, package_root, execute=args.execute)
    notsofar = migrate_notsofar(data_root, package_root, execute=args.execute)
    if args.execute:
        update_turing_stats(data_root)
    mode = "migrated" if args.execute else "validated"
    print(f"{mode}: Turing={len(turing)}, MLC-SLM-2={len(mlc)}, NOTSOFAR1={len(notsofar)}")


if __name__ == "__main__":
    main()
