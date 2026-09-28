# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Preparation for the preference-asr benchmark.

Ground-truth convention (per benchmark author):
    For preference-asr the ground truth is in the `preference_text` field of
    the source manifest, NOT in `text` / `answer`. Both the contextless and
    fine eval modes are scored against `preference_text`. The two modes
    differ only in the user prompt:
      - contextless: a plain "Transcribe this recording." prompt; measures
        whether the baseline model already produces the preferred form.
      - fine: the per-example styling instruction; measures whether the
        prompt makes the model produce the preferred form.

Both modes use preference-aware WER and retain the metadata needed to score
the requested format. Empty references for disfluency removal remain valid.
Named-entity-only metrics are not applicable to this style-rewrite benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from importlib import import_module
from pathlib import Path

import yaml

from nemo_skills.dataset.preference_asr_prompts import PROMPT_POLICY, rewrite_instruction

_audio_asr = import_module("nemo_skills.dataset.preference-asr.audio_asr")
AUDIO_ROOT_ENV = _audio_asr.AUDIO_ROOT_ENV
DEFAULT_AUDIO_ROOT = _audio_asr.DEFAULT_AUDIO_ROOT
asr_record = _audio_asr.asr_record
coerce_text = _audio_asr.coerce_text
copy_audio_file = _audio_asr.copy_audio_file
expand_file_pattern = _audio_asr.expand_file_pattern
extract_audio_path = _audio_asr.extract_audio_path
extract_duration = _audio_asr.extract_duration
iter_jsonl = _audio_asr.iter_jsonl
write_jsonl = _audio_asr.write_jsonl

DEFAULT_CORPUS_YAML = "/lustre/fsw/portfolios/llmservice/users/pzelasko/data/preference_asr_bench.yaml"
PROMPT_CONTEXTLESS = "Transcribe this recording."


def _resolve_path_key(entry: dict, field: str, preferred_key: str | None) -> str | None:
    paths = entry.get("paths")
    if not isinstance(paths, dict):
        return None
    if preferred_key and preferred_key in paths and field in paths[preferred_key]:
        return preferred_key
    default = paths.get("default", {})
    if isinstance(default, dict):
        key = default.get(field)
        if key in paths:
            return key
    for key, value in paths.items():
        if key == "default":
            continue
        if isinstance(value, dict) and field in value:
            return key
    return None


def _manifest_patterns(entry: dict, path_key: str | None) -> list[str]:
    if "manifest_filepath" in entry:
        return [entry["manifest_filepath"]]

    key = _resolve_path_key(entry, "manifest_filepath", path_key)
    if key is None:
        raise ValueError(f"No manifest_filepath found for corpus entry: {entry.get('corpus')}")
    value = entry["paths"][key]["manifest_filepath"]
    if isinstance(value, list):
        return value
    return [value]


def _audio_base(entry: dict, path_key: str | None) -> Path | None:
    if "audio_filepaths" in entry:
        return Path(str(entry["audio_filepaths"]))

    key = _resolve_path_key(entry, "audio_filepaths", path_key)
    if key is None:
        return None
    value = entry["paths"][key]["audio_filepaths"]
    if isinstance(value, list):
        value = value[0]
    return Path(str(value))


def _iter_manifest_rows(patterns: list[str]):
    for pattern in patterns:
        if pattern.startswith(("s3://", "ais://")):
            raise ValueError(
                f"Remote manifest path is not directly readable by this prepare script: {pattern}. "
                "Use a corpus YAML path key with IAD/local manifest files, or stage the manifest under /lustre first."
            )
        for expanded in expand_file_pattern(pattern):
            path = Path(expanded)
            if not path.exists():
                raise FileNotFoundError(f"Preference ASR manifest not found: {path}")
            yield from iter_jsonl(path)


def _corpus_yaml_from_args(args: argparse.Namespace) -> Path:
    value = args.corpus_yaml or os.environ.get("PREFERENCE_ASR_BENCH_YAML") or DEFAULT_CORPUS_YAML
    path = Path(value)
    if not path.exists():
        raise SystemExit(
            f"Preference ASR corpus YAML not found: {path}. "
            "Pass --corpus-yaml or set PREFERENCE_ASR_BENCH_YAML. "
            "The GitLab URL requires authentication and was not vendored here."
        )
    return path


def _resolve_audio_path(audio_value: str, audio_base: Path | None) -> Path:
    audio_path = Path(audio_value)
    if audio_path.is_absolute() or audio_base is None:
        return audio_path
    return audio_base / audio_path


def _prepare_row(
    *,
    row: dict,
    benchmark: str,
    output_dir: Path,
    audio_root: str,
    audio_base: Path | None,
    corpus_name: str,
    idx: int,
    copy_audio: bool,
    dry_run: bool = False,
) -> tuple[dict, dict] | None:
    # preference_text is the ground-truth reference for both eval modes.
    preference_text = coerce_text(row.get("preference_text"))
    instruction = coerce_text(row.get("instruction")) or PROMPT_CONTEXTLESS
    audio_value = extract_audio_path(row)
    if row.get("preference_text") is None or not audio_value:
        return None
    if not preference_text and not (row.get("preference_type") == "disfluencies" and row.get("direction") == "remove"):
        raise ValueError("Empty preference_text is only valid for disfluency removal")

    duration = extract_duration(row)
    source_audio = _resolve_audio_path(audio_value, audio_base)
    sample_id = coerce_text(row.get("id") or row.get("sample_id") or source_audio.stem or f"{corpus_name}-{idx}")

    if copy_audio:
        if not source_audio.exists():
            raise FileNotFoundError(f"Missing Preference ASR audio file: {source_audio}")
        dst_rel = Path("audio") / corpus_name / source_audio.name
        if not dry_run:
            copy_audio_file(source_audio, output_dir / dst_rel)
        eval_audio_path = f"{audio_root}/{benchmark}/{dst_rel.as_posix()}"
    else:
        eval_audio_path = str(source_audio)

    common = {
        "benchmark": benchmark,
        "sample_id": sample_id,
        "audio_path": eval_audio_path,
        "duration": duration,
        "domain_label": coerce_text(row.get("preference_type")) or "preference-asr",
        "subset_for_metrics": coerce_text(row.get("dataset")) or corpus_name,
        # Kept for downstream debugging; the eval scorer only reads expected_answer.
        "extra_fields": {
            "corpus": corpus_name,
            "preference_type": row.get("preference_type"),
            "sub_category": row.get("sub_category"),
            "direction": row.get("direction"),
            "target_case": row.get("target_case"),
            "instruction": rewrite_instruction(instruction),
            "source_instruction": instruction,
            "prompt_policy": PROMPT_POLICY,
            "preference_text": preference_text,
            "source_text": coerce_text(row.get("text")),
        },
    }
    # Entity lists are not needed by preference-aware WER.
    records = (
        asr_record(
            **common,
            expected_answer=preference_text,
            user_prompt=PROMPT_CONTEXTLESS,
            entity_list=[],
        ),
        asr_record(
            **common,
            expected_answer=preference_text,
            user_prompt=rewrite_instruction(instruction),
            entity_list=[],
        ),
    )

    for mode, record in zip(("contextless", "fine"), records):
        record["id"] = f"{benchmark}-{mode}-{corpus_name}-{idx:06d}"
        record["preference_prompt_variant"] = mode
    return records


def prepare(args: argparse.Namespace) -> int:
    corpus_yaml = _corpus_yaml_from_args(args)
    with corpus_yaml.open("r", encoding="utf-8") as fin:
        entries = yaml.safe_load(fin) or []
    if isinstance(entries, dict):
        entries = entries.get("corpora") or entries.get("datasets") or [entries]
    if not isinstance(entries, list):
        raise TypeError(f"Expected list-like corpus YAML, got {type(entries).__name__}: {corpus_yaml}")

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent
    audio_root = os.environ.get(AUDIO_ROOT_ENV, DEFAULT_AUDIO_ROOT).rstrip("/")
    benchmark = "preference-asr"
    rows_by_mode = {"contextless": [], "fine": []}
    dry_run = getattr(args, "dry_run", False)
    for mode in rows_by_mode:
        if (output_dir / mode / "test.jsonl").exists():
            raise FileExistsError(f"Use a new immutable output directory: {output_dir}")
    manifest_hashes = {}

    for entry in entries:
        corpus_name = str(entry.get("corpus") or entry.get("name") or benchmark).replace("/", "_")
        patterns = _manifest_patterns(entry, args.path_key)
        for pattern in patterns:
            for path in expand_file_pattern(pattern):
                manifest_hashes[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        audio_base = _audio_base(entry, args.path_key)
        for idx, row in enumerate(_iter_manifest_rows(patterns)):
            if args.max_samples is not None and len(rows_by_mode["contextless"]) >= args.max_samples:
                break
            records = _prepare_row(
                row=row,
                benchmark=benchmark,
                output_dir=output_dir,
                audio_root=audio_root,
                audio_base=audio_base,
                corpus_name=corpus_name,
                idx=idx,
                copy_audio=args.copy_audio,
                dry_run=dry_run,
            )
            if records is not None:
                contextless, fine = records
                rows_by_mode["contextless"].append(contextless)
                rows_by_mode["fine"].append(fine)
        if args.max_samples is not None and len(rows_by_mode["contextless"]) >= args.max_samples:
            break

    total = 0
    for mode, rows in rows_by_mode.items():
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate source row IDs in {mode}")
        count = len(rows) if dry_run else write_jsonl(output_dir / mode / "test.jsonl", rows)
        total += count
        print(f"Prepared {count} Preference ASR {mode} samples in {output_dir / mode}")
    provenance = {
        "benchmark": benchmark,
        "prompt_policy": PROMPT_POLICY,
        "scoring_version": "private-preference-aware-v1",
        "source_manifest_sha256": manifest_hashes,
        "profiles": {
            mode: {
                "rows": len(rows),
                "empty_references": sum(not row["expected_answer"] for row in rows),
                "manifest_sha256": hashlib.sha256(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
                ).hexdigest(),
            }
            for mode, rows in rows_by_mode.items()
        },
    }
    print(json.dumps(provenance, indent=2))
    if not dry_run:
        (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--corpus-yaml", default=None)
    parser.add_argument("--path-key", default="iad")
    parser.add_argument("--copy-audio", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
