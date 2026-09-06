# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Exact-once, resumable hosted judging for already-generated VoiceBench rows."""

import argparse
import asyncio
import hashlib
import json
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from nemo_skills.evaluation.evaluator.voicebench import (
    OFFICIAL_CODE_REVISION,
    OPEN_TASKS,
    VoiceBenchEvaluator,
    load_pinned_pedant,
)

SOURCE_INDEX_KEY = "voicebench_judge_source_index"
JUDGE_FIELDS = {
    "official_code_revision",
    "response",
    "score",
    "voicebench_open_ratings",
    "voicebench_qa_gpt_correct",
    "voicebench_qa_pedant_correct",
    "voicebench_judge_model",
    "voicebench_judge_api_key_env",
}


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_complete(row: dict) -> bool:
    score = row.get("score")
    if not isinstance(score, list) or len(score) != 3:
        return False
    if row.get("official_code_revision") != OFFICIAL_CODE_REVISION:
        return False
    task = row.get("voicebench_task")
    if task in OPEN_TASKS:
        ratings = row.get("voicebench_open_ratings")
        return isinstance(ratings, list) and len(ratings) == 3
    if task == "sd-qa":
        return isinstance(row.get("voicebench_qa_gpt_correct"), bool) and isinstance(
            row.get("voicebench_qa_pedant_correct"), bool
        )
    return False


def _atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> None:
    started = datetime.now(timezone.utc)
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "output.jsonl"
    done_file = output_dir / "output.jsonl.done"
    partial_file = output_dir / "output.jsonl.partial"
    receipt_file = output_dir / "judge-receipt.json"

    if done_file.is_file():
        rows = _read_jsonl(output_file)
        if rows and all(_is_complete(row) for row in rows):
            print(f"VoiceBench judge already complete: rows={len(rows)}")
            return
        raise RuntimeError("VoiceBench judge done marker exists but output is incomplete")
    if not input_file.is_file():
        raise FileNotFoundError(input_file)
    if not os.environ.get(args.api_key_env):
        raise RuntimeError(f"VoiceBench judge requires {args.api_key_env}")

    source_rows = _read_jsonl(input_file)
    if not source_rows:
        raise RuntimeError("VoiceBench judge input is empty")
    tasks = {str(row.get("voicebench_task")) for row in source_rows}
    if len(tasks) != 1 or not (tasks <= (set(OPEN_TASKS) | {"sd-qa"})):
        raise RuntimeError(f"VoiceBench judge input has unsupported task set: {sorted(tasks)}")

    completed: dict[int, dict] = {}
    if partial_file.is_file():
        for row in _read_jsonl(partial_file):
            index = row.get(SOURCE_INDEX_KEY)
            if isinstance(index, int) and 0 <= index < len(source_rows) and _is_complete(row):
                completed[index] = row

    thread_state = threading.local()

    def get_evaluator() -> VoiceBenchEvaluator:
        evaluator = getattr(thread_state, "evaluator", None)
        if evaluator is None:
            evaluator = VoiceBenchEvaluator(
                {
                    "openai_api_key_env": args.api_key_env,
                    "openai_base_url": args.base_url,
                    "judge_model": args.judge_model,
                    "judge_max_retries": args.judge_max_retries,
                    "judge_error_sleep_seconds": args.judge_error_sleep_seconds,
                    "judge_success_sleep_seconds": args.judge_success_sleep_seconds,
                    "enable_pedant": False,
                }
            )
            thread_state.evaluator = evaluator
        return evaluator

    pedant_evaluator = None
    if tasks == {"sd-qa"}:
        pedant_evaluator = VoiceBenchEvaluator({"enable_pedant": True})
        pedant_evaluator._pedant = load_pinned_pedant()

    def judge_one(index: int) -> tuple[int, dict]:
        source = source_rows[index]
        row = {key: value for key, value in source.items() if key not in JUDGE_FIELDS}
        row[SOURCE_INDEX_KEY] = index
        result = asyncio.run(get_evaluator().eval_single(row))
        row.update(result)
        if row.get("voicebench_task") == "sd-qa":
            response = row.get("generation") if isinstance(row.get("generation"), str) else ""
            row["voicebench_qa_pedant_correct"] = pedant_evaluator._pedant_sync(row, response)
        row["voicebench_judge_model"] = args.judge_model
        row["voicebench_judge_api_key_env"] = args.api_key_env
        if not _is_complete(row):
            raise RuntimeError("VoiceBench judge produced an incomplete row")
        return index, row

    missing = [index for index in range(len(source_rows)) if index not in completed]
    errors = Counter()
    partial_mode = "a" if partial_file.is_file() else "w"
    with partial_file.open(partial_mode, encoding="utf-8") as partial_sink:  # noqa: SIM117
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(judge_one, index): index for index in missing}
            for ordinal, future in enumerate(as_completed(futures), start=1):
                try:
                    index, row = future.result()
                except Exception as exc:  # noqa: BLE001 - preserve partial exact-once progress
                    errors[type(exc).__name__] += 1
                    continue
                completed[index] = row
                partial_sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                partial_sink.flush()
                if ordinal % 100 == 0 or ordinal == len(missing):
                    print(
                        f"VoiceBench judge progress: completed={len(completed)}/{len(source_rows)} "
                        f"failed={sum(errors.values())}"
                    )

    if errors or len(completed) != len(source_rows):
        print(f"VoiceBench judge incomplete: completed={len(completed)}/{len(source_rows)} errors={dict(errors)}")
        raise SystemExit(1)

    ordered_rows = [completed[index] for index in range(len(source_rows))]
    if not all(_is_complete(row) for row in ordered_rows):
        raise RuntimeError("VoiceBench judge final validation failed")
    _atomic_write_jsonl(output_file, ordered_rows)
    finished = datetime.now(timezone.utc)
    _atomic_write_json(
        receipt_file,
        {
            "api_key_env": args.api_key_env,
            "base_url": args.base_url,
            "completed_rows": len(ordered_rows),
            "finished_utc": finished.isoformat(),
            "input_file": str(input_file),
            "input_sha256": _sha256(input_file),
            "judge_model": args.judge_model,
            "max_workers": args.max_workers,
            "official_code_revision": OFFICIAL_CODE_REVISION,
            "output_file": str(output_file),
            "output_sha256": _sha256(output_file),
            "request_n": 3,
            "started_utc": started.isoformat(),
        },
    )
    done_file.touch()
    print(f"VoiceBench judge complete: rows={len(ordered_rows)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--api-key-env", default="NVIDIA_INFERENCE_HUB_API_KEY")
    parser.add_argument("--base-url", default="https://inference-api.nvidia.com/v1")
    parser.add_argument("--judge-model", default="openai/openai/gpt-4o-mini")
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--judge-max-retries", type=int, default=25)
    parser.add_argument("--judge-error-sleep-seconds", type=float, default=30.0)
    parser.add_argument("--judge-success-sleep-seconds", type=float, default=0.5)
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    run(args)


if __name__ == "__main__":
    main()
