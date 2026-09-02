# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

"""Prepare the deterministic public Marco-LongSpeech test subset.

The source revision is immutable. Prepared requests retain the exact stored
user instruction and supply only the referenced audio path; the assistant
message is moved to ``expected_answer`` and never included in the request.
"""

import argparse
import copy
import fcntl
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

REPO_ID = "ATH-MaaS/Marco_Longspeech"
REVISION = "bb50d8c11fe9aa9fc5e899a0d941e3b40acbc740"
MIN_AUDIO_DURATION_SECONDS = 0.02
PROVENANCE_FILENAME = "provenance.json"

SUPPORTED_TASKS = {
    "asr": "ASR",
    "translation": "translation",
    "summary": "summary",
    "content-separation": "content_separation",
    "speaker-count": "speaker_count",
    "language-detection": "language_detection",
}
SOURCE_CONTRACTS = {
    "asr": {
        "rows": 15274,
        "prepared_rows": 15273,
        "unique_audio": 15273,
        "sha256": "2af259b2455eb4135fa82864f5227d974362c39e8724c6b64b0bc550b6065c95",
    },
    "translation": {
        "rows": 6309,
        "unique_audio": 4687,
        "sha256": "fa3e8fa5bdc2f5d251f726e428577863abaa1cc70dda3d0b430c4db7d480964f",
    },
    "summary": {
        "rows": 937,
        "unique_audio": 937,
        "sha256": "8f5383be11645bf9c32082448581cbb336c5aa13d822a29a18a96254d5fb1031",
    },
    "content-separation": {
        "rows": 1263,
        "unique_audio": 1263,
        "sha256": "714f0e26724f6e61a96c8d27dede5f6164e82e7c87be751436d860cac8c65fc8",
    },
    "speaker-count": {
        "rows": 1263,
        "unique_audio": 1263,
        "sha256": "e7e7d73ebfee052c7c66636050ce673fdff241a4899aa2b4b650941e091625db",
    },
    "language-detection": {
        "rows": 3170,
        "unique_audio": 3170,
        "sha256": "520aed2a12a9d447edb6734e0b0c95d37f19267a0cf06de0fc920d11beeef660",
    },
}
# The immutable source manifest references this one absent repository object.
# Preserve the raw manifest contract, but exclude only this proven unservable
# row from the prepared audio-complete subset and retain original row indices.
SOURCE_MISSING_AUDIO = {
    "asr": {"LongSpeech_p1/wavs/part_02/020934.wav": (12375,)},
}
UNSUPPORTED_TASKS = {
    "temporal-qa": (
        "Temporal QA requires the unpublished GPT-4-Turbo YES/NO/PARTIALLY "
        "judge prompt, so an exact public scorer cannot be reproduced."
    ),
    "emotion": (
        "Emotion is not fully specified: the paper's coarse-label mapping "
        "omits the Joyful label present in the pinned public test split."
    ),
}


def _download_repo_file(filename: str, local_dir: Path) -> Path:
    """Download one file from the immutable public dataset revision."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            revision=REVISION,
            local_dir=str(local_dir),
        )
    )


def _parse_tasks(value: str) -> list[str]:
    requested = [item.strip() for item in value.split(",") if item.strip()]
    if not requested or requested == ["all"]:
        return list(SUPPORTED_TASKS)

    unknown = [task for task in requested if task not in SUPPORTED_TASKS]
    if unknown:
        explanations = [
            f"{task}: {UNSUPPORTED_TASKS[task]}" if task in UNSUPPORTED_TASKS else f"{task}: unknown task"
            for task in unknown
        ]
        raise ValueError("Unsupported LongSpeech task(s): " + " ".join(explanations))
    return list(dict.fromkeys(requested))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_strings(values: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(values), separators=(",", ":")).encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, description: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"LongSpeech {description} is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"LongSpeech {description} must be a JSON object: {path}")

    return value


def _safe_audio_relpath(value: str, source_task: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"LongSpeech {source_task} row is missing its audio-only path")
    if "\\" in value:
        raise ValueError(f"LongSpeech {source_task} audio path uses a non-portable separator: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"LongSpeech {source_task} audio path is not a safe relative path: {value!r}")
    return path.as_posix()


def _validate_source_row(row: dict, source_task: str) -> tuple[dict, dict]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError(f"LongSpeech {source_task} row must have exactly two stored messages")
    user, assistant = messages
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        raise ValueError(f"LongSpeech {source_task} row has unexpected message roles")
    _safe_audio_relpath(user.get("audio"), source_task)
    if "content" not in user or "content" not in assistant:
        raise ValueError(f"LongSpeech {source_task} row is missing stored message content")
    return user, assistant


def _format_row(
    row: dict,
    *,
    benchmark_task: str,
    source_task: str,
    row_index: int,
    audio_prefix: str,
    audio_duration: float,
) -> dict:
    """Convert one source row without modifying its exact stored messages."""
    user, assistant = _validate_source_row(row, source_task)
    audio_relpath = _safe_audio_relpath(user["audio"], source_task)
    audio_path = f"{audio_prefix.rstrip('/')}/{audio_relpath}"
    prepared_user = copy.deepcopy(user)
    prepared_user["audio"] = {"path": audio_path, "duration": audio_duration}

    return {
        "messages": [prepared_user],
        "expected_answer": copy.deepcopy(assistant["content"]),
        "audio_filepath": audio_path,
        "language": row.get("language"),
        "longspeech_task": benchmark_task,
        "source_task": row.get("task"),
        "source_audio_filepath": user["audio"],
        "source_messages": copy.deepcopy(row["messages"]),
        "source_revision": REVISION,
        "source_manifest_sha256": SOURCE_CONTRACTS[benchmark_task]["sha256"],
        "sample_id": f"{source_task}:test:{row_index:06d}",
        "task_type": "LONGSPEECH",
    }


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from error


def _download_audio_files(audio_relpaths: list[str], audio_dir: Path, workers: int) -> None:
    def download(audio_relpath: str) -> Path:
        return _download_repo_file(audio_relpath, audio_dir)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        downloaded = list(pool.map(download, audio_relpaths))

    invalid = [str(path) for path in downloaded if not path.is_file() or path.stat().st_size <= 0]
    if invalid:
        preview = ", ".join(invalid[:5])
        raise ValueError(f"LongSpeech downloaded audio closure is incomplete: {preview}")


def _read_audio_durations(audio_relpaths: list[str], audio_dir: Path) -> dict[str, float]:
    """Validate every local audio header and return exact durations for balancing."""
    import soundfile as sf

    durations: dict[str, float] = {}
    for audio_relpath in audio_relpaths:
        path = audio_dir / audio_relpath
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"LongSpeech audio is missing or empty: {path}")
        try:
            info = sf.info(str(path))
        except Exception as error:
            raise ValueError(f"LongSpeech audio header is unreadable: {path}") from error
        if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
            raise ValueError(
                f"LongSpeech audio header is invalid: {path} "
                f"(frames={info.frames}, samplerate={info.samplerate}, channels={info.channels})"
            )
        duration = info.frames / info.samplerate
        if duration < MIN_AUDIO_DURATION_SECONDS:
            raise ValueError(
                f"LongSpeech audio is shorter than the {MIN_AUDIO_DURATION_SECONDS:.3f}s feature minimum: "
                f"{path} ({duration:.6f}s)"
            )
        durations[audio_relpath] = duration
    return durations


def _validate_source_contract(benchmark_task: str, source_manifest: Path, rows: list[dict]) -> str:
    contract = SOURCE_CONTRACTS[benchmark_task]
    digest = _sha256_file(source_manifest)
    if digest != contract["sha256"]:
        raise ValueError(f"LongSpeech {benchmark_task} source hash drift: expected {contract['sha256']}, got {digest}")
    if len(rows) != contract["rows"]:
        raise ValueError(f"LongSpeech {benchmark_task} row-count drift: expected {contract['rows']}, got {len(rows)}")
    return digest


def _select_audio_complete_rows(benchmark_task: str, source_task: str, rows: list[dict]) -> list[tuple[int, dict]]:
    expected_missing = SOURCE_MISSING_AUDIO.get(benchmark_task, {})
    observed_missing: dict[str, list[int]] = {}
    selected: list[tuple[int, dict]] = []

    for row_index, row in enumerate(rows):
        user, _assistant = _validate_source_row(row, source_task)
        audio_relpath = _safe_audio_relpath(user["audio"], source_task)
        if audio_relpath in expected_missing:
            observed_missing.setdefault(audio_relpath, []).append(row_index)
            continue
        selected.append((row_index, row))

    observed_missing_tuples = {path: tuple(row_indices) for path, row_indices in observed_missing.items()}
    if observed_missing_tuples != expected_missing:
        raise ValueError(
            f"LongSpeech {benchmark_task} missing-audio contract drift: "
            f"expected {expected_missing}, got {observed_missing_tuples}"
        )

    contract = SOURCE_CONTRACTS[benchmark_task]
    expected_rows = contract.get("prepared_rows", contract["rows"])
    if len(selected) != expected_rows:
        raise ValueError(
            f"LongSpeech {benchmark_task} prepared row-count drift: expected {expected_rows}, got {len(selected)}"
        )
    return selected


def _task_paths(output_root: Path, benchmark_task: str) -> tuple[Path, Path]:
    task_dir = output_root / benchmark_task
    return task_dir / "test.jsonl", task_dir / PROVENANCE_FILENAME


def _publish_task_output(
    *,
    output_root: Path,
    benchmark_task: str,
    candidate: Path,
    receipt: dict,
    require_existing: bool,
    adopt_existing: bool,
) -> None:
    """Publish one deterministic leaf, or exactly revalidate it under a task lock."""
    output_file, receipt_path = _task_paths(output_root, benchmark_task)
    lock_path = output_root / ".task-locks" / f"{benchmark_task}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_sha256 = receipt["manifest_sha256"]

    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            sealed = (output_root / PROVENANCE_FILENAME).exists()
            if output_file.exists():
                if not output_file.is_file() or _sha256_file(output_file) != candidate_sha256:
                    raise ValueError(
                        f"Existing LongSpeech {benchmark_task} manifest fails deterministic closure: {output_file}"
                    )
                if receipt_path.exists():
                    if _read_json(receipt_path, f"{benchmark_task} task receipt") != receipt:
                        raise ValueError(
                            f"Existing LongSpeech {benchmark_task} task receipt fails deterministic closure"
                        )
                elif sealed:
                    raise ValueError(f"Sealed LongSpeech root lacks the {benchmark_task} task receipt")
                elif not adopt_existing:
                    raise ValueError(
                        f"LongSpeech {benchmark_task} legacy manifest has no authenticated task receipt; "
                        "after proving its writer stopped, rerun with --task-only --no-audio --adopt-existing"
                    )
                else:
                    # Explicitly recover output made by the legacy serial adapter
                    # only after reconstructing exact bytes from the pinned source
                    # and validating every referenced audio header/duration.
                    _write_json_atomic(receipt_path, receipt)
            else:
                if require_existing or adopt_existing:
                    raise FileNotFoundError(f"LongSpeech {benchmark_task} manifest is missing: {output_file}")
                if sealed:
                    raise ValueError(f"Refusing to add LongSpeech {benchmark_task} to a sealed root")
                if receipt_path.exists():
                    raise ValueError(f"LongSpeech {benchmark_task} receipt exists without its manifest")
                output_file.parent.mkdir(parents=True, exist_ok=True)
                os.replace(candidate, output_file)
                _write_json_atomic(receipt_path, receipt)

            if _sha256_file(output_file) != candidate_sha256:
                raise ValueError(f"Published LongSpeech {benchmark_task} manifest changed during validation")
            if _read_json(receipt_path, f"{benchmark_task} task receipt") != receipt:
                raise ValueError(f"Published LongSpeech {benchmark_task} receipt changed during validation")
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _publish_final_receipt(output_root: Path, *, audio_prefix: str) -> int:
    """Require all six authenticated leaves before sealing the shared root."""
    task_summaries: dict[str, dict] = {}
    total_rows = 0
    for benchmark_task, source_task in SUPPORTED_TASKS.items():
        output_file, receipt_path = _task_paths(output_root, benchmark_task)
        if not output_file.is_file() or not receipt_path.is_file():
            raise FileNotFoundError(f"LongSpeech {benchmark_task} has no complete task manifest and receipt")
        receipt = _read_json(receipt_path, f"{benchmark_task} task receipt")
        contract = SOURCE_CONTRACTS[benchmark_task]
        expected_rows = contract.get("prepared_rows", contract["rows"])
        expected = {
            "schema_version": 1,
            "dataset": "longspeech",
            "task": benchmark_task,
            "manifest": f"{benchmark_task}/test.jsonl",
            "source_repo": REPO_ID,
            "source_revision": REVISION,
            "source_manifest": f"LongSpeechQA/{source_task}/test.jsonl",
            "source_manifest_sha256": contract["sha256"],
            "rows": expected_rows,
            "unique_audio": contract["unique_audio"],
            "audio_prefix": audio_prefix.rstrip("/"),
            "minimum_audio_duration_seconds": MIN_AUDIO_DURATION_SECONDS,
        }
        for field, value in expected.items():
            if receipt.get(field) != value:
                raise ValueError(
                    f"LongSpeech {benchmark_task} task receipt drift for {field}: "
                    f"expected {value!r}, got {receipt.get(field)!r}"
                )
        manifest_sha256 = receipt.get("manifest_sha256")
        audio_inventory_sha256 = receipt.get("audio_inventory_sha256")
        if not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64:
            raise ValueError(f"LongSpeech {benchmark_task} task receipt has no manifest SHA-256")
        if not isinstance(audio_inventory_sha256, str) or len(audio_inventory_sha256) != 64:
            raise ValueError(f"LongSpeech {benchmark_task} task receipt has no audio-inventory SHA-256")
        if _sha256_file(output_file) != manifest_sha256:
            raise ValueError(f"LongSpeech {benchmark_task} task manifest hash mismatch")
        task_summaries[benchmark_task] = {
            "manifest": receipt["manifest"],
            "manifest_sha256": manifest_sha256,
            "source_manifest_sha256": receipt["source_manifest_sha256"],
            "rows": receipt["rows"],
            "unique_audio": receipt["unique_audio"],
            "audio_inventory_sha256": audio_inventory_sha256,
        }
        total_rows += receipt["rows"]

    expected_total = sum(contract.get("prepared_rows", contract["rows"]) for contract in SOURCE_CONTRACTS.values())
    if total_rows != expected_total:
        raise ValueError(f"LongSpeech total-row drift: expected {expected_total}, got {total_rows}")
    receipt = {
        "schema_version": 1,
        "dataset": "longspeech",
        "source_repo": REPO_ID,
        "source_revision": REVISION,
        "tasks": task_summaries,
        "total_rows": total_rows,
        "audio_prefix": audio_prefix.rstrip("/"),
        "minimum_audio_duration_seconds": MIN_AUDIO_DURATION_SECONDS,
    }
    final_path = output_root / PROVENANCE_FILENAME
    lock_path = output_root / ".task-locks" / "finalize.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if final_path.exists():
                if _read_json(final_path, "final receipt") != receipt:
                    raise ValueError("Existing LongSpeech final receipt fails deterministic closure")
            else:
                _write_json_atomic(final_path, receipt)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return total_rows


def finalize_longspeech(output_root: Path, *, audio_prefix: str) -> int:
    """Download nothing; require receipts, then reconstruct every leaf before sealing."""
    for benchmark_task in SUPPORTED_TASKS:
        output_file, receipt_path = _task_paths(output_root, benchmark_task)
        if not output_file.is_file() or not receipt_path.is_file():
            raise FileNotFoundError(f"LongSpeech {benchmark_task} has no complete task manifest and receipt")
    for benchmark_task in SUPPORTED_TASKS:
        prepare_task(
            benchmark_task,
            output_root=output_root,
            audio_prefix=audio_prefix,
            download_audio=False,
            download_workers=1,
            require_existing=True,
        )
    return _publish_final_receipt(output_root, audio_prefix=audio_prefix)


def prepare_task(
    benchmark_task: str,
    *,
    output_root: Path,
    audio_prefix: str,
    download_audio: bool,
    download_workers: int,
    require_existing: bool = False,
    adopt_existing: bool = False,
) -> int:
    source_task = SUPPORTED_TASKS[benchmark_task]
    source_filename = f"LongSpeechQA/{source_task}/test.jsonl"
    output_file, _receipt_path = _task_paths(output_root, benchmark_task)
    if (require_existing or adopt_existing) and not output_file.is_file():
        raise FileNotFoundError(f"LongSpeech {benchmark_task} manifest is missing: {output_file}")

    source_root = output_root / "_source"
    if require_existing:
        source_manifest = source_root / source_filename
        if not source_manifest.is_file():
            raise FileNotFoundError(
                f"LongSpeech {benchmark_task} cached source manifest is missing: {source_manifest}"
            )
    else:
        source_manifest = _download_repo_file(source_filename, source_root)

    source_rows = list(_iter_jsonl(source_manifest))
    source_digest = _validate_source_contract(benchmark_task, source_manifest, source_rows)
    rows = _select_audio_complete_rows(benchmark_task, source_task, source_rows)
    audio_relpaths = []
    for _row_index, row in rows:
        user, _assistant = _validate_source_row(row, source_task)
        audio_relpaths.append(_safe_audio_relpath(user["audio"], source_task))
    audio_relpaths = list(dict.fromkeys(audio_relpaths))
    expected_audio = SOURCE_CONTRACTS[benchmark_task]["unique_audio"]
    if len(audio_relpaths) != expected_audio:
        raise ValueError(
            f"LongSpeech {benchmark_task} audio-count drift: expected {expected_audio}, got {len(audio_relpaths)}"
        )

    if download_audio:
        _download_audio_files(
            audio_relpaths,
            output_root / "audio",
            workers=download_workers,
        )
    audio_durations = _read_audio_durations(audio_relpaths, output_root / "audio")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_file.with_name(f".{output_file.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with temp_output.open("w", encoding="utf-8") as output:
            for row_index, row in rows:
                audio_relpath = _safe_audio_relpath(row["messages"][0]["audio"], source_task)
                formatted = _format_row(
                    row,
                    benchmark_task=benchmark_task,
                    source_task=source_task,
                    row_index=row_index,
                    audio_prefix=audio_prefix,
                    audio_duration=audio_durations[audio_relpath],
                )
                output.write(json.dumps(formatted, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        receipt = {
            "schema_version": 1,
            "dataset": "longspeech",
            "task": benchmark_task,
            "manifest": f"{benchmark_task}/test.jsonl",
            "manifest_sha256": _sha256_file(temp_output),
            "source_repo": REPO_ID,
            "source_revision": REVISION,
            "source_manifest": source_filename,
            "source_manifest_sha256": source_digest,
            "rows": len(rows),
            "unique_audio": len(audio_relpaths),
            "audio_inventory_sha256": _sha256_strings(audio_relpaths),
            "audio_prefix": audio_prefix.rstrip("/"),
            "minimum_audio_duration_seconds": MIN_AUDIO_DURATION_SECONDS,
        }
        _publish_task_output(
            output_root=output_root,
            benchmark_task=benchmark_task,
            candidate=temp_output,
            receipt=receipt,
            require_existing=require_existing,
            adopt_existing=adopt_existing,
        )
    finally:
        temp_output.unlink(missing_ok=True)

    print(f"Prepared longspeech.{benchmark_task}: {len(rows)} test rows, {len(audio_relpaths)} unique audio files")
    return len(rows)


def _validate_recovery_modes(
    tasks: list[str],
    *,
    task_only: bool,
    finalize_only: bool,
    adopt_existing: bool,
    no_audio: bool,
) -> None:
    if task_only and len(tasks) != 1:
        raise ValueError("--task-only requires exactly one LongSpeech task")
    if finalize_only and tasks != list(SUPPORTED_TASKS):
        raise ValueError("--finalize-only requires --tasks=all")
    if adopt_existing and (not task_only or not no_audio):
        raise ValueError("--adopt-existing requires --task-only and --no-audio")
    if finalize_only and no_audio:
        raise ValueError("--finalize-only already downloads nothing; do not combine it with --no-audio")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        default=os.getenv("NEMO_SKILLS_DATA_DIR"),
        help="Base NeMo Skills data directory; output is written under longspeech/.",
    )
    parser.add_argument(
        "--tasks",
        default="all",
        help=(
            "Comma-separated supported task leaves or all. Temporal QA and emotion "
            "are deliberately unsupported because their public protocols are incomplete."
        ),
    )
    parser.add_argument(
        "--audio-prefix",
        default="/data/longspeech/audio",
        help="Container-visible prefix corresponding to <data_dir>/longspeech/audio.",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help=(
            "Skip downloads only; referenced audio must already exist under "
            "<data_dir>/longspeech/audio so headers and durations can still be validated."
        ),
    )
    parser.add_argument("--download-workers", type=int, default=16)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--task-only",
        action="store_true",
        help="Require exactly one --tasks leaf and publish only that authenticated task receipt.",
    )
    mode.add_argument(
        "--finalize-only",
        action="store_true",
        help="Download nothing; require and revalidate all six task receipts/audio, then publish the final receipt.",
    )
    parser.add_argument(
        "--adopt-existing",
        action="store_true",
        help=(
            "Authenticate an exact legacy manifest after its writer is proven stopped; "
            "requires --task-only and --no-audio."
        ),
    )
    args = parser.parse_args()

    if args.download_workers < 1:
        raise ValueError("--download-workers must be at least 1")
    tasks = _parse_tasks(args.tasks)
    _validate_recovery_modes(
        tasks,
        task_only=args.task_only,
        finalize_only=args.finalize_only,
        adopt_existing=args.adopt_existing,
        no_audio=args.no_audio,
    )

    if args.data_dir:
        output_root = Path(args.data_dir) / "longspeech"
    else:
        package_root = Path(__file__).parent
        if "site-packages" in str(package_root) or "dist-packages" in str(package_root):
            raise SystemExit("Missing --data_dir and NEMO_SKILLS_DATA_DIR; refusing to write into site-packages.")
        output_root = package_root
    output_root.mkdir(parents=True, exist_ok=True)
    if args.finalize_only:
        total = finalize_longspeech(output_root, audio_prefix=args.audio_prefix)
        print(f"Finalized {total} LongSpeech public test examples across {len(SUPPORTED_TASKS)} tasks")
        return

    total = 0
    for task in tasks:
        total += prepare_task(
            task,
            output_root=output_root,
            audio_prefix=args.audio_prefix,
            download_audio=not args.no_audio,
            download_workers=args.download_workers,
            adopt_existing=args.adopt_existing,
        )

    if tasks == list(SUPPORTED_TASKS):
        _publish_final_receipt(output_root, audio_prefix=args.audio_prefix)
    print(f"Prepared {total} LongSpeech public test examples across {len(tasks)} tasks")


if __name__ == "__main__":
    main()
