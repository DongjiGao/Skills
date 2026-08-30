# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

"""Prepare the deterministic public Marco-LongSpeech test subset.

The source revision is immutable. Prepared requests retain the exact stored
user instruction and supply only the referenced audio path; the assistant
message is moved to ``expected_answer`` and never included in the request.
"""

import argparse
import copy
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

REPO_ID = "ATH-MaaS/Marco_Longspeech"
REVISION = "bb50d8c11fe9aa9fc5e899a0d941e3b40acbc740"

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
        "unique_audio": 15274,
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
) -> dict:
    """Convert one source row without modifying its exact stored messages."""
    user, assistant = _validate_source_row(row, source_task)
    audio_relpath = _safe_audio_relpath(user["audio"], source_task)
    audio_path = f"{audio_prefix.rstrip('/')}/{audio_relpath}"
    prepared_user = copy.deepcopy(user)
    prepared_user["audio"] = {"path": audio_path}

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


def _validate_source_contract(benchmark_task: str, source_manifest: Path, rows: list[dict]) -> str:
    contract = SOURCE_CONTRACTS[benchmark_task]
    digest = _sha256_file(source_manifest)
    if digest != contract["sha256"]:
        raise ValueError(f"LongSpeech {benchmark_task} source hash drift: expected {contract['sha256']}, got {digest}")
    if len(rows) != contract["rows"]:
        raise ValueError(f"LongSpeech {benchmark_task} row-count drift: expected {contract['rows']}, got {len(rows)}")
    return digest


def prepare_task(
    benchmark_task: str,
    *,
    output_root: Path,
    audio_prefix: str,
    download_audio: bool,
    download_workers: int,
) -> int:
    source_task = SUPPORTED_TASKS[benchmark_task]
    source_filename = f"LongSpeechQA/{source_task}/test.jsonl"
    source_root = output_root / "_source"
    source_manifest = _download_repo_file(source_filename, source_root)

    rows = list(_iter_jsonl(source_manifest))
    _validate_source_contract(benchmark_task, source_manifest, rows)
    audio_relpaths = []
    for row in rows:
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

    output_file = output_root / benchmark_task / "test.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_file.with_name(f".{output_file.name}.tmp")
    with temp_output.open("w", encoding="utf-8") as output:
        for row_index, row in enumerate(rows):
            formatted = _format_row(
                row,
                benchmark_task=benchmark_task,
                source_task=source_task,
                row_index=row_index,
                audio_prefix=audio_prefix,
            )
            output.write(json.dumps(formatted, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temp_output, output_file)

    print(f"Prepared longspeech.{benchmark_task}: {len(rows)} test rows, {len(audio_relpaths)} unique audio files")
    return len(rows)


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
        help="Prepare manifests only; referenced audio must already exist at audio-prefix.",
    )
    parser.add_argument("--download-workers", type=int, default=16)
    args = parser.parse_args()

    if args.download_workers < 1:
        raise ValueError("--download-workers must be at least 1")
    tasks = _parse_tasks(args.tasks)

    if args.data_dir:
        output_root = Path(args.data_dir) / "longspeech"
    else:
        package_root = Path(__file__).parent
        if "site-packages" in str(package_root) or "dist-packages" in str(package_root):
            raise SystemExit("Missing --data_dir and NEMO_SKILLS_DATA_DIR; refusing to write into site-packages.")
        output_root = package_root
    output_root.mkdir(parents=True, exist_ok=True)

    total = 0
    for task in tasks:
        total += prepare_task(
            task,
            output_root=output_root,
            audio_prefix=args.audio_prefix,
            download_audio=not args.no_audio,
            download_workers=args.download_workers,
        )
    print(f"Prepared {total} LongSpeech public test examples across {len(tasks)} tasks")


if __name__ == "__main__":
    main()
