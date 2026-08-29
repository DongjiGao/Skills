# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

"""Prepare the pinned MMAU-Pro non-music profile.

The default filter is intentionally literal: it removes only rows where
``category == "music"``. Use ``--music-policy=exclude-any-music`` to request
the separately documented stricter policy, which also removes mixed
top-level categories such as ``sound_music``. The benchmark name remains
``mmau-pro-nonmusic`` in either case; the selected policy and immutable source
pins are recorded in every prepared row.

The official evaluator source is pinned for provenance. NeMo Skills reuses
its existing local ports of all three evaluator paths: NV-Embed-v2 for closed
form, Qwen2.5-7B-Instruct judging for open ended, and deterministic AIF
constraint checks for instruction following.
"""

import argparse
import hashlib
import json
import os
import shutil
import stat
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from nemo_skills.dataset.utils import get_mcq_fields

HF_REPO_ID = "gamma-lab-umd/MMAU-Pro"
HF_REVISION = "b8516c217464d53be36d6dc334fda3cd578ed103"
OFFICIAL_SCORER_REPO = "sonalkum/MMAUPro"
OFFICIAL_SCORER_REVISION = "960e93a415e2267e987ae242516e21030a7d45dc"
PARQUET_SHA256 = "6a0d3d5ded7ea0ced0aa73365a75804a9eeec19db00216890b2349975d097c97"

BENCHMARK_NAME = "mmau-pro-nonmusic"
CATEGORY_ONLY_POLICY = "exclude-category-music"
EXCLUDE_ANY_MUSIC_POLICY = "exclude-any-music"
DEFAULT_MUSIC_POLICY = CATEGORY_ONLY_POLICY
MUSIC_POLICIES = (CATEGORY_ONLY_POLICY, EXCLUDE_ANY_MUSIC_POLICY)

# Exact census from the immutable test.parquet above. Keeping all categories,
# rather than only aggregate totals, detects upstream/schema drift before a
# manifest can be written under this benchmark name.
PINNED_CATEGORY_COUNTS = {
    "instruction following": 87,
    "multi": 430,
    "music": 1418,
    "music_speech": 46,
    "open": 625,
    "sound": 1048,
    "sound_music": 50,
    "sound_music_speech": 7,
    "sound_speech": 88,
    "spatial_audio": 325,
    "speech": 891,
    "voice_chat": 290,
}
PINNED_TOTAL_ROWS = 5305
PINNED_CATEGORY_MUSIC_ROWS = 1418
PINNED_CATEGORY_ONLY_ROWS = 3887
PINNED_EXCLUDE_ANY_MUSIC_ROWS = 3784
PINNED_AUDIO_ARITY_COUNTS = {1: 4849, 2: 430, 3: 26}


def _download_repo_file(filename: str, local_dir: Path) -> Path:
    """Download one file from the immutable dataset revision."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            repo_type="dataset",
            revision=HF_REVISION,
            local_dir=str(local_dir),
            token=os.environ.get("HF_TOKEN"),
        )
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pinned_rows(source_root: Path) -> list[dict]:
    """Download, authenticate, and load the small pinned metadata parquet."""
    import pyarrow.parquet as pq

    parquet_path = _download_repo_file("test.parquet", source_root)
    actual_hash = _sha256(parquet_path)
    if actual_hash != PARQUET_SHA256:
        raise ValueError(f"MMAU-Pro test.parquet hash mismatch: expected {PARQUET_SHA256}, got {actual_hash}")
    rows = pq.read_table(parquet_path).to_pylist()
    validate_pinned_census(rows)
    return rows


def _has_music_domain(category: object) -> bool:
    """Return whether a top-level domain label has a ``music`` component."""
    return isinstance(category, str) and "music" in category.split("_")


def select_rows(rows: list[dict], music_policy: str = DEFAULT_MUSIC_POLICY) -> list[dict]:
    """Apply one explicit policy without changing source order."""
    if music_policy not in MUSIC_POLICIES:
        raise ValueError(f"Unknown music policy {music_policy!r}; choose from {MUSIC_POLICIES}")

    if music_policy == CATEGORY_ONLY_POLICY:
        selected = [row for row in rows if row.get("category") != "music"]
        expected = PINNED_CATEGORY_ONLY_ROWS
    else:
        selected = [row for row in rows if not _has_music_domain(row.get("category"))]
        expected = PINNED_EXCLUDE_ANY_MUSIC_ROWS

    if len(rows) == PINNED_TOTAL_ROWS and len(selected) != expected:
        raise ValueError(f"Pinned MMAU-Pro {music_policy} filter yielded {len(selected)} rows; expected {expected}")
    return selected


def validate_pinned_census(rows: list[dict]) -> None:
    """Fail closed if the pinned metadata no longer matches its recorded census."""
    category_counts = Counter(row.get("category") for row in rows)
    arity_counts = Counter(len(row.get("audio_path") or []) for row in rows)
    if len(rows) != PINNED_TOTAL_ROWS:
        raise ValueError(f"Pinned MMAU-Pro has {len(rows)} rows; expected {PINNED_TOTAL_ROWS}")
    if category_counts != Counter(PINNED_CATEGORY_COUNTS):
        raise ValueError(f"Pinned MMAU-Pro category census mismatch: {dict(sorted(category_counts.items()))}")
    if arity_counts != Counter(PINNED_AUDIO_ARITY_COUNTS):
        raise ValueError(f"Pinned MMAU-Pro audio arity census mismatch: {dict(sorted(arity_counts.items()))}")


def _safe_relative_path(filename: str) -> PurePosixPath:
    """Validate a source archive member as a portable relative path."""
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"Invalid MMAU-Pro audio path: {filename!r}")
    path = PurePosixPath(filename)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe MMAU-Pro archive path: {filename!r}")
    return path


def _is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0o170000
    return mode == stat.S_IFLNK


def extract_selected_audio(archive_path: Path, required_paths: list[str], audio_root: Path) -> None:
    """Safely stream only selected audio members; never use ``extractall``."""
    relative_paths = [_safe_relative_path(path) for path in required_paths]
    required_names = {path.as_posix() for path in relative_paths}
    root = audio_root.resolve()
    audio_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as archive:
        members: dict[str, zipfile.ZipInfo] = {}
        for info in archive.infolist():
            if info.filename in required_names:
                if info.filename in members:
                    raise ValueError(f"Duplicate MMAU-Pro archive member: {info.filename}")
                members[info.filename] = info

        missing = sorted(required_names - members.keys())
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"Pinned MMAU-Pro data.zip is missing {len(missing)} selected audio files: {preview}"
            )

        for relative_path in relative_paths:
            info = members[relative_path.as_posix()]
            if info.is_dir() or _is_zip_symlink(info):
                raise ValueError(f"Unsafe MMAU-Pro audio archive member: {info.filename!r}")
            target = audio_root.joinpath(*relative_path.parts)
            resolved_target = target.resolve()
            if not resolved_target.is_relative_to(root):
                raise ValueError(f"MMAU-Pro archive member escapes audio root: {info.filename!r}")
            if target.is_file() and target.stat().st_size == info.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".partial")
            if not partial.resolve().is_relative_to(root):
                raise ValueError(f"MMAU-Pro temporary path escapes audio root: {partial!s}")
            with archive.open(info) as source, partial.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            partial.replace(target)


def _audio_duration_seconds(path: Path) -> float:
    """Read duration from the actual local audio headers."""
    import soundfile as sf

    info = sf.info(str(path))
    if info.samplerate <= 0 or info.frames <= 0:
        raise ValueError(f"MMAU-Pro audio has no valid frames: {path}")
    return round(float(info.frames) / float(info.samplerate), 6)


def _container_audio_path(relative_path: PurePosixPath, audio_prefix: str) -> str:
    return f"{audio_prefix.rstrip('/')}/{relative_path.as_posix()}"


def format_entry(
    entry: dict,
    *,
    local_audio_root: Path,
    audio_prefix: str,
    music_policy: str,
) -> dict:
    """Format one row, retaining audio order and measuring every duration."""
    source_audio_paths = entry.get("audio_path") or []
    if not isinstance(source_audio_paths, list) or not 1 <= len(source_audio_paths) <= 3:
        raise ValueError(f"MMAU-Pro row {entry.get('id')!r} must reference between one and three ordered audio files")

    ordered_audio = []
    for filename in source_audio_paths:
        relative_path = _safe_relative_path(filename)
        local_path = local_audio_root.joinpath(*relative_path.parts)
        if not local_path.is_file():
            raise FileNotFoundError(f"Missing MMAU-Pro audio for row {entry.get('id')!r}: {local_path}")
        ordered_audio.append(
            {
                "path": _container_audio_path(relative_path, audio_prefix),
                "duration": _audio_duration_seconds(local_path),
            }
        )

    choices = entry.get("choices", []) or []
    category = entry.get("category", "")
    formatted_entry = {
        "expected_answer": entry.get("answer"),
        **get_mcq_fields(entry["question"], choices),
        **{key: value for key, value in entry.items() if key not in {"answer", "audio_path"}},
        "source_audio_path": list(source_audio_paths),
        "dataset_revision": HF_REVISION,
        "official_scorer_repo": OFFICIAL_SCORER_REPO,
        "official_scorer_revision": OFFICIAL_SCORER_REVISION,
        "music_policy": music_policy,
        "duration": round(sum(audio["duration"] for audio in ordered_audio), 6),
    }

    if category not in {"open", "instruction following"}:
        formatted_entry["subset_for_metrics"] = category

    if category == "open":
        content = entry["question"]
    elif len(choices) > 1:
        options_text = "\n".join(f"{chr(65 + index)}. {choice}" for index, choice in enumerate(choices))
        content = f"{entry['question']}\n\n{options_text}"
    else:
        content = entry["question"]

    user_message = {"role": "user", "content": content}
    if len(ordered_audio) == 1:
        formatted_entry["audio_path"] = ordered_audio[0]["path"]
        user_message["audio"] = ordered_audio[0]
    else:
        formatted_entry["audio_path"] = [audio["path"] for audio in ordered_audio]
        user_message["audios"] = ordered_audio
    formatted_entry["messages"] = [user_message]
    return formatted_entry


def _category_leaf(category: object) -> str:
    if category == "instruction following":
        return "instruction_following"
    if category == "open":
        return "open_ended"
    return "closed_form"


def prepare(
    *,
    output_root: Path,
    audio_prefix: str,
    music_policy: str,
    download_audio: bool,
) -> dict[str, int]:
    source_root = output_root / "_source"
    source_root.mkdir(parents=True, exist_ok=True)
    rows = load_pinned_rows(source_root)
    selected = select_rows(rows, music_policy)

    required_audio_paths = list(dict.fromkeys(path for row in selected for path in (row.get("audio_path") or [])))
    local_audio_root = output_root / "audio"
    if download_audio:
        archive_path = _download_repo_file("data.zip", source_root)
        extract_selected_audio(archive_path, required_audio_paths, local_audio_root)

    output_files = {
        leaf: output_root / leaf / "test.jsonl" for leaf in ("closed_form", "open_ended", "instruction_following")
    }
    for path in output_files.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    handles = {leaf: path.open("w", encoding="utf-8") for leaf, path in output_files.items()}
    counts = Counter()
    try:
        for entry in selected:
            leaf = _category_leaf(entry.get("category"))
            formatted = format_entry(
                entry,
                local_audio_root=local_audio_root,
                audio_prefix=audio_prefix,
                music_policy=music_policy,
            )
            handles[leaf].write(json.dumps(formatted, ensure_ascii=False) + "\n")
            counts[leaf] += 1
    finally:
        for handle in handles.values():
            handle.close()

    print(
        f"Prepared {len(selected)} {BENCHMARK_NAME} rows with policy={music_policy}: "
        + ", ".join(f"{leaf}={counts[leaf]}" for leaf in output_files)
    )
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        default=os.getenv("NEMO_SKILLS_DATA_DIR"),
        help=f"Base NeMo Skills data directory; output is written under {BENCHMARK_NAME}/.",
    )
    parser.add_argument(
        "--music-policy",
        choices=MUSIC_POLICIES,
        default=DEFAULT_MUSIC_POLICY,
        help=(
            "exclude-category-music (default) removes only category == 'music' "
            f"and yields {PINNED_CATEGORY_ONLY_ROWS} rows; exclude-any-music also "
            "removes mixed top-level music domains and yields "
            f"{PINNED_EXCLUDE_ANY_MUSIC_ROWS} rows."
        ),
    )
    parser.add_argument(
        "--audio-prefix",
        default=f"/data/{BENCHMARK_NAME}/audio",
        help=f"Container-visible prefix corresponding to <data_dir>/{BENCHMARK_NAME}/audio.",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help=(
            "Do not download/extract data.zip. All selected files must already exist "
            "under the local audio directory because real durations are always measured."
        ),
    )
    args = parser.parse_args()

    if args.data_dir:
        output_root = Path(args.data_dir) / BENCHMARK_NAME
    else:
        package_root = Path(__file__).parent
        if "site-packages" in str(package_root) or "dist-packages" in str(package_root):
            raise SystemExit("Missing --data_dir and NEMO_SKILLS_DATA_DIR; refusing to write into site-packages.")
        output_root = package_root
    output_root.mkdir(parents=True, exist_ok=True)

    prepare(
        output_root=output_root,
        audio_prefix=args.audio_prefix,
        music_policy=args.music_policy,
        download_audio=not args.no_audio,
    )


if __name__ == "__main__":
    main()
