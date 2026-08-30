# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare LibriSpeech-PC for ASR evaluation with punctuation and capitalization.

LibriSpeech-PC provides manifests with punctuation/capitalization from OpenSLR-145.
Audio files are downloaded from original LibriSpeech at OpenSLR-12.

Usage:
    ns prepare_data librispeech-pc --data_dir <path_to_data_dir>
    ns prepare_data librispeech-pc --split test-clean (or test-other) --data_dir <path_to_data_dir>
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

from tqdm import tqdm


def download_with_progress(url: str, output_path: Path, desc: str):
    """Download file with tqdm progress bar."""
    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, desc=desc) as pbar:

        def reporthook(blocknum, blocksize, totalsize):
            if pbar.total != totalsize:
                pbar.total = totalsize
            downloaded = blocknum * blocksize
            pbar.update(max(0, downloaded - pbar.n))

        urllib.request.urlretrieve(url, output_path, reporthook)


# LibriSpeech-PC manifests (with punctuation and capitalization)
MANIFESTS_URL = "https://www.openslr.org/resources/145/manifests.tar.gz"

MANIFESTS_SHA256 = "96d4eae2222b29b66437a21959252419bcd4762e5042e71e023790171054d1c0"
MANIFESTS_BYTES = 25_892_530
MANIFEST_FIELDS = frozenset({"audio_filepath", "duration", "text", "text_raw"})
MANIFEST_CONTRACTS = {
    "test-clean": {
        "rows": 2_417,
        "sha256": "ed0d410566ccce2cc9aa9c1198d064a1aeca358b7f8374011db38111d1738a8e",
    },
    "test-other": {
        "rows": 2_856,
        "sha256": "c388f7d91d3a192398facdd06ece0f45da9c3148ecf33c94b0207f918bab7281",
    },
}
AUDIO_CONTRACTS = {
    "test-clean": {
        "url": "https://www.openslr.org/resources/12/test-clean.tar.gz",
        "md5": "32fa31d27d2e1cad72775fee3f4849a9",
        "bytes": 346_663_984,
    },
    "test-other": {
        "url": "https://www.openslr.org/resources/12/test-other.tar.gz",
        "md5": "fb5a50374b501bb3bac4815ee91d3135",
        "bytes": 328_757_843,
    },
}
FLAC_MAGIC = b"fLaC"


def _file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_source_file(path: Path, *, algorithm: str, expected_digest: str, expected_bytes: int) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(f"Source size mismatch for {path}: expected={expected_bytes}, actual={actual_bytes}")
    actual_digest = _file_digest(path, algorithm)
    if actual_digest != expected_digest:
        raise ValueError(f"Source {algorithm} mismatch for {path}: expected={expected_digest}, actual={actual_digest}")
    return path


def _download_verified(
    url: str,
    destination: Path,
    *,
    algorithm: str,
    expected_digest: str,
    expected_bytes: int,
    desc: str,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return _validate_source_file(
            destination,
            algorithm=algorithm,
            expected_digest=expected_digest,
            expected_bytes=expected_bytes,
        )
    partial = destination.with_suffix(destination.suffix + ".partial")
    download_with_progress(url, partial, desc)
    _validate_source_file(
        partial,
        algorithm=algorithm,
        expected_digest=expected_digest,
        expected_bytes=expected_bytes,
    )
    partial.replace(destination)
    return destination


def _manifest_audio_parts(audio_filepath: str, split: str) -> tuple[str, str, str, str]:
    if not isinstance(audio_filepath, str):
        raise ValueError(f"Unexpected {split} audio path: {audio_filepath!r}")
    path = PurePosixPath(audio_filepath)
    if path.is_absolute() or ".." in path.parts or "." in path.parts or len(path.parts) != 4:
        raise ValueError(f"Unexpected {split} audio path: {audio_filepath!r}")
    if path.parts[0] != split or path.suffix != ".flac":
        raise ValueError(f"Unexpected {split} audio path: {audio_filepath!r}")
    speaker, chapter, filename = path.parts[1:]
    stem_parts = Path(filename).stem.split("-")
    if len(stem_parts) != 3 or stem_parts[:2] != [speaker, chapter] or not all(part.isdigit() for part in stem_parts):
        raise ValueError(f"Unexpected LibriSpeech utterance identity: {audio_filepath!r}")
    return path.parts


def _load_manifest(path: Path, split: str) -> list[dict]:
    contract = MANIFEST_CONTRACTS[split]
    actual_digest = _file_digest(path, "sha256")
    if actual_digest != contract["sha256"]:
        raise ValueError(
            f"LibriSpeech-PC {split} manifest hash drift: expected={contract['sha256']}, actual={actual_digest}"
        )
    entries = []
    identities: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"Blank line in {path}:{line_number}")
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {error}") from error
            if not isinstance(entry, dict) or set(entry) != MANIFEST_FIELDS:
                actual_fields = sorted(entry) if isinstance(entry, dict) else type(entry).__name__
                raise ValueError(
                    f"Unexpected manifest fields in {path}:{line_number}: "
                    f"expected={sorted(MANIFEST_FIELDS)}, actual={actual_fields}"
                )
            parts = _manifest_audio_parts(entry["audio_filepath"], split)
            identity = Path(parts[-1]).stem
            if identity in identities:
                raise ValueError(f"Duplicate utterance ID {identity!r} in {path}")
            identities.add(identity)
            if isinstance(entry["duration"], bool) or not isinstance(entry["duration"], (int, float)):
                raise ValueError(f"Invalid duration in {path}:{line_number}: {entry['duration']!r}")
            if entry["duration"] <= 0:
                raise ValueError(f"Non-positive duration in {path}:{line_number}: {entry['duration']!r}")
            for field in ("text", "text_raw"):
                if not isinstance(entry[field], str) or not entry[field].strip():
                    raise ValueError(f"Missing {field} in {path}:{line_number}")
            entries.append(entry)
    if len(entries) != contract["rows"]:
        raise ValueError(f"LibriSpeech-PC {split} row-count drift: expected={contract['rows']}, actual={len(entries)}")
    return entries


def _safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in tar.getmembers():
        target = (destination / member.name).resolve()
        if root not in target.parents and target != root:
            raise ValueError(f"Unsafe archive member: {member.name!r}")
    if sys.version_info >= (3, 11, 4):
        tar.extractall(destination, filter="data")
    else:
        tar.extractall(destination)


def download_manifests(output_dir: Path) -> Path:
    """Materialize the exact official OpenSLR-145 test manifests."""
    manifest_paths = {split: output_dir / f"{split}.json" for split in MANIFEST_CONTRACTS}
    if all(path.is_file() for path in manifest_paths.values()):
        for split, path in manifest_paths.items():
            _load_manifest(path, split)
        return output_dir

    tar_path = _download_verified(
        MANIFESTS_URL,
        output_dir / "sources" / "manifests.tar.gz",
        algorithm="sha256",
        expected_digest=MANIFESTS_SHA256,
        expected_bytes=MANIFESTS_BYTES,
        desc="Downloading LibriSpeech-PC manifests",
    )
    with tarfile.open(tar_path, "r:gz") as tar:
        wanted = set(path.name for path in manifest_paths.values())
        members: dict[str, tarfile.TarInfo] = {}
        for member in tar.getmembers():
            archive_path = PurePosixPath(member.name)
            name = archive_path.name
            if name not in wanted:
                continue
            if archive_path.is_absolute() or ".." in archive_path.parts or not member.isfile():
                raise ValueError(f"Unsafe or invalid manifest archive member: {member.name!r}")
            if name in members:
                raise ValueError(f"Duplicate manifest archive member: {name}")
            members[name] = member
        missing = wanted - set(members)
        if missing:
            raise ValueError(f"Official manifest archive is missing: {sorted(missing)}")
        for split, out_path in manifest_paths.items():
            if out_path.exists():
                _load_manifest(out_path, split)
                continue
            fobj = tar.extractfile(members[out_path.name])
            if fobj is None:
                raise ValueError(f"Cannot read official manifest member: {out_path.name}")
            partial = out_path.with_suffix(out_path.suffix + ".partial")
            with partial.open("wb") as output:
                shutil.copyfileobj(fobj, output)
            _load_manifest(partial, split)
            partial.replace(out_path)

    for split, path in manifest_paths.items():
        _load_manifest(path, split)
    return output_dir


def download_audio(split: str, audio_dir: Path) -> Path:
    """Download and verify the exact OpenSLR-12 audio source archive."""
    contract = AUDIO_CONTRACTS[split]
    tar_path = _download_verified(
        contract["url"],
        audio_dir / "sources" / f"{split}.tar.gz",
        algorithm="md5",
        expected_digest=contract["md5"],
        expected_bytes=contract["bytes"],
        desc=f"Downloading LibriSpeech {split}",
    )
    split_dir = audio_dir / "LibriSpeech" / split
    if split_dir.is_dir():
        return split_dir
    if split_dir.exists():
        raise ValueError(f"LibriSpeech split sentinel is not a directory: {split_dir}")
    with tarfile.open(tar_path, "r:gz") as tar:
        _safe_extract(tar, audio_dir)
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Official archive did not create expected split sentinel: {split_dir}")
    return split_dir


def _validate_audio_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing LibriSpeech-PC audio: {path}")
    with path.open("rb") as source:
        header = source.read(len(FLAC_MAGIC))
    if header != FLAC_MAGIC:
        raise ValueError(f"Invalid FLAC header for {path}: {header!r}")


def process_split(split: str, data_dir: Path, audio_dir: Path, with_audio: bool) -> int:
    """Process one exact LibriSpeech-PC split into NeMo Skills format."""
    output_file = data_dir / split / "test.jsonl"
    legacy_output_file = data_dir / f"{split}.jsonl"
    manifest_file = data_dir / f"{split}.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")
    entries = _load_manifest(manifest_file, split)
    if with_audio:
        download_audio(split, audio_dir)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    audio_root = os.getenv("NEMO_SKILLS_AUDIO_ROOT", "/data").rstrip("/")
    output_entries = []
    for entry in entries:
        rel_audio_path = entry["audio_filepath"]
        parts = _manifest_audio_parts(rel_audio_path, split)
        local_audio = audio_dir / "LibriSpeech" / Path(*parts)
        _validate_audio_file(local_audio)
        audio_id = Path(parts[-1]).stem
        container_path = f"{audio_root}/librispeech-pc/LibriSpeech/{rel_audio_path}"
        duration = float(entry["duration"])
        audio = {"path": container_path, "duration": duration}
        output_entries.append(
            {
                "id": audio_id,
                "audio_filepath": container_path,
                "duration": duration,
                "text": entry["text"],
                "expected_answer": entry["text"],
                "task_type": "ASR-PC",
                "sample_id": audio_id,
                "split": split,
                "subset_for_metrics": split,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant. /no_think"},
                    {
                        "role": "user",
                        "content": "Transcribe the audio with proper punctuation and capitalization.",
                        "audio": audio,
                    },
                ],
                "source_provenance": {
                    "manifest_url": MANIFESTS_URL,
                    "manifest_sha256": MANIFEST_CONTRACTS[split]["sha256"],
                    "manifest_rows": MANIFEST_CONTRACTS[split]["rows"],
                    "audio_url": AUDIO_CONTRACTS[split]["url"],
                    "audio_md5": AUDIO_CONTRACTS[split]["md5"],
                },
            }
        )

    partial = output_file.with_suffix(output_file.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as output:
        for output_entry in output_entries:
            output.write(json.dumps(output_entry, ensure_ascii=False) + "\n")
    partial.replace(output_file)
    legacy_partial = legacy_output_file.with_suffix(legacy_output_file.suffix + ".partial")
    shutil.copyfile(output_file, legacy_partial)
    legacy_partial.replace(legacy_output_file)
    print(f"✓ {split}: {len(output_entries)} verified samples")
    return len(output_entries)


def main():
    parser = argparse.ArgumentParser(description="Prepare LibriSpeech-PC for ASR evaluation")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.getenv("NEMO_SKILLS_DATA_DIR"),
        help=(
            "Base data dir (defaults to $NEMO_SKILLS_DATA_DIR). "
            "If provided, output goes under <data_dir>/librispeech-pc. "
            "If omitted, writes into this package's dataset directory (only allowed outside site-packages)."
        ),
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", "test-clean", "test-other"],
        help="Which split to prepare (default: all)",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Do not download audio; require a complete pre-existing LibriSpeech tree instead",
    )
    args = parser.parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir) / "librispeech-pc"
    else:
        pkg_dir = Path(__file__).parent
        pkg_dir_str = str(pkg_dir)
        if "site-packages" in pkg_dir_str or "dist-packages" in pkg_dir_str:
            raise SystemExit(
                "Missing --data_dir and NEMO_SKILLS_DATA_DIR is not set. "
                "Refusing to write into the installed package directory; please set NEMO_SKILLS_DATA_DIR "
                "or pass --data_dir."
            )
        data_dir = pkg_dir

    audio_dir = data_dir
    audio_dir.mkdir(parents=True, exist_ok=True)

    download_manifests(data_dir)

    splits = ["test-clean", "test-other"] if args.split == "all" else [args.split]
    total = sum(process_split(split, data_dir, audio_dir, not args.no_audio) for split in splits)

    print(f"\n✓ Complete: {total} samples")


if __name__ == "__main__":
    main()
