# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Prepare the CS-FLEURS code-switched ASR benchmark for NeMo Skills.

Downloads the requested CS-FLEURS test sets from HuggingFace (metadata + audio,
laid out as ``<subset>/<split>/metadata.jsonl`` + ``.../audio/<lang3>/<id>.wav``)
and writes one ``test.jsonl`` per sub-benchmark with OpenAI-format ASR records.

Each record carries ``task_type="Multilingual-ASR"`` plus ``extra_fields`` with
``use_cer`` (CER for scriptio-continua matrix languages, WER otherwise) and
``src_lang`` (matrix-language ISO 639-1 code) so the audio evaluator scores it
the same way as the plain ``fleurs`` benchmark. ``subset_for_metrics`` is the
code-switched language pair, giving a per-pair WER/CER breakdown.

Dataset: https://huggingface.co/datasets/byan/cs-fleurs

Usage:
    # Prepare all four test sets into the shared data dir
    ns prepare_data cs-fleurs --data_dir=/path/to/skills_data

    # Prepare a subset of the test sets / split (standalone)
    python -m nemo_skills.dataset.cs-fleurs.prepare \
        --data_dir=/path/to/skills_data --subsets read mms
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

# The package dir is "cs-fleurs" (hyphen), so the sibling languages module is not
# importable by dotted name. Load it by file path instead.
_lang_spec = importlib.util.spec_from_file_location("cs_fleurs_languages", Path(__file__).parent / "languages.py")
languages = importlib.util.module_from_spec(_lang_spec)
_lang_spec.loader.exec_module(languages)

HF_REPO_ID = "byan/cs-fleurs"
HF_REVISION = "0cdbf166c5517ae4b6eb1c54248522eedec53017"
SOURCE_CONTRACTS = {
    "read": {"rows": 5818, "sha256": "1ed381025ed51c368e591e263eace85a6e4e59a215d3f8888ca07c1e188fd709"},
    "mms": {"rows": 14932, "sha256": "48c065a79260058951ba2f25e58c865bea313874beb15fc07db7e72b6bc63f37"},
    "xtts-test1": {"rows": 11321, "sha256": "677893d3db633645aedbb0a1d838e046afbad53c9817f66b3e5977c24fd0065d"},
    "xtts-test2": {"rows": 17226, "sha256": "bdee4723f79c3485e03b5476ec90215c0b24dd3dfc285f65a09491d8f705b9c8"},
}

# Sub-benchmark name -> (HuggingFace top-level dir, split dir name).
SUBSETS: dict[str, tuple[str, str]] = {
    "read": ("read", "test"),
    "mms": ("mms", "test"),
    "xtts-test1": ("xtts", "test1"),
    "xtts-test2": ("xtts", "test2"),
}

ASR_INSTRUCTION = "Produce a verbatim transcript of the audio."


def _validate_metadata(metadata_path: Path, subset: str) -> list[dict]:
    """Load metadata only when it exactly matches the immutable source contract."""
    payload = metadata_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    expected = SOURCE_CONTRACTS[subset]
    if digest != expected["sha256"]:
        raise ValueError(f"CS-FLEURS {subset} metadata hash drift: expected {expected['sha256']}, found {digest}")

    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != expected["rows"]:
        raise ValueError(f"CS-FLEURS {subset} row-count drift: expected {expected['rows']}, found {len(rows)}")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"CS-FLEURS {subset} contains duplicate row ids")
    if len({row["file_name"] for row in rows}) != len(rows):
        raise ValueError(f"CS-FLEURS {subset} contains duplicate audio paths")
    return rows


def _missing_audio(split_dir: Path, rows: list[dict]) -> list[str]:
    return [row["file_name"] for row in rows if not (split_dir / row["file_name"]).is_file()]


def _download_subset(local_dir: Path, subset: str, hf_dir: str, split: str, no_audio: bool) -> Path:
    """Snapshot-download one CS-FLEURS subset; return its split directory on disk."""
    split_dir = local_dir / hf_dir / split
    metadata_path = split_dir / "metadata.jsonl"
    if metadata_path.exists():
        rows = _validate_metadata(metadata_path, subset)
        missing = [] if no_audio else _missing_audio(split_dir, rows)
        if not missing:
            print(f"  (already downloaded {len(rows)} pinned files, skipping HF snapshot)")
            return split_dir
        print(f"  (incomplete: {len(missing)}/{len(rows)} referenced audio files missing, re-downloading)")

    from huggingface_hub import snapshot_download

    patterns = [f"{hf_dir}/{split}/metadata.jsonl"]
    if not no_audio:
        patterns.append(f"{hf_dir}/{split}/audio/**")

    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=patterns,
        revision=HF_REVISION,
    )
    rows = _validate_metadata(metadata_path, subset)
    missing = [] if no_audio else _missing_audio(split_dir, rows)
    if missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(f"CS-FLEURS {subset} download is incomplete: {len(missing)} files missing ({preview})")

    return split_dir


def _container_audio_path(raw_subdir: str, file_name: str) -> str:
    """Path the audio resolves to at eval time.

    Rooted at ``NEMO_SKILLS_AUDIO_ROOT`` (default ``/data``) to match the repo's
    ``<data_dir>:/data`` mount and the other audio benchmarks (librispeech-pc,
    musan). For a local eval, set ``NEMO_SKILLS_AUDIO_ROOT`` to the host data dir
    so the paths resolve directly without a container mount. ``raw_subdir`` is
    ``<hf_dir>/<split>`` and ``file_name`` is ``audio/<lang3>/<id>.wav``.
    """
    audio_root = os.getenv("NEMO_SKILLS_AUDIO_ROOT", "/data")
    return f"{audio_root}/cs-fleurs/raw/{raw_subdir}/{file_name}"


def _build_record(row: dict, raw_subdir: str, subset: str) -> dict:
    language = row["language"]
    matrix, embedded = languages.split_pair(language)
    cpath = _container_audio_path(raw_subdir, row["file_name"])
    duration = float(row["duration"])
    audio_metadata = {"path": cpath, "duration": duration}

    extra_fields = {
        "src_text": row["text"],
        "src_lang": languages.get_iso1(matrix),  # ISO 639-1 for num2words / normalizer
        "matrix_lang": matrix,
        "matrix_lang_name": languages.get_lang_name(matrix),
        "embedded_lang": embedded,
        "embedded_lang_name": languages.get_lang_name(embedded) if embedded else "",
        "lang_pair": language,
        "use_cer": languages.uses_cer(matrix),
        # Score code-switched utterances with Mixed Error Rate: scriptio-continua
        # scripts counted by character, space-delimited scripts by word, in the
        # same utterance. use_mer supersedes use_cer in the audio evaluator. Use
        # grapheme-cluster segmentation when either side is a combining-mark
        # script (Thai/Lao/Myanmar/Khmer); harmless no-op for the other scripts.
        "use_mer": True,
        "mer_grapheme": languages.uses_grapheme(matrix) or (bool(embedded) and languages.uses_grapheme(embedded)),
        "speaker": row.get("speaker"),
    }
    if "fluency" in row:
        extra_fields["fluency"] = row["fluency"]

    return {
        "id": row["id"],
        "expected_answer": row["text"],
        "audio_path": cpath,
        "duration": duration,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. /no_think"},
            {"role": "user", "content": ASR_INSTRUCTION, "audio": audio_metadata},
        ],
        "subset_for_metrics": language,
        "task_type": "Multilingual-ASR",
        "extra_fields": extra_fields,
        "source_provenance": {
            "dataset": HF_REPO_ID,
            "dataset_revision": HF_REVISION,
            "metadata_sha256": SOURCE_CONTRACTS[subset]["sha256"],
            "expected_rows": SOURCE_CONTRACTS[subset]["rows"],
            "subset": subset,
        },
    }


def _default_data_dir() -> Path:
    """Resolve the CLI-managed immutable data root before falling back locally."""
    if data_root := os.getenv("NEMO_SKILLS_DATA_DIR"):
        return Path(data_root) / "cs-fleurs"
    return Path(__file__).parent


def _publish_records(out_path: Path, records: list[dict]) -> bool:
    """Atomically publish a deterministic manifest, or validate a completed one."""
    payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records).encode("utf-8")
    if out_path.exists():
        if not out_path.is_file() or out_path.read_bytes() != payload:
            raise ValueError(f"Existing CS-FLEURS manifest fails deterministic closure: {out_path}")
        return False

    partial = out_path.with_suffix(out_path.suffix + ".partial")
    with partial.open("wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    partial.replace(out_path)
    if out_path.read_bytes() != payload:
        raise ValueError(f"Published CS-FLEURS manifest failed byte revalidation: {out_path}")
    return True


def prepare_cs_fleurs(data_dir: Path, subsets: list[str], no_audio: bool) -> None:
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for subset in subsets:
        hf_dir, split = SUBSETS[subset]
        raw_subdir = f"{hf_dir}/{split}"
        print(f"\n=== {subset} (hf: {raw_subdir}) ===")
        split_dir = _download_subset(raw_dir, subset, hf_dir, split, no_audio)
        rows = _validate_metadata(split_dir / "metadata.jsonl", subset)

        records = [_build_record(row, raw_subdir, subset) for row in rows]

        out_dir = data_dir / subset
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "test.jsonl"
        published = _publish_records(out_path, records)

        n_cer = sum(1 for r in records if r["extra_fields"]["use_cer"])
        action = "wrote" if published else "revalidated existing"
        print(f"  {action} {len(records)} records -> {out_path} ({n_cer} CER, {len(records) - n_cer} WER)")


def main():
    parser = argparse.ArgumentParser(description="Prepare CS-FLEURS code-switched ASR benchmark")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Output directory (a 'cs-fleurs' subdir is created; defaults to this package directory)",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=list(SUBSETS),
        choices=list(SUBSETS),
        help="Which CS-FLEURS test sets to prepare",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip downloading audio (only build manifests; not usable for actual evaluation)",
    )
    args = parser.parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir) / "cs-fleurs"
    else:
        data_dir = _default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    prepare_cs_fleurs(data_dir=data_dir, subsets=args.subsets, no_audio=args.no_audio)


if __name__ == "__main__":
    main()
