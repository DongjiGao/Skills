#!/usr/bin/env python3

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict, deque
from itertools import combinations
from pathlib import Path
from typing import Any

ID_KEYS = (
    "id",
    "uid",
    "uniq_id",
    "audio_id",
    "sample_id",
    "question_id",
    "task_id",
    "prompt_id",
    "index",
)
IDENTITY_DISCRIMINATOR_PRIORITY = (
    "instruction",
    "task_type",
    "subset_for_metrics",
    "direction",
    "preference_type",
)
MUTABLE_SOURCE_KEYS = {"text"}


def unescape_shell_command(command: str) -> list[str]:
    """Unescape special shell characters so they are correctly interpreted before execution."""
    command = command.strip() if command else ""
    return shlex.split(command)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL row in {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL row in {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise TypeError(f"Non-object JSONL row in {path}:{line_number}")
            rows.append(row)
    return rows


def _identity(row: dict[str, Any], identity_keys: tuple[str, ...], location: str) -> str:
    missing = [key for key in identity_keys if key not in row]
    if missing:
        raise ValueError(f"Row in {location} is missing stable ID fields: {missing}")
    return _canonical_json([row[key] for key in identity_keys])


def _select_identity_keys(source_rows: list[dict[str, Any]]) -> tuple[str, ...]:
    if not source_rows:
        return ()

    base_key = next((key for key in ID_KEYS if all(key in row for row in source_rows)), None)
    if base_key is None:
        raise ValueError(f"Source rows do not share a stable ID field; tried {ID_KEYS}")

    common_keys = set.intersection(*(set(row) - MUTABLE_SOURCE_KEYS for row in source_rows))
    candidates = tuple(
        dict.fromkeys(
            key
            for key in (*IDENTITY_DISCRIMINATOR_PRIORITY, *sorted(common_keys))
            if key != base_key and key in common_keys
        )
    )
    for extra_count in range(min(2, len(candidates)) + 1):
        for extras in combinations(candidates, extra_count):
            identity_keys = (base_key, *extras)
            identities = [_identity(row, identity_keys, "source") for row in source_rows]
            if len(set(identities)) == len(identities):
                return identity_keys

    raise ValueError(f"Source has duplicate stable IDs for {base_key} and no unique composite identity")


def _has_named_identity(source_rows: list[dict[str, Any]]) -> bool:
    return bool(source_rows) and any(all(key in row for row in source_rows) for key in ID_KEYS)


def _source_projection_keys(source_rows: list[dict[str, Any]]) -> tuple[str, ...]:
    if not source_rows:
        return ()
    common_keys = set.intersection(*(set(row) - MUTABLE_SOURCE_KEYS for row in source_rows))
    if not common_keys:
        raise ValueError("Source rows have no common immutable identity fields")
    return tuple(sorted(set.union(*(set(row) - MUTABLE_SOURCE_KEYS for row in source_rows))))


def _source_projection(row: dict[str, Any], source_keys: tuple[str, ...]) -> str:
    # Include key names as well as values so optional source fields remain part
    # of the identity. Keys added by generation are outside source_keys and are
    # intentionally ignored.
    return _canonical_json({key: row[key] for key in source_keys if key in row})


def _validate_projection_inventory(
    source_projections: list[str],
    output_projections: list[str],
) -> None:
    source_counts = Counter(source_projections)
    output_counts = Counter(output_projections)
    if source_counts != output_counts:
        missing = source_counts - output_counts
        extra = output_counts - source_counts
        raise ValueError(
            "Chunk source projections do not match source: "
            f"missing={sum(missing.values())} extra={sum(extra.values())} "
            f"missing_examples={list(missing.items())[:5]} "
            f"extra_examples={list(extra.items())[:5]}"
        )


def _ordered_rows_by_projection(
    source_rows: list[dict[str, Any]],
    output_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_keys = _source_projection_keys(source_rows)
    source_projections = [_source_projection(row, source_keys) for row in source_rows]
    output_projections = [_source_projection(row, source_keys) for row in output_rows]
    _validate_projection_inventory(source_projections, output_projections)

    rows_by_projection: defaultdict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for projection, row in zip(output_projections, output_rows):
        rows_by_projection[projection].append(row)
    return [rows_by_projection[projection].popleft() for projection in source_projections]


def _validate_identity_inventory(
    source_identities: list[str],
    output_identities: list[str],
    identity_keys: tuple[str, ...],
) -> None:
    if len(set(output_identities)) != len(output_identities):
        raise ValueError(f"Duplicate stable ID in chunks for {identity_keys}")

    source_identity_set = set(source_identities)
    output_identity_set = set(output_identities)
    missing = [identity for identity in source_identities if identity not in output_identity_set]
    extra = sorted(output_identity_set - source_identity_set)
    if missing or extra:
        raise ValueError(
            f"Chunk stable IDs do not match source for {identity_keys}: "
            f"missing={len(missing)} extra={len(extra)} "
            f"missing_examples={missing[:5]} extra_examples={extra[:5]}"
        )


def _ordered_rows(source_file: str, input_files: list[str]) -> list[dict[str, Any]]:
    source_rows = _load_jsonl(source_file)
    output_rows = [row for input_file in input_files for row in _load_jsonl(input_file)]
    if len(source_rows) != len(output_rows):
        raise ValueError(
            f"Chunk row count does not match source: source={len(source_rows)} chunks={len(output_rows)}"
        )
    if not source_rows:
        return []
    if not _has_named_identity(source_rows):
        return _ordered_rows_by_projection(source_rows, output_rows)

    identity_keys = _select_identity_keys(source_rows)
    source_identities = [_identity(row, identity_keys, source_file) for row in source_rows]
    rows_by_identity: dict[str, dict[str, Any]] = {}
    output_identities: list[str] = []
    for row in output_rows:
        identity = _identity(row, identity_keys, "chunks")
        output_identities.append(identity)
        rows_by_identity.setdefault(identity, row)

    _validate_identity_inventory(source_identities, output_identities, identity_keys)
    return [rows_by_identity[identity] for identity in source_identities]


def _validate_published_output(output_file: str, source_file: str) -> None:
    source_rows = _load_jsonl(source_file)
    output_rows = _load_jsonl(output_file)
    if len(source_rows) != len(output_rows):
        raise ValueError(
            f"Published output row count does not match source: "
            f"source={len(source_rows)} output={len(output_rows)}"
        )
    if not source_rows:
        return
    if not _has_named_identity(source_rows):
        source_keys = _source_projection_keys(source_rows)
        source_projections = [_source_projection(row, source_keys) for row in source_rows]
        output_projections = [_source_projection(row, source_keys) for row in output_rows]
        _validate_projection_inventory(source_projections, output_projections)
        if output_projections != source_projections:
            raise ValueError("Published output source projections are not in source order")
        return

    identity_keys = _select_identity_keys(source_rows)
    source_identities = [_identity(row, identity_keys, source_file) for row in source_rows]
    output_identities = [_identity(row, identity_keys, output_file) for row in output_rows]
    _validate_identity_inventory(source_identities, output_identities, identity_keys)
    if output_identities != source_identities:
        raise ValueError(f"Published output stable IDs are not in source order for {identity_keys}")


def _write_output(output_file: str, input_files: list[str], source_file: str | None) -> None:
    output_dir = os.path.dirname(output_file) or "."
    tmp_output_file: str | None = None
    try:
        mode = "wb" if source_file is None else "w"
        kwargs = {} if source_file is None else {"encoding": "utf-8"}
        with tempfile.NamedTemporaryFile(
            mode,
            dir=output_dir,
            prefix=f".{os.path.basename(output_file)}.",
            delete=False,
            **kwargs,
        ) as output_stream:
            tmp_output_file = output_stream.name
            if source_file is None:
                for input_file in input_files:
                    with open(input_file, "rb") as input_stream:
                        shutil.copyfileobj(input_stream, output_stream)
            else:
                for row in _ordered_rows(source_file, input_files):
                    output_stream.write(_canonical_json(row) + "\n")
            output_stream.flush()
            os.fsync(output_stream.fileno())
        os.replace(tmp_output_file, output_file)
    except Exception:
        if tmp_output_file is not None:
            try:
                os.remove(tmp_output_file)
            except OSError:
                pass
        raise


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str] | None]:
    if "--" in argv:
        sep_index = argv.index("--")
        file_args = argv[:sep_index]
        post_merge_command = unescape_shell_command(" ".join(argv[sep_index + 1 :]))
    else:
        file_args = argv
        post_merge_command = None

    parser = argparse.ArgumentParser()
    parser.add_argument("output_file")
    parser.add_argument("input_files", nargs="+")
    parser.add_argument("--source-file")
    return parser.parse_args(file_args), post_merge_command


def main(argv: list[str] | None = None) -> int:
    args, post_merge_command = _parse_args(sys.argv[1:] if argv is None else argv)
    output_done_file = f"{args.output_file}.done"
    if os.path.isfile(output_done_file) and os.path.isfile(args.output_file):
        if args.source_file:
            _validate_published_output(args.output_file, args.source_file)
        print(f"Info: {args.output_file} is already merged. Exiting.")
        return 0

    # A chunk may be empty when num_chunks exceeds the number of examples. In
    # that case Skills creates its .done marker but no JSONL file.
    existing_input_files = []
    for input_file in args.input_files:
        done_file = f"{input_file}.done"
        if not os.path.isfile(done_file):
            print(f"Info: {done_file} not found. Skipping the rest of the script.")
            return 0
        if os.path.isfile(input_file):
            existing_input_files.append(input_file)
        else:
            print(f"Info: {input_file} not found but {done_file} exists; treating as an empty chunk.")

    _write_output(args.output_file, existing_input_files, args.source_file)
    action = "ordered" if args.source_file else "concatenated"
    print(
        f"Successfully {action} {len(existing_input_files)} non-empty files "
        f"from {len(args.input_files)} chunks to {args.output_file}"
    )

    Path(output_done_file).touch()
    if post_merge_command:
        print(f"Executing post-merge command: {' '.join(post_merge_command)}")
        subprocess.run(" ".join(post_merge_command), shell=True, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
