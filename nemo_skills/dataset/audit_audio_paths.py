# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Heavyweight audit for prepared audio benchmark JSONLs.

Walks every ``*.jsonl`` under a prepared host ``--data-dir`` tree, finds every
audio path string in every record (any field, any depth), substitutes the
in-container ``--audio-prefix`` for the host root, and verifies the resulting
host path exists. Also flags any string containing a stale ``/dataset/``
substring -- a leftover from manifests written before the path unification.

Companion to the lightweight unit tests in ``tests/test_audio_path_prefix.py``.
The unit tests catch helper/wrapper regressions in CI without network. This
tool runs against real prepared data on /lustre to catch issues unit tests
cannot see (host filesystem layout drift, stale cached manifests, missing
audio files).

Usage::

    python -m nemo_skills.dataset.audit_audio_paths \\
        --data-dir /lustre/.../skills_data \\
        --audio-prefix /data

Exit code:
    0 -- all audio paths resolve and no stale tokens.
    1 -- one or more issues found (see report).
    2 -- bad invocation (missing or invalid arguments).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

AUDIO_EXTS: tuple[str, ...] = (".wav", ".flac", ".mp3", ".ogg", ".opus")
STALE_TOKEN: str = "/dataset/"


@dataclass
class AuditStats:
    """Per-file audit summary."""

    rows: int = 0
    audio_paths: int = 0
    missing: list[tuple[int, str, str]] = field(default_factory=list)
    stale: list[tuple[int, str]] = field(default_factory=list)
    wrong_prefix: list[tuple[int, str]] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return bool(self.missing or self.stale or self.wrong_prefix)


def walk_strings(value) -> Iterator[str]:
    """Yield every string leaf in a nested JSON-style value."""
    if isinstance(value, dict):
        for v in value.values():
            yield from walk_strings(v)
    elif isinstance(value, list):
        for item in value:
            yield from walk_strings(item)
    elif isinstance(value, str):
        yield value


def looks_like_audio_path(s: str) -> bool:
    """Return True if ``s`` ends with a known audio file extension."""
    return s.endswith(AUDIO_EXTS)


def audit_jsonl(jsonl_path: Path, audio_prefix: str, data_dir: Path) -> AuditStats:
    """Audit one JSONL file. Returns per-file ``AuditStats``."""
    audio_prefix = audio_prefix.rstrip("/")
    stats = AuditStats()

    with jsonl_path.open(encoding="utf-8") as f:
        for line_num, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            stats.rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            for s in walk_strings(row):
                if STALE_TOKEN in s:
                    stats.stale.append((line_num, s))
                if not looks_like_audio_path(s):
                    continue
                stats.audio_paths += 1
                if not s.startswith(audio_prefix + "/"):
                    stats.wrong_prefix.append((line_num, s))
                    continue
                rel = s[len(audio_prefix) :].lstrip("/")
                host_path = data_dir / rel
                if not host_path.exists():
                    stats.missing.append((line_num, s, str(host_path)))
    return stats


def _format_per_file_report(rel: Path, stats: AuditStats, max_show: int) -> Iterator[str]:
    marker = "ISSUE" if stats.has_issues else "OK"
    yield (
        f"[{marker}] {rel} -- rows={stats.rows}, audio={stats.audio_paths}, "
        f"missing={len(stats.missing)}, stale={len(stats.stale)}, "
        f"wrong_prefix={len(stats.wrong_prefix)}"
    )
    for line_num, path in stats.stale[:max_show]:
        yield f"    stale line {line_num}: {path}"
    if len(stats.stale) > max_show:
        yield f"    ... and {len(stats.stale) - max_show} more stale entries"
    for line_num, path, host_path in stats.missing[:max_show]:
        yield f"    miss  line {line_num}: {path} -> {host_path}"
    if len(stats.missing) > max_show:
        yield f"    ... and {len(stats.missing) - max_show} more missing files"
    for line_num, path in stats.wrong_prefix[:max_show]:
        yield f"    pref  line {line_num}: {path}"
    if len(stats.wrong_prefix) > max_show:
        yield f"    ... and {len(stats.wrong_prefix) - max_show} more wrong-prefix entries"


def audit_data_dir(data_dir: Path, audio_prefix: str, max_show: int = 10, include_empty: bool = False) -> int:
    """Audit every JSONL under ``data_dir``. Prints a report and returns an exit code."""
    if not data_dir.is_dir():
        print(f"ERROR: --data-dir {data_dir} is not a directory", file=sys.stderr)
        return 2

    jsonl_files = sorted(data_dir.rglob("*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found under {data_dir}")
        return 0

    print(f"Auditing {len(jsonl_files)} JSONL files under {data_dir}")
    print(f"Substitution: {audio_prefix} -> {data_dir}")
    print("-" * 72)

    totals: defaultdict[str, int] = defaultdict(int)
    any_issues = False
    for jsonl_path in jsonl_files:
        stats = audit_jsonl(jsonl_path, audio_prefix, data_dir)
        if not include_empty and stats.audio_paths == 0 and not stats.stale:
            continue
        totals["rows"] += stats.rows
        totals["audio_paths"] += stats.audio_paths
        totals["missing"] += len(stats.missing)
        totals["stale"] += len(stats.stale)
        totals["wrong_prefix"] += len(stats.wrong_prefix)
        any_issues = any_issues or stats.has_issues
        for line in _format_per_file_report(jsonl_path.relative_to(data_dir), stats, max_show):
            print(line)

    print("-" * 72)
    print(
        f"TOTAL: rows={totals['rows']}, audio_paths={totals['audio_paths']}, "
        f"missing={totals['missing']}, stale={totals['stale']}, "
        f"wrong_prefix={totals['wrong_prefix']}"
    )
    return 1 if any_issues else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit prepared audio benchmark JSONLs for path correctness and missing files",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Host root that mirrors the in-container audio prefix "
        "(e.g., /lustre/.../skills_data). The root is what gets mounted at --audio-prefix at eval time.",
    )
    parser.add_argument(
        "--audio-prefix",
        type=str,
        default="/data",
        help="In-container audio root substituted for --data-dir when checking host files (default: /data).",
    )
    parser.add_argument(
        "--max-show",
        type=int,
        default=10,
        help="Max issues to show per category per file (default: 10).",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Show files with no audio paths in the per-file summary.",
    )
    args = parser.parse_args(argv)
    return audit_data_dir(
        data_dir=args.data_dir,
        audio_prefix=args.audio_prefix,
        max_show=args.max_show,
        include_empty=args.include_empty,
    )


if __name__ == "__main__":
    sys.exit(main())
