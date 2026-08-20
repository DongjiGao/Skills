#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Build merged ``{group}/test.jsonl`` for multispeaker-sot benchmark groups.

Each group module (e.g. ``multispeaker-sot/under20s/__init__.py``) declares a
``SUBSETS`` list of member eval-set directory names. This script concatenates
each member's already-prepared ``<data_dir>/multispeaker-sot/<set>/test.jsonl``
into ``<data_dir>/multispeaker-sot/<group>/test.jsonl`` so the group can be run
as a *single* benchmark (one vLLM job + one summarize job).

Group definitions are read from the dataset package next to this script, since
``prepare.py`` writes only ``<set>/test.jsonl`` into the data dir and leaves no
``__init__.py`` there to discover. Canonical package definitions take precedence
over stale copies in the data dir; data-local modules can still add custom group
names that do not exist in the package.

Records are copied verbatim, so each keeps its ``subset_for_metrics`` tag and
audio_filepath — the merge is a pure concatenation (no audio is touched). This
is intentionally stdlib-only so it runs on a cluster login node.

Usage:
    python3 build_group_manifests.py --data-dir /path/to/skills_data
    python3 build_group_manifests.py --data-dir /path/to/skills_data --groups under20s,lsm
    python3 build_group_manifests.py --data-dir /path/to/skills_data --groups full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BENCHMARK_NAME = "multispeaker-sot"


def _read_subsets(group_init: Path) -> list[str]:
    """Return the SUBSETS list declared in a group ``__init__.py``.

    The group modules contain only literal assignments, so we exec the file in
    an isolated namespace rather than importing (the package name contains a
    hyphen and is not importable as a normal module).
    """
    namespace: dict = {}
    try:
        exec(compile(group_init.read_text(encoding="utf-8"), str(group_init), "exec"), namespace)
    except Exception as exc:  # noqa: BLE001 - report and skip malformed modules
        print(f"[build-group] WARN: could not parse {group_init}: {exc}", file=sys.stderr)
        return []
    subsets = namespace.get("SUBSETS")
    if not isinstance(subsets, list):
        return []
    return [str(s) for s in subsets]


def _discover_groups(*roots: Path) -> dict[str, list[str]]:
    """Find group dirs (``<root>/<group>/__init__.py`` defining SUBSETS).

    Roots are searched in order and the first definition of a group wins.
    """
    groups: dict[str, list[str]] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for init_path in sorted(root.glob("*/__init__.py")):
            group = init_path.parent.name
            if group in groups:
                continue
            subsets = _read_subsets(init_path)
            if subsets:
                groups[group] = subsets
    return groups


def _merge_group(root: Path, group: str, subsets: list[str], audio_rewrite: tuple[str, str] | None = None) -> int:
    group_dir = root / group
    group_dir.mkdir(parents=True, exist_ok=True)
    out_path = group_dir / "test.jsonl"

    lines_written = 0
    missing: list[str] = []
    tmp_path = out_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as fout:
        for subset in subsets:
            src = root / subset / "test.jsonl"
            if not src.exists():
                missing.append(subset)
                continue
            with src.open("r", encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        if not line.endswith("\n"):
                            line += "\n"
                        # Rewrite the prep-time audio mount prefix (e.g. "/data/
                        # multispeaker-sot/") to the real data-dir path so the eval
                        # job -- which does NOT mount /data -- can open the wavs. This
                        # is a plain substring replace on the audio subtree only, so
                        # source_audio_filepath ("/data/multi_speaker_asr/...") is left
                        # untouched.
                        if audio_rewrite is not None:
                            line = line.replace(audio_rewrite[0], audio_rewrite[1])
                        fout.write(line)
                        lines_written += 1

    if missing:
        tmp_path.unlink(missing_ok=True)
        raise FileNotFoundError(
            f"group {group!r}: missing member test.jsonl for {missing} under {root}. "
            "Prepare/copy those eval sets first."
        )

    tmp_path.replace(out_path)
    print(f"[build-group] {BENCHMARK_NAME}.{group}: merged {len(subsets)} sets -> {out_path} ({lines_written} rows)")
    return lines_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Skills data root that contains the multispeaker-sot/ tree.",
    )
    parser.add_argument(
        "--groups",
        default="",
        help="Comma-separated group names to build (default: all discovered groups).",
    )
    parser.add_argument(
        "--audio-src-prefix",
        default="/data",
        help=(
            "Prep-time audio mount prefix to rewrite to the real --data-dir (default "
            "'/data'). Only the '<prefix>/multispeaker-sot/' audio subtree is rewritten; "
            "pass '' to disable rewriting."
        ),
    )
    args = parser.parse_args()

    data_root = Path(args.data_dir).expanduser()
    root = data_root / BENCHMARK_NAME
    if not root.is_dir():
        raise SystemExit(f"[build-group] ERROR: {root} does not exist")

    audio_rewrite: tuple[str, str] | None = None
    if args.audio_src_prefix:
        old_prefix = f"{args.audio_src_prefix.rstrip('/')}/{BENCHMARK_NAME}/"
        new_prefix = f"{str(data_root).rstrip('/')}/{BENCHMARK_NAME}/"
        if old_prefix != new_prefix:
            audio_rewrite = (old_prefix, new_prefix)
            print(f"[build-group] audio path rewrite: {old_prefix!r} -> {new_prefix!r}")

    package_root = Path(__file__).resolve().parent
    discovered = _discover_groups(package_root, root)
    if not discovered:
        raise SystemExit(
            f"[build-group] ERROR: no group modules with SUBSETS found under {root} or {package_root}"
        )

    requested = [g.strip() for g in args.groups.split(",") if g.strip()]
    if requested:
        unknown = [g for g in requested if g not in discovered]
        if unknown:
            raise SystemExit(
                f"[build-group] ERROR: unknown group(s) {unknown}; known: {sorted(discovered)}"
            )
        selected = {g: discovered[g] for g in requested}
    else:
        selected = discovered

    for group, subsets in selected.items():
        _merge_group(root, group, subsets, audio_rewrite=audio_rewrite)


if __name__ == "__main__":
    main()
