#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0

"""Convert Turing bracketed NSV labels to Nemotron Transcribe event tokens.

The converter updates both ``text`` and ``expected_answer`` so the Skills
reference contract stays consistent. It writes UTF-8 JSONL atomically and never
emits escaped Unicode text. Unknown bracketed annotations are rejected instead
of being silently assigned to an unrelated event class.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


NEMOTRON_NSV_TOKENS = (
    "<ev:breath>",
    "<ev:crying>",
    "<ev:effort_strain>",
    "<ev:gesture_noise>",
    "<ev:laughter>",
    "<ev:mouth_noise>",
    "<ev:music_vocal>",
    "<ev:pause_timing>",
    "<ev:shout_exclaim>",
    "<ev:throat_cough>",
)

# The mapping is semantic, with observed misspellings and malformed compound
# labels assigned according to their recognizable event suffix.
SOURCE_TO_NEMOTRON_NSV = {
    # Respiration and breath sounds.
    "[inhale]": "<ev:breath>",
    "[inahle]": "<ev:breath>",
    "[inhae]": "<ev:breath>",
    "[exhale]": "<ev:breath>",
    "[exale]": "<ev:breath>",
    "[breath]": "<ev:breath>",
    "[breathe]": "<ev:breath>",
    "[gasp]": "<ev:breath>",
    "[sigh]": "<ev:breath>",
    "[sniff]": "<ev:breath>",
    "[blow]": "<ev:breath>",
    "[yawn]": "<ev:breath>",
    "[mas-inhale]": "<ev:breath>",
    "[no-fim-das-contas-inhale]": "<ev:breath>",
    "[se-v-inhale]": "<ev:breath>",
    "[num-inhale]": "<ev:breath>",
    "[seguidas-inhale]": "<ev:breath>",
    "[كتر-ما-اليوم-بيبقى-مضغوط-breath]": "<ev:breath>",
    # Crying.
    "[cry]": "<ev:crying>",
    # Strained or effortful vocalizations.
    "[grunt]": "<ev:effort_strain>",
    "[groan]": "<ev:effort_strain>",
    "[moan]": "<ev:effort_strain>",
    "[growl]": "<ev:effort_strain>",
    # Generic environmental, handling, or physical noise.
    "[other-noise]": "<ev:gesture_noise>",
    "[other-nosie]": "<ev:gesture_noise>",
    "[other-wise]": "<ev:gesture_noise>",
    "[noise]": "<ev:gesture_noise>",
    "[isso-noise]": "<ev:gesture_noise>",
    "[typing]": "<ev:gesture_noise>",
    # Laughter-family vocalizations.
    "[chuckle]": "<ev:laughter>",
    "[chukle]": "<ev:laughter>",
    "[laugh]": "<ev:laughter>",
    "[laught]": "<ev:laughter>",
    "[giggle]": "<ev:laughter>",
    "[ginggle]": "<ev:laughter>",
    "[scoff]": "<ev:laughter>",
    # Oral clicks, smacks, trills, and other mouth noises.
    "[lip-smack]": "<ev:mouth_noise>",
    "[ip-smack]": "<ev:mouth_noise>",
    "[tongue-click]": "<ev:mouth_noise>",
    "[tongue-clikc]": "<ev:mouth_noise>",
    "[tingue-click]": "<ev:mouth_noise>",
    "[teeth-suck]": "<ev:mouth_noise>",
    "[tsk]": "<ev:mouth_noise>",
    "[swallow]": "<ev:mouth_noise>",
    "[snort]": "<ev:mouth_noise>",
    "[click]": "<ev:mouth_noise>",
    "[shush]": "<ev:mouth_noise>",
    "[lip-trill]": "<ev:mouth_noise>",
    # Melodic vocal sounds.
    "[hum-tune]": "<ev:music_vocal>",
    "[whistle]": "<ev:music_vocal>",
    # Exclamatory vocal sounds.
    "[whoop]": "<ev:shout_exclaim>",
    # Throat clearing, cough-like events, and upper-airway expulsions.
    "[clear-throat]": "<ev:throat_cough>",
    "[clear-thorat]": "<ev:throat_cough>",
    "[throat-clear]": "<ev:throat_cough>",
    "[throat-clearing]": "<ev:throat_cough>",
    "[cough]": "<ev:throat_cough>",
    "[sneeze]": "<ev:throat_cough>",
    "[hiccup]": "<ev:throat_cough>",
}

# These bracketed labels are transcription content/quality annotations rather
# than non-speech events, so converting them would invent the wrong acoustics.
PRESERVED_BRACKETED_ANNOTATIONS = {
    "[unintelligible]",
    "[unintellifible]",
    "[unintelligence]",
    "[crosstalk]",
}

# Spoken fillers are lexical transcript content. Remove annotation brackets and
# keep them as ordinary lowercase words instead of converting them to events.
BRACKETED_FILLER_TO_TEXT = {
    "[ah]": "ah",
    "[eh]": "eh",
    "[er]": "er",
    "[erm]": "erm",
    "[hm]": "hm",
    "[hmm]": "hmm",
    "[mhm]": "mhm",
    "[mm]": "mm",
    "[oh]": "oh",
    "[uh]": "uh",
    "[uhh]": "uh",
    "[uhm]": "uhm",
    "[um]": "um",
    "[umm]": "um",
}

BRACKETED_TOKEN_RE = re.compile(r"\[[^\[\]\n]+\]")
SPECIAL_TOKEN_RE = re.compile(r"<(?:spk:\d+|ev:[^<>]+)>")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_special_token_spacing(text: str) -> str:
    """Make speaker and event tokens standalone, ASCII-space-delimited tokens."""
    spaced = SPECIAL_TOKEN_RE.sub(lambda match: f" {match.group(0)} ", text)
    return re.sub(r"\s+", " ", spaced).strip()


def convert_text(
    text: str,
    source_counts: collections.Counter[str] | None = None,
    filler_counts: collections.Counter[str] | None = None,
) -> str:
    """Convert all recognized Turing NSV annotations in one text string."""
    bracketed_tokens = set(BRACKETED_TOKEN_RE.findall(text))
    known_fillers = {
        token for token in bracketed_tokens if token.casefold() in BRACKETED_FILLER_TO_TEXT
    }
    unknown = bracketed_tokens - set(SOURCE_TO_NEMOTRON_NSV) - known_fillers
    unknown -= PRESERVED_BRACKETED_ANNOTATIONS
    if unknown:
        raise ValueError(f"Unknown bracketed annotation(s): {sorted(unknown)}")

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        replacement = SOURCE_TO_NEMOTRON_NSV.get(token)
        if replacement is not None:
            if source_counts is not None:
                source_counts[token] += 1
            return replacement
        filler = BRACKETED_FILLER_TO_TEXT.get(token.casefold())
        if filler is not None:
            if filler_counts is not None:
                filler_counts[token] += 1
            return filler
        return token

    return apply_special_token_spacing(BRACKETED_TOKEN_RE.sub(replace, text))


def convert_manifest(input_path: Path, output_path: Path, stats_path: Path | None = None) -> dict:
    """Convert one Skills JSONL manifest atomically and return conversion statistics."""
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    source_sha256 = sha256_file(input_path)
    source_counts: collections.Counter[str] = collections.Counter()
    filler_counts: collections.Counter[str] = collections.Counter()
    preserved_counts: collections.Counter[str] = collections.Counter()
    final_target_counts: collections.Counter[str] = collections.Counter()
    rows = 0
    changed_rows = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.part.", dir=output_path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        with input_path.open(encoding="utf-8") as source, temporary_path.open(
            "w", encoding="utf-8"
        ) as output:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                old_text = str(row.get("text", ""))
                if row.get("expected_answer") != old_text:
                    raise ValueError(f"Line {line_number}: expected_answer differs from text")
                for token in BRACKETED_TOKEN_RE.findall(old_text):
                    if token in PRESERVED_BRACKETED_ANNOTATIONS:
                        preserved_counts[token] += 1
                new_text = convert_text(old_text, source_counts, filler_counts)
                final_target_counts.update(re.findall(r"<ev:[^<>]+>", new_text))
                row["text"] = new_text
                row["expected_answer"] = new_text
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows += 1
                changed_rows += new_text != old_text
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    previous_report = None
    if stats_path is not None and stats_path.is_file():
        candidate = json.loads(stats_path.read_text(encoding="utf-8"))
        candidate_targets = collections.Counter(candidate.get("target_nsv_counts", {}))
        current_targets = collections.Counter(
            {token: final_target_counts[token] for token in NEMOTRON_NSV_TOKENS}
        )
        if candidate.get("output_sha256") == source_sha256 or candidate_targets == current_targets:
            previous_report = candidate
    reported_source_counts = source_counts
    if not reported_source_counts and previous_report is not None:
        reported_source_counts = collections.Counter(previous_report.get("source_nsv_counts", {}))
    reported_filler_counts = filler_counts
    if not reported_filler_counts and previous_report is not None:
        reported_filler_counts = collections.Counter(previous_report.get("source_filler_counts", {}))
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "input_sha256": source_sha256,
        "output_sha256": sha256_file(output_path),
        "rows": rows,
        "changed_rows": changed_rows,
        "source_nsv_total": sum(reported_source_counts.values()),
        "source_nsv_counts": dict(reported_source_counts.most_common()),
        "source_filler_total": sum(reported_filler_counts.values()),
        "source_filler_counts": dict(reported_filler_counts.most_common()),
        "target_nsv_counts": {
            token: final_target_counts[token] for token in NEMOTRON_NSV_TOKENS
        },
        "preserved_non_nsv_counts": dict(preserved_counts.most_common()),
        "mapping": dict(SOURCE_TO_NEMOTRON_NSV),
        "filler_mapping": dict(BRACKETED_FILLER_TO_TEXT),
        "unknown_bracketed_annotations": [],
        "unicode_escaped_output": False,
    }
    if stats_path is not None:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return report


def parse_args() -> argparse.Namespace:
    default_manifest = Path(__file__).with_name("test.jsonl")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_manifest)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--stats-output", type=Path)
    args = parser.parse_args()
    if args.in_place == (args.output is not None):
        parser.error("choose exactly one of --in-place or --output")
    if args.in_place:
        args.output = args.input
    return args


def main() -> None:
    args = parse_args()
    report = convert_manifest(args.input, args.output, args.stats_output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
