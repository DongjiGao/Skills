# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Utilities for LibriSpeechMix SOT cpWER evaluation."""

from __future__ import annotations

import itertools
import json
import logging
import re
import string
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

SOT_STRICT_TAG_RE = re.compile(r"\[(s\d+)\]", re.IGNORECASE)
SOT_CANONICAL_TAG_RE = re.compile(
    r"\[(s\d+)\]|<spk:(\d+)>|(?:^|\s)speaker[_\s-]*(\d+)\s*[:\-]",
    re.IGNORECASE,
)
PUNCT_TRANSLATION = str.maketrans({char: " " for char in string.punctuation})

_PROMPT_ARTIFACT_PATTERNS = [
    re.compile(r"these\s+audio\s+clips\s+are\s+enrollment\s+samples[^.?!]*(?:[.?!]|$)", re.IGNORECASE),
    re.compile(r"transcribe\s+this\s+audio\s+with\s+punctuation\s+and\s+capitalization[^.?!]*(?:[.?!]|$)", re.IGNORECASE),
]


def normalize_no_pnc(text: str) -> str:
    """Normalize English SOT text for LibriSpeech-style WER.

    This mirrors the regular ASR evaluator's standard Whisper normalizer so
    digit/number-word variants are handled consistently. If the optional
    normalizer is unavailable, fall back to lowercase punctuation stripping.
    """
    text = text or ""
    try:
        from whisper_normalizer.english import EnglishTextNormalizer

        text = EnglishTextNormalizer()(text.lower())
    except Exception:
        text = text.lower().translate(PUNCT_TRANSLATION)
    return re.sub(r"\s+", " ", text).strip()


def _strip_prompt_artifacts(text: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    cleaned = text or ""
    for pattern in _PROMPT_ARTIFACT_PATTERNS:
        cleaned, replacements = pattern.subn(" ", cleaned)
        if replacements:
            warnings.append("stripped_prompt_artifact")
    if warnings:
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned, warnings


def _append_sot_segment(segments: list[dict[str, str]], speaker: str, text: str) -> None:
    text = text.strip()
    if text:
        segments.append({"speaker": speaker, "text": text})


def parse_sot_canonical(text: str, *, mode: str = "canonical", default_speaker: str = "s0") -> dict[str, Any]:
    """Parse a SOT transcript into canonical ordered speaker segments."""
    if mode not in {"strict", "canonical"}:
        raise ValueError(f"Unsupported SOT parse mode: {mode}")

    cleaned, warnings = _strip_prompt_artifacts(text or "")
    recovered = bool(warnings)
    segments: list[dict[str, str]] = []
    speaker_mapping: dict[str, str] = {}
    current_speaker: str | None = None
    cursor = 0
    tag_re = SOT_STRICT_TAG_RE if mode == "strict" else SOT_CANONICAL_TAG_RE
    matches = list(tag_re.finditer(cleaned))

    if not matches and cleaned.strip():
        if mode == "canonical":
            return {
                "segments": [],
                "format_parse_status": "failed",
                "format_warnings": warnings + ["no_speaker_tags_found"],
                "canonical_prediction": {"segments": []},
                "speaker_mapping": {},
            }
        return {
            "segments": [],
            "format_parse_status": "failed",
            "format_warnings": warnings + ["no_strict_speaker_tags_found"],
            "canonical_prediction": {"segments": []},
            "speaker_mapping": {},
        }

    for match in matches:
        chunk = cleaned[cursor : match.start()].strip()
        if chunk:
            if current_speaker is None:
                if mode == "strict":
                    warnings.append("untagged_text_before_first_speaker_tag")
                    _append_sot_segment(segments, "__invalid_untagged__", chunk)
                else:
                    warnings.append("untagged_text_assigned_default_speaker")
                    recovered = True
                    _append_sot_segment(segments, default_speaker, chunk)
            else:
                _append_sot_segment(segments, current_speaker, chunk)

        if mode == "strict":
            speaker = match.group(1).lower()
        else:
            bracket_tag, spk_tag, speaker_line_tag = match.groups()
            if bracket_tag is not None:
                speaker = bracket_tag.lower()
            elif spk_tag is not None:
                speaker = f"s{int(spk_tag)}"
                raw_label = f"<spk:{int(spk_tag)}>"
                speaker_mapping[raw_label] = speaker
                recovered = True
            else:
                speaker = f"s{int(speaker_line_tag)}"
                raw_label = f"speaker_{int(speaker_line_tag)}"
                speaker_mapping[raw_label] = speaker
                recovered = True
        current_speaker = speaker
        cursor = match.end()

    tail = cleaned[cursor:].strip()
    if tail:
        if current_speaker is None:
            if mode == "canonical":
                warnings.append("untagged_text_assigned_default_speaker")
                recovered = True
            current_speaker = default_speaker
        _append_sot_segment(segments, current_speaker, tail)

    if mode == "canonical":
        if not segments and (text or "").strip():
            status = "failed"
        elif recovered:
            status = "recovered"
        else:
            status = "exact"
    else:
        status = "failed" if warnings else ("exact" if segments or not (text or "").strip() else "failed")

    return {
        "segments": segments,
        "format_parse_status": status,
        "format_warnings": warnings,
        "canonical_prediction": {"segments": segments},
        "speaker_mapping": speaker_mapping,
    }


def _streams_from_sot_segments(segments: list[dict[str, str]]) -> OrderedDict[str, str]:
    streams: OrderedDict[str, list[str]] = OrderedDict()
    for segment in segments:
        streams.setdefault(segment["speaker"], []).append(segment["text"])
    return OrderedDict((speaker, normalize_no_pnc(" ".join(chunks))) for speaker, chunks in streams.items())


def parse_sot_speaker_streams(
    text: str, *, default_speaker: str = "s0", mode: str = "canonical"
) -> OrderedDict[str, str]:
    """Parse SOT text into speaker streams keyed by speaker token without brackets."""
    parsed = parse_sot_canonical(text, mode=mode, default_speaker=default_speaker)
    return _streams_from_sot_segments(parsed["segments"])


def _strip_sot_tags_for_content(text: str) -> str:
    return SOT_CANONICAL_TAG_RE.sub(" ", text or "")


def edit_counts(reference: list[str], hypothesis: list[str]) -> dict[str, int]:
    """Return Levenshtein substitutions, insertions, and deletions."""
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    dp: list[list[tuple[int, int, int, int]]] = [[(0, 0, 0, 0) for _ in range(cols)] for _ in range(rows)]

    for row in range(1, rows):
        prev = dp[row - 1][0]
        dp[row][0] = (prev[0] + 1, prev[1], prev[2], prev[3] + 1)
    for col in range(1, cols):
        prev = dp[0][col - 1]
        dp[0][col] = (prev[0] + 1, prev[1], prev[2] + 1, prev[3])

    for row in range(1, rows):
        for col in range(1, cols):
            if reference[row - 1] == hypothesis[col - 1]:
                hit_prev = dp[row - 1][col - 1]
                hit = (hit_prev[0], hit_prev[1], hit_prev[2], hit_prev[3])
                dp[row][col] = hit
                continue

            sub_prev = dp[row - 1][col - 1]
            ins_prev = dp[row][col - 1]
            del_prev = dp[row - 1][col]
            candidates = [
                (sub_prev[0] + 1, sub_prev[1] + 1, sub_prev[2], sub_prev[3]),
                (ins_prev[0] + 1, ins_prev[1], ins_prev[2] + 1, ins_prev[3]),
                (del_prev[0] + 1, del_prev[1], del_prev[2], del_prev[3] + 1),
            ]
            dp[row][col] = min(candidates, key=lambda item: (item[0], item[1], item[3], item[2]))

    errors, substitutions, insertions, deletions = dp[-1][-1]
    return {
        "errors": errors,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "ref_words": len(reference),
    }


def _counts_for_text(reference: str, hypothesis: str) -> dict[str, int]:
    return edit_counts(reference.split(), hypothesis.split())


def _cpwer_from_streams(ref_streams: OrderedDict[str, str], hyp_streams: OrderedDict[str, str]) -> dict[str, Any]:
    ref_items = list(ref_streams.items())
    hyp_items = list(hyp_streams.items())
    ref_count = len(ref_items)

    def empty_totals() -> dict[str, Any]:
        return {"errors": 0, "substitutions": 0, "insertions": 0, "deletions": 0, "ref_words": 0}

    def add_counts(total: dict[str, Any], counts: dict[str, int]) -> dict[str, Any]:
        out = dict(total)
        for key in ("errors", "substitutions", "insertions", "deletions", "ref_words"):
            out[key] += counts[key]
        return out

    def score_key(state: dict[str, Any]) -> tuple[int, int, int, int]:
        return (state["errors"], state["substitutions"], state["deletions"], state["insertions"])

    def maybe_keep(states: dict[int, dict[str, Any]], mask: int, candidate: dict[str, Any]) -> None:
        current = states.get(mask)
        if current is None or score_key(candidate) < score_key(current):
            states[mask] = candidate

    if ref_count == 0:
        state = empty_totals()
        for _, hyp_words in hyp_items:
            state = add_counts(state, _counts_for_text("", hyp_words))
        return {
            **state,
            "cpwer": state["errors"] / state["ref_words"] if state["ref_words"] else 0.0,
            "assignment": [
                {"reference": f"__empty_ref_{idx}", "hypothesis": hyp_speaker}
                for idx, (hyp_speaker, _) in enumerate(hyp_items)
            ],
            "reference_streams": dict(ref_streams),
            "hypothesis_streams": dict(hyp_streams),
        }

    dp: dict[int, dict[str, Any]] = {0: {**empty_totals(), "assignments": [None] * ref_count, "extras": []}}
    for hyp_index, (_, hyp_words) in enumerate(hyp_items):
        next_dp: dict[int, dict[str, Any]] = {}
        extra_counts = _counts_for_text("", hyp_words)
        for mask, state in dp.items():
            extra_state = add_counts(state, extra_counts)
            extra_state["assignments"] = list(state["assignments"])
            extra_state["extras"] = [*state["extras"], hyp_index]
            maybe_keep(next_dp, mask, extra_state)

            for ref_index, (_, ref_words) in enumerate(ref_items):
                bit = 1 << ref_index
                if mask & bit:
                    continue
                pair_state = add_counts(state, _counts_for_text(ref_words, hyp_words))
                assignments = list(state["assignments"])
                assignments[ref_index] = hyp_index
                pair_state["assignments"] = assignments
                pair_state["extras"] = list(state["extras"])
                maybe_keep(next_dp, mask | bit, pair_state)
        dp = next_dp

    best: dict[str, Any] | None = None
    best_key: tuple[int, int, int, int] | None = None
    for mask, state in dp.items():
        candidate = dict(state)
        assignments = list(candidate.pop("assignments"))
        extras = list(candidate.pop("extras"))
        for ref_index, (_, ref_words) in enumerate(ref_items):
            if assignments[ref_index] is None:
                candidate = add_counts(candidate, _counts_for_text(ref_words, ""))
        key = score_key(candidate)
        if best_key is None or key < best_key:
            assignment = []
            for ref_index, (ref_speaker, _) in enumerate(ref_items):
                hyp_index = assignments[ref_index]
                hyp_speaker = hyp_items[hyp_index][0] if hyp_index is not None else f"__empty_hyp_{ref_index}"
                assignment.append({"reference": ref_speaker, "hypothesis": hyp_speaker})
            for extra_idx, hyp_index in enumerate(extras):
                assignment.append({"reference": f"__empty_ref_{extra_idx}", "hypothesis": hyp_items[hyp_index][0]})
            candidate["assignment"] = assignment
            best = candidate
            best_key = key

    if best is None:
        best = empty_totals()
        best["assignment"] = []
    best["cpwer"] = best["errors"] / best["ref_words"] if best["ref_words"] else 0.0
    best["reference_streams"] = dict(ref_streams)
    best["hypothesis_streams"] = dict(hyp_streams)
    return best


def cpwer(reference_text: str, hypothesis_text: str, *, hypothesis_mode: str = "canonical") -> dict[str, Any]:
    """Compute cpWER using minimum speaker permutation matching."""
    ref_parse = parse_sot_canonical(reference_text, mode="strict")
    ref_streams = _streams_from_sot_segments(ref_parse["segments"])
    hyp_parse = parse_sot_canonical(hypothesis_text, mode=hypothesis_mode)
    if (
        hypothesis_mode == "canonical"
        and not hyp_parse["segments"]
        and (hypothesis_text or "").strip()
        and len(ref_streams) == 1
    ):
        speaker = next(iter(ref_streams), "s0")
        warnings = list(hyp_parse.get("format_warnings", []))
        if "untagged_text_assigned_single_reference_speaker" not in warnings:
            warnings.append("untagged_text_assigned_single_reference_speaker")
        hyp_parse = {
            "segments": [{"speaker": speaker, "text": hypothesis_text}],
            "format_parse_status": "recovered",
            "format_warnings": warnings,
            "canonical_prediction": {"segments": [{"speaker": speaker, "text": hypothesis_text}]},
            "speaker_mapping": {},
        }
    if hypothesis_mode == "canonical" and len(ref_streams) == 1 and len(hyp_parse["segments"]) > 1:
        ref_speaker = next(iter(ref_streams), "s0")
        observed_speakers = {segment["speaker"] for segment in hyp_parse["segments"]}
        if len(observed_speakers) > 1:
            warnings = list(hyp_parse.get("format_warnings", []))
            if "collapsed_predicted_speakers_single_reference" not in warnings:
                warnings.append("collapsed_predicted_speakers_single_reference")
            hyp_parse = {
                **hyp_parse,
                "segments": [{**segment, "speaker": ref_speaker} for segment in hyp_parse["segments"]],
                "format_parse_status": "recovered",
                "format_warnings": warnings,
                "canonical_prediction": {
                    "segments": [{**segment, "speaker": ref_speaker} for segment in hyp_parse["segments"]]
                },
            }

    hyp_streams = _streams_from_sot_segments(hyp_parse["segments"])
    result = _cpwer_from_streams(ref_streams, hyp_streams)
    result["format_parse_status"] = hyp_parse["format_parse_status"]
    result["format_warnings"] = hyp_parse["format_warnings"]
    result["canonical_prediction"] = hyp_parse["canonical_prediction"]
    result["canonical_reference"] = ref_parse["canonical_prediction"]
    result["speaker_mapping"] = hyp_parse["speaker_mapping"]
    return result


def content_wer(reference_text: str, hypothesis_text: str) -> dict[str, Any]:
    """Compute flattened content WER after stripping SOT speaker tags."""
    ref = normalize_no_pnc(_strip_sot_tags_for_content(reference_text))
    hyp = normalize_no_pnc(_strip_sot_tags_for_content(hypothesis_text))
    counts = edit_counts(ref.split(), hyp.split())
    return {
        "wer": counts["errors"] / counts["ref_words"] if counts["ref_words"] else 0.0,
        "errors": counts["errors"],
        "substitutions": counts["substitutions"],
        "insertions": counts["insertions"],
        "deletions": counts["deletions"],
        "ref_words": counts["ref_words"],
    }


def sot_to_seglst(record: dict[str, Any], text_key: str) -> list[dict[str, Any]]:
    """Convert a SOT transcript into coarse SegLST entries for MeetEval cpWER."""
    text = record.get(text_key, "") or ""
    streams = parse_sot_speaker_streams(text)
    session_id = record.get("sample_id") or record.get("id") or Path(record.get("audio_filepath", "sample")).stem
    offset = float(record.get("offset", 0.0) or 0.0)
    duration = max(float(record.get("duration", 0.0) or 0.0), 0.01)
    step = duration / max(len(streams), 1)

    entries = []
    for idx, (speaker, words) in enumerate(streams.items()):
        start = offset + idx * step
        end = offset + (idx + 1) * step
        entries.append(
            {
                "session_id": str(session_id),
                "words": words,
                "speaker": speaker,
                "start_time": round(start, 2),
                "end_time": round(end, 2),
            }
        )
    return entries


def write_meeteval_artifacts(records: list[dict[str, Any]], output_file: str | Path) -> dict[str, str]:
    """Write reference and hypothesis SegLST JSON files next to an evaluated output file."""
    output_path = Path(output_file)
    stem = output_path.with_suffix("")
    ref_path = stem.parent / f"{stem.name}_ref.seglst.json"
    hyp_path = stem.parent / f"{stem.name}_hyp.seglst.json"
    avg_path = stem.parent / f"{stem.name}_meeteval_average.json"
    per_path = stem.parent / f"{stem.name}_meeteval_per_reco.json"

    ref_entries: list[dict[str, Any]] = []
    hyp_entries: list[dict[str, Any]] = []
    for record in records:
        ref_entries.extend(sot_to_seglst(record, "text"))
        hyp_record = dict(record)
        hyp_record["pred_text"] = record.get("pred_text") or record.get("generation", "")
        hyp_entries.extend(sot_to_seglst(hyp_record, "pred_text"))

    ref_path.write_text(json.dumps(ref_entries, indent=2) + "\n", encoding="utf-8")
    hyp_path.write_text(json.dumps(hyp_entries, indent=2) + "\n", encoding="utf-8")

    artifacts = {"ref": str(ref_path), "hyp": str(hyp_path)}
    command = [
        "meeteval-wer",
        "cpwer",
        "-r",
        str(ref_path),
        "-h",
        str(hyp_path),
        "--normalizer",
        "chime8",
        "--average-out",
        str(avg_path),
        "--per-reco-out",
        str(per_path),
        "--partial",
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        LOG.warning("meeteval-wer is not installed; wrote SegLST artifacts but skipped MeetEval cpWER.")
    except subprocess.CalledProcessError as exc:
        LOG.warning("meeteval-wer cpwer failed: %s", exc.stderr.strip() or exc.stdout.strip())
    else:
        artifacts["average"] = str(avg_path)
        artifacts["per_reco"] = str(per_path)
    return artifacts
