# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scoring for the versioned NFA timestamp public reconstruction.

Timestamp tokens use ``<|N|>`` where one tick is 80 ms. A timestamp true
positive requires a word-sequence match and strictly less than 240 ms error at
both boundaries. Word matches are established by deterministic Levenshtein
alignment. WER is computed after removing timestamp tokens.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from nemo_skills.evaluation.evaluator.base import BaseEvaluator, BaseEvaluatorConfig
from nemo_skills.evaluation.metrics.base import BaseMetrics, as_float, as_int, as_percentage
from nemo_skills.utils import nested_dataclass

PROTOCOL_VERSION = "nfa-timestamp-public-reconstruction-v1"
TICK_MILLISECONDS = 80
TOLERANCE_MILLISECONDS = 240
# The paper criterion is strict (<240 ms), so a three-tick error is not a TP.
TOLERANCE_TICKS = TOLERANCE_MILLISECONDS // TICK_MILLISECONDS

SOURCE_KEYS = (
    "librispeech-test-other",
    "common-voice-16.1-en",
    "common-voice-16.1-es",
    "common-voice-16.1-fr",
    "common-voice-16.1-de",
)

_TIMESTAMP_TOKEN_RE = re.compile(r"<\|(\d+)\|>")
# Lookahead permits a boundary token to serve as both one word's end and the
# next word's start. Repeated end/start tokens are accepted as well.
_TIMESTAMP_SPAN_RE = re.compile(r"(?=(<\|(?P<start>\d+)\|>\s*(?P<word>[^<>\s]+)\s*<\|(?P<end>\d+)\|>))")


def strip_timestamp_tokens(text: object) -> str:
    """Strip only numeric NFA timestamp tokens and normalize whitespace."""
    return " ".join(_TIMESTAMP_TOKEN_RE.sub(" ", str(text or "")).split())


def _parse_timestamped_hypothesis(generation: str) -> tuple[list[str], dict[int, tuple[int, int]], dict[str, int]]:
    """Return stripped words, valid timestamp spans by word index, diagnostics."""
    hypothesis_words = strip_timestamp_tokens(generation).split()
    spans: dict[int, tuple[int, int]] = {}
    used_token_offsets: set[int] = set()
    invalid_spans = 0

    for match in _TIMESTAMP_SPAN_RE.finditer(generation):
        word = match.group("word")
        word_index = len(strip_timestamp_tokens(generation[: match.start("word")]).split())
        start_tick = int(match.group("start"))
        end_tick = int(match.group("end"))
        if (
            word_index >= len(hypothesis_words)
            or hypothesis_words[word_index] != word
            or word_index in spans
            or end_tick < start_tick
        ):
            invalid_spans += 1
            continue
        spans[word_index] = (start_tick, end_tick)
        used_token_offsets.add(match.start("start") - 2)
        used_token_offsets.add(match.start("end") - 2)

    token_offsets = {match.start() for match in _TIMESTAMP_TOKEN_RE.finditer(generation)}
    diagnostics = {
        "timestamped_hyp_words": len(spans),
        "untimestamped_hyp_words": len(hypothesis_words) - len(spans),
        "invalid_timestamp_spans": invalid_spans,
        "unparsed_timestamp_tokens": len(token_offsets - used_token_offsets),
    }
    return hypothesis_words, spans, diagnostics


def _word_alignment(reference: list[str], hypothesis: list[str]) -> dict[str, Any]:
    """Levenshtein counts plus exact-word index pairs with stable tie breaking."""
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    distance = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
    for ref_index in range(ref_len + 1):
        distance[ref_index][0] = ref_index
    for hyp_index in range(hyp_len + 1):
        distance[0][hyp_index] = hyp_index

    for ref_index in range(1, ref_len + 1):
        for hyp_index in range(1, hyp_len + 1):
            substitution_cost = reference[ref_index - 1] != hypothesis[hyp_index - 1]
            distance[ref_index][hyp_index] = min(
                distance[ref_index - 1][hyp_index - 1] + substitution_cost,
                distance[ref_index - 1][hyp_index] + 1,
                distance[ref_index][hyp_index - 1] + 1,
            )

    ref_index = ref_len
    hyp_index = hyp_len
    substitutions = insertions = deletions = 0
    matches: list[tuple[int, int]] = []
    while ref_index or hyp_index:
        if (
            ref_index
            and hyp_index
            and reference[ref_index - 1] == hypothesis[hyp_index - 1]
            and distance[ref_index][hyp_index] == distance[ref_index - 1][hyp_index - 1]
        ):
            matches.append((ref_index - 1, hyp_index - 1))
            ref_index -= 1
            hyp_index -= 1
        elif ref_index and hyp_index and distance[ref_index][hyp_index] == distance[ref_index - 1][hyp_index - 1] + 1:
            substitutions += 1
            ref_index -= 1
            hyp_index -= 1
        elif hyp_index and distance[ref_index][hyp_index] == distance[ref_index][hyp_index - 1] + 1:
            insertions += 1
            hyp_index -= 1
        else:
            deletions += 1
            ref_index -= 1

    matches.reverse()
    return {
        "wer_errors": substitutions + insertions + deletions,
        "wer_substitutions": substitutions,
        "wer_insertions": insertions,
        "wer_deletions": deletions,
        "word_matches": matches,
    }


def _reference_words(data_point: dict[str, Any]) -> list[dict[str, Any]]:
    words = data_point.get("reference_words")
    if not isinstance(words, list) or not words:
        raise ValueError("NFA reconstruction row is missing non-empty reference_words")
    validated = []
    for index, item in enumerate(words):
        if not isinstance(item, dict) or set(item) != {"word", "start_tick", "end_tick"}:
            raise ValueError(f"Invalid reference_words[{index}] schema")
        word = item["word"]
        start_tick = item["start_tick"]
        end_tick = item["end_tick"]
        if not isinstance(word, str) or not word or word.split() != [word]:
            raise ValueError(f"Invalid reference word at index {index}: {word!r}")
        if (
            isinstance(start_tick, bool)
            or isinstance(end_tick, bool)
            or not isinstance(start_tick, int)
            or not isinstance(end_tick, int)
            or start_tick < 0
            or end_tick < start_tick
        ):
            raise ValueError(f"Invalid reference ticks at index {index}: {start_tick!r}, {end_tick!r}")
        validated.append(item)
    return validated


def evaluate_nfa_timestamp_sample(data_point: dict[str, Any]) -> dict[str, Any]:
    """Score one generation while retaining additive corpus statistics."""
    if data_point.get("protocol_profile") != PROTOCOL_VERSION:
        raise ValueError(f"Expected protocol_profile={PROTOCOL_VERSION!r}, got {data_point.get('protocol_profile')!r}")
    source = data_point.get("subset_for_metrics")
    if source not in SOURCE_KEYS:
        raise ValueError(f"Unexpected NFA reconstruction source: {source!r}")

    reference_items = _reference_words(data_point)
    reference = [item["word"] for item in reference_items]
    generation = str(data_point.get("generation") or "")
    hypothesis, timestamp_spans, diagnostics = _parse_timestamped_hypothesis(generation)
    alignment = _word_alignment(reference, hypothesis)

    true_positives = 0
    timestamped_word_matches = 0
    start_deviation_ticks_sum = 0
    end_deviation_ticks_sum = 0
    for ref_index, hyp_index in alignment["word_matches"]:
        if hyp_index not in timestamp_spans:
            continue
        hyp_start, hyp_end = timestamp_spans[hyp_index]
        ref_start = int(reference_items[ref_index]["start_tick"])
        ref_end = int(reference_items[ref_index]["end_tick"])
        start_error = abs(hyp_start - ref_start)
        end_error = abs(hyp_end - ref_end)
        timestamped_word_matches += 1
        start_deviation_ticks_sum += start_error
        end_deviation_ticks_sum += end_error
        if start_error < TOLERANCE_TICKS and end_error < TOLERANCE_TICKS:
            true_positives += 1

    reference_count = len(reference)
    hypothesis_count = len(hypothesis)
    return {
        "generation": generation,
        "text": " ".join(reference),
        "pred_text": " ".join(hypothesis),
        "nfa_source": source,
        "wer": alignment["wer_errors"] / reference_count,
        "wer_errors": alignment["wer_errors"],
        "wer_substitutions": alignment["wer_substitutions"],
        "wer_insertions": alignment["wer_insertions"],
        "wer_deletions": alignment["wer_deletions"],
        "reference_words_count": reference_count,
        "hypothesis_words_count": hypothesis_count,
        "timestamp_true_positives": true_positives,
        "timestamp_predictions": hypothesis_count,
        "timestamp_references": reference_count,
        "timestamped_word_matches": timestamped_word_matches,
        "start_deviation_ticks_sum": start_deviation_ticks_sum,
        "end_deviation_ticks_sum": end_deviation_ticks_sum,
        **diagnostics,
    }


@nested_dataclass(kw_only=True)
class NFATimestampReconstructionEvaluatorConfig(BaseEvaluatorConfig):
    pass


class NFATimestampReconstructionEvaluator(BaseEvaluator):
    async def eval_single(self, data_point: dict[str, Any]) -> dict[str, Any]:
        return evaluate_nfa_timestamp_sample(data_point)


def _empty_totals() -> dict[str, int]:
    return {
        "entries": 0,
        "wer_errors": 0,
        "wer_substitutions": 0,
        "wer_insertions": 0,
        "wer_deletions": 0,
        "reference_words": 0,
        "hypothesis_words": 0,
        "true_positives": 0,
        "timestamped_word_matches": 0,
        "start_deviation_ticks_sum": 0,
        "end_deviation_ticks_sum": 0,
        "untimestamped_hyp_words": 0,
        "invalid_timestamp_spans": 0,
        "unparsed_timestamp_tokens": 0,
    }


def _accumulate(totals: dict[str, int], prediction: dict[str, Any]) -> None:
    totals["entries"] += 1
    totals["wer_errors"] += int(prediction["wer_errors"])
    totals["wer_substitutions"] += int(prediction["wer_substitutions"])
    totals["wer_insertions"] += int(prediction["wer_insertions"])
    totals["wer_deletions"] += int(prediction["wer_deletions"])
    totals["reference_words"] += int(prediction["reference_words_count"])
    totals["hypothesis_words"] += int(prediction["hypothesis_words_count"])
    totals["true_positives"] += int(prediction["timestamp_true_positives"])
    totals["timestamped_word_matches"] += int(prediction["timestamped_word_matches"])
    totals["start_deviation_ticks_sum"] += int(prediction["start_deviation_ticks_sum"])
    totals["end_deviation_ticks_sum"] += int(prediction["end_deviation_ticks_sum"])
    totals["untimestamped_hyp_words"] += int(prediction["untimestamped_hyp_words"])
    totals["invalid_timestamp_spans"] += int(prediction["invalid_timestamp_spans"])
    totals["unparsed_timestamp_tokens"] += int(prediction["unparsed_timestamp_tokens"])


def _reported_values(totals: dict[str, int], *, include_counts: bool) -> dict[str, float | int]:
    reference_words = totals["reference_words"]
    hypothesis_words = totals["hypothesis_words"]
    matches = totals["timestamped_word_matches"]
    values: dict[str, float | int] = {
        "wer": round(100.0 * totals["wer_errors"] / reference_words, 2),
        "timestamp_precision": round(
            100.0 * totals["true_positives"] / hypothesis_words if hypothesis_words else 0.0,
            2,
        ),
        "timestamp_recall": round(100.0 * totals["true_positives"] / reference_words, 2),
    }
    if matches:
        values["start_deviation_ms"] = round(
            TICK_MILLISECONDS * totals["start_deviation_ticks_sum"] / matches,
            2,
        )
        values["end_deviation_ms"] = round(
            TICK_MILLISECONDS * totals["end_deviation_ticks_sum"] / matches,
            2,
        )
    if include_counts:
        values.update(
            {
                "wer_errors": totals["wer_errors"],
                "wer_substitutions": totals["wer_substitutions"],
                "wer_insertions": totals["wer_insertions"],
                "wer_deletions": totals["wer_deletions"],
                "reference_words": reference_words,
                "hypothesis_words": hypothesis_words,
                "timestamp_true_positives": totals["true_positives"],
                "timestamped_word_matches": matches,
                "untimestamped_hyp_words": totals["untimestamped_hyp_words"],
                "invalid_timestamp_spans": totals["invalid_timestamp_spans"],
                "unparsed_timestamp_tokens": totals["unparsed_timestamp_tokens"],
                "scored_entries": totals["entries"],
            }
        )
    return values


class NFATimestampReconstructionMetrics(BaseMetrics):
    """Micro-aggregate NFA timestamp and stripped-token WER statistics."""

    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        super().__init__(compute_no_answer=compute_no_answer)
        if max_k != 1:
            raise ValueError("NFATimestampReconstructionMetrics supports one greedy generation only")
        self.max_k = 1

    def reset(self):
        super().reset()
        self.corpus_totals = _empty_totals()
        self.subset_totals: dict[str, dict[str, int]] = defaultdict(_empty_totals)

    def _get_score_dict(self, prediction):
        return {
            "timestamp_exact": (
                prediction.get("timestamp_true_positives") == prediction.get("timestamp_references")
                and prediction.get("wer_errors") == 0
            )
        }

    def get_incorrect_sample(self, prediction):
        prediction = prediction.copy()
        reference_words = int(prediction.get("reference_words_count", 0))
        prediction.update(
            wer_errors=reference_words,
            wer_substitutions=0,
            wer_insertions=0,
            wer_deletions=reference_words,
            hypothesis_words_count=0,
            timestamp_true_positives=0,
            timestamped_word_matches=0,
            start_deviation_ticks_sum=0,
            end_deviation_ticks_sum=0,
            timestamped_hyp_words=0,
            untimestamped_hyp_words=0,
            invalid_timestamp_spans=0,
            unparsed_timestamp_tokens=0,
            generation="",
        )
        return prediction

    def update(self, predictions):
        if len(predictions) != 1:
            raise ValueError("NFATimestampReconstructionMetrics expects exactly one prediction per sample")
        prediction = predictions[0]
        source = prediction.get("nfa_source")
        if source not in SOURCE_KEYS:
            raise ValueError(f"Unexpected NFA reconstruction source: {source!r}")
        super().update(predictions)
        _accumulate(self.corpus_totals, prediction)
        _accumulate(self.subset_totals[source], prediction)

    def get_metrics(self):
        if self.total <= 0 or self.corpus_totals["reference_words"] <= 0:
            raise ValueError("NFA timestamp reconstruction has no reference words to score")
        values = _reported_values(self.corpus_totals, include_counts=True)
        for source in SOURCE_KEYS:
            totals = self.subset_totals.get(source)
            if not totals or totals["reference_words"] <= 0:
                continue
            subset_values = _reported_values(totals, include_counts=False)
            for metric_name, metric_value in subset_values.items():
                values[f"{metric_name}::{source}"] = metric_value
        values["protocol_profile"] = PROTOCOL_VERSION
        values["comparability"] = "public-reconstruction-proxy"
        self.update_common_metrics(values)
        return {"pass@1": values}

    def evaluations_to_print(self):
        return ["pass@1"]

    def metrics_to_print(self):
        metrics = {
            "wer": as_percentage,
            "timestamp_precision": as_percentage,
            "timestamp_recall": as_percentage,
            "start_deviation_ms": as_float,
            "end_deviation_ms": as_float,
            "wer_errors": as_int,
            "wer_substitutions": as_int,
            "wer_insertions": as_int,
            "wer_deletions": as_int,
            "reference_words": as_int,
            "hypothesis_words": as_int,
            "timestamp_true_positives": as_int,
            "timestamped_word_matches": as_int,
            "untimestamped_hyp_words": as_int,
            "invalid_timestamp_spans": as_int,
            "unparsed_timestamp_tokens": as_int,
            "scored_entries": as_int,
            "num_entries": as_int,
        }
        for source in SOURCE_KEYS:
            metrics[f"wer::{source}"] = as_percentage
            metrics[f"timestamp_precision::{source}"] = as_percentage
            metrics[f"timestamp_recall::{source}"] = as_percentage
            metrics[f"start_deviation_ms::{source}"] = as_float
            metrics[f"end_deviation_ms::{source}"] = as_float
        return metrics
