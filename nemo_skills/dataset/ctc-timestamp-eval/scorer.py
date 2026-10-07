# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speaker-aware word timestamp scoring for ``ctc-timestamp-eval``."""

from __future__ import annotations

import itertools
import json
import math
import unicodedata
from collections import defaultdict
from typing import Any

from nemo_skills.evaluation.evaluator.base import BaseEvaluator, BaseEvaluatorConfig
from nemo_skills.evaluation.metrics.base import BaseMetrics, as_float, as_int, as_percentage
from nemo_skills.utils import nested_dataclass

DEFAULT_TOLERANCE_SECONDS = 0.2


def _normalized_word(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    start = 0
    end = len(text)
    while start < end and unicodedata.category(text[start])[0] not in "LMN":
        start += 1
    while end > start and unicodedata.category(text[end - 1])[0] not in "LMN":
        end -= 1
    return text[start:end]


def _word_alignment(reference: list[str], hypothesis: list[str]) -> dict[str, Any]:
    """Return rapidfuzz-backed Levenshtein counts and exact-word index pairs."""
    if reference == hypothesis:
        return {
            "errors": 0,
            "substitutions": 0,
            "insertions": 0,
            "deletions": 0,
            "matches": list(zip(range(len(reference)), range(len(hypothesis)))),
        }
    if not reference or not hypothesis:
        return {
            "errors": len(reference) + len(hypothesis),
            "substitutions": 0,
            "insertions": len(hypothesis),
            "deletions": len(reference),
            "matches": [],
        }

    import jiwer

    alignment = jiwer.process_words(" ".join(reference), " ".join(hypothesis))
    matches = []
    for chunk in alignment.alignments[0]:
        if chunk.type == "equal":
            matches.extend(
                zip(
                    range(chunk.ref_start_idx, chunk.ref_end_idx),
                    range(chunk.hyp_start_idx, chunk.hyp_end_idx),
                )
            )
    return {
        "errors": alignment.substitutions + alignment.insertions + alignment.deletions,
        "substitutions": alignment.substitutions,
        "insertions": alignment.insertions,
        "deletions": alignment.deletions,
        "matches": matches,
    }


def _validate_word(item: dict[str, Any], index: int, *, reference: bool) -> dict[str, Any] | None:
    word = item.get("word", item.get("text"))
    normalized = _normalized_word(word)
    if not normalized:
        return None
    if normalized.split() != [normalized]:
        if reference:
            raise ValueError(f"reference_words[{index}] must contain exactly one word")
        return None
    start = item.get("start", item.get("start_time"))
    end = item.get("end", item.get("end_time"))
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, (int, float))
        or not isinstance(end, (int, float))
    ):
        if reference:
            raise ValueError(f"reference_words[{index}] has invalid start/end values")
        return None
    start = float(start)
    end = float(end)
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
        if reference:
            raise ValueError(f"reference_words[{index}] has an invalid interval: {start}, {end}")
        return None
    speaker = str(item.get("speaker", item.get("speaker_tag", "speaker0")))
    return {"word": str(word), "normalized": normalized, "start": start, "end": end, "speaker": speaker}


def _reference_words(data_point: dict[str, Any]) -> list[dict[str, Any]]:
    items = data_point.get("reference_words")
    if not isinstance(items, list) or not items:
        raise ValueError("ctc-timestamp-eval row is missing non-empty reference_words")
    words = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(f"reference_words[{index}] must be an object")
        validated = _validate_word(item, index, reference=True)
        if validated is not None:
            words.append(validated)
    if not words:
        raise ValueError("ctc-timestamp-eval row has no scoreable reference words")
    return words


def _decode_json(value: object) -> object:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _find_timestamp_payload(data_point: dict[str, Any]) -> object:
    candidates = [data_point, _decode_json(data_point.get("generation"))]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for key in ("speaker_word_timestamps", "word_timestamps", "words"):
            if key in candidate:
                return candidate[key]
        for key in ("timestamps", "ctc_timestamps", "timestamp_result"):
            nested = _decode_json(candidate.get(key))
            if not isinstance(nested, dict):
                continue
            for word_key in ("speaker_word_timestamps", "word_timestamps", "words"):
                if word_key in nested:
                    return nested[word_key]
    raise ValueError("Prediction has no word timestamps; expected speaker_word_timestamps, word_timestamps, or words")


def _timestamp_error(data_point: dict[str, Any]) -> str | None:
    """Why the serving plugin returned no word timestamps: its ``ctc_timestamps.error``, if set."""
    payload = _decode_json(data_point.get("ctc_timestamps"))
    error = payload.get("error") if isinstance(payload, dict) else None
    return str(error) if error else None


def _prediction_words(data_point: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    payload = _find_timestamp_payload(data_point)
    flattened = []
    malformed = 0
    if isinstance(payload, dict):
        iterable = []
        for speaker, items in payload.items():
            if isinstance(items, list):
                iterable.extend((speaker, item) for item in items)
            else:
                malformed += 1
    elif isinstance(payload, list):
        iterable = [(None, item) for item in payload]
    else:
        raise TypeError("Word timestamp payload must be a list or a speaker-to-list mapping")

    for index, (speaker, item) in enumerate(iterable):
        if not isinstance(item, dict):
            malformed += 1
            continue
        item = dict(item)
        if speaker is not None and "speaker" not in item and "speaker_tag" not in item:
            item["speaker"] = speaker
        validated = _validate_word(item, index, reference=False)
        if validated is None:
            malformed += 1
        else:
            flattened.append(validated)
    return sorted(flattened, key=lambda item: (item["start"], item["end"])), malformed


def _by_speaker(words: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = defaultdict(list)
    for item in words:
        grouped[item["speaker"]].append(item)
    return dict(grouped)


def _speaker_assignment(
    reference: dict[str, list[dict[str, Any]]], hypothesis: dict[str, list[dict[str, Any]]]
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """Find the minimum-word-error permutation, including dummy speakers."""
    ref_streams = list(reference.values())
    hyp_streams = list(hypothesis.values())
    size = max(len(ref_streams), len(hyp_streams))
    if size > 8:
        raise ValueError(f"At most 8 speakers are supported, got {size}")
    ref_streams.extend([[] for _ in range(size - len(ref_streams))])
    hyp_streams.extend([[] for _ in range(size - len(hyp_streams))])

    def cost(ref_items, hyp_items):
        ref_words = [item["normalized"] for item in ref_items]
        hyp_words = [item["normalized"] for item in hyp_items]
        return _word_alignment(ref_words, hyp_words)["errors"]

    costs = [[cost(ref_items, hyp_items) for hyp_items in hyp_streams] for ref_items in ref_streams]
    permutation = min(itertools.permutations(range(size)), key=lambda p: sum(costs[i][p[i]] for i in range(size)))
    return [(ref_streams[index], hyp_streams[hyp_index]) for index, hyp_index in enumerate(permutation)]


def evaluate_ctc_timestamp_sample(
    data_point: dict[str, Any], tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS
) -> dict[str, Any]:
    """Evaluate one row using speaker-permuted WER and aligned boundary errors."""
    if tolerance_seconds < 0:
        raise ValueError("tolerance_seconds must be non-negative")
    reference = _reference_words(data_point)
    hypothesis, malformed = _prediction_words(data_point)
    assignment = _speaker_assignment(_by_speaker(reference), _by_speaker(hypothesis))

    totals = defaultdict(int)
    start_error_sum = end_error_sum = 0.0
    for ref_items, hyp_items in assignment:
        alignment = _word_alignment(
            [item["normalized"] for item in ref_items], [item["normalized"] for item in hyp_items]
        )
        for name in ("errors", "substitutions", "insertions", "deletions"):
            totals[name] += alignment[name]
        for ref_index, hyp_index in alignment["matches"]:
            ref_item = ref_items[ref_index]
            hyp_item = hyp_items[hyp_index]
            start_error = abs(ref_item["start"] - hyp_item["start"])
            end_error = abs(ref_item["end"] - hyp_item["end"])
            start_error_sum += start_error
            end_error_sum += end_error
            totals["timestamped_matches"] += 1
            totals["timestamp_true_positives"] += int(start_error <= tolerance_seconds + 1e-9)
            totals["timestamp_true_positives"] += int(end_error <= tolerance_seconds + 1e-9)

    reference_count = len(reference)
    hypothesis_count = len(hypothesis)
    result = {
        "timestamp_subset": str(data_point.get("subset_for_metrics", "unknown")),
        "wer": totals["errors"] / reference_count,
        "wer_errors": totals["errors"],
        "wer_substitutions": totals["substitutions"],
        "wer_insertions": totals["insertions"],
        "wer_deletions": totals["deletions"],
        "reference_words_count": reference_count,
        "hypothesis_words_count": hypothesis_count,
        "timestamp_true_positives": totals["timestamp_true_positives"],
        "timestamped_word_matches": totals["timestamped_matches"],
        "start_error_seconds_sum": start_error_sum,
        "end_error_seconds_sum": end_error_sum,
        "malformed_timestamp_words": malformed,
        "timestamp_error": _timestamp_error(data_point),
        "timestamp_tolerance_seconds": tolerance_seconds,
    }
    return result


@nested_dataclass(kw_only=True)
class CTCTimestampEvaluatorConfig(BaseEvaluatorConfig):
    tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS


class CTCTimestampEvaluator(BaseEvaluator):
    def __init__(self, config: dict[str, Any], num_parallel_requests: int = 10):
        super().__init__(CTCTimestampEvaluatorConfig(**config), num_parallel_requests)

    async def eval_single(self, data_point: dict[str, Any]) -> dict[str, Any]:
        return evaluate_ctc_timestamp_sample(data_point, float(self.config.tolerance_seconds))


def _empty_totals() -> dict[str, int | float]:
    return {
        "entries": 0,
        "wer_errors": 0,
        "wer_substitutions": 0,
        "wer_insertions": 0,
        "wer_deletions": 0,
        "reference_words": 0,
        "hypothesis_words": 0,
        "true_positives": 0,
        "timestamped_matches": 0,
        "start_error_seconds_sum": 0.0,
        "end_error_seconds_sum": 0.0,
        "malformed_timestamp_words": 0,
        "timestamp_unavailable_entries": 0,
    }


def _accumulate(totals: dict[str, int | float], prediction: dict[str, Any]) -> None:
    totals["entries"] += 1
    totals["wer_errors"] += int(prediction["wer_errors"])
    totals["wer_substitutions"] += int(prediction["wer_substitutions"])
    totals["wer_insertions"] += int(prediction["wer_insertions"])
    totals["wer_deletions"] += int(prediction["wer_deletions"])
    totals["reference_words"] += int(prediction["reference_words_count"])
    totals["hypothesis_words"] += int(prediction["hypothesis_words_count"])
    totals["true_positives"] += int(prediction["timestamp_true_positives"])
    totals["timestamped_matches"] += int(prediction["timestamped_word_matches"])
    totals["start_error_seconds_sum"] += float(prediction["start_error_seconds_sum"])
    totals["end_error_seconds_sum"] += float(prediction["end_error_seconds_sum"])
    totals["malformed_timestamp_words"] += int(prediction["malformed_timestamp_words"])
    totals["timestamp_unavailable_entries"] += int(bool(prediction.get("timestamp_error")))


def _reported_values(totals: dict[str, int | float], include_counts: bool) -> dict[str, float | int]:
    references = int(totals["reference_words"])
    hypotheses = int(totals["hypothesis_words"])
    true_positives = int(totals["true_positives"])
    matches = int(totals["timestamped_matches"])
    precision = true_positives / (2 * hypotheses) if hypotheses else 0.0
    recall = true_positives / (2 * references) if references else 0.0
    values: dict[str, float | int] = {
        "wer": round(100.0 * int(totals["wer_errors"]) / references, 2),
        "timestamp_precision": round(100.0 * precision, 2),
        "timestamp_recall": round(100.0 * recall, 2),
        "timestamp_f1": round(200.0 * precision * recall / (precision + recall), 2) if precision + recall else 0.0,
    }
    if matches:
        values["start_mae_ms"] = round(1000.0 * float(totals["start_error_seconds_sum"]) / matches, 2)
        values["end_mae_ms"] = round(1000.0 * float(totals["end_error_seconds_sum"]) / matches, 2)
    if include_counts:
        values.update(
            {
                "wer_errors": int(totals["wer_errors"]),
                "wer_substitutions": int(totals["wer_substitutions"]),
                "wer_insertions": int(totals["wer_insertions"]),
                "wer_deletions": int(totals["wer_deletions"]),
                "reference_words": references,
                "hypothesis_words": hypotheses,
                "timestamp_true_positives": true_positives,
                "timestamp_reference_events": 2 * references,
                "timestamp_hypothesis_events": 2 * hypotheses,
                "timestamped_word_matches": matches,
                "malformed_timestamp_words": int(totals["malformed_timestamp_words"]),
                "timestamp_unavailable_entries": int(totals["timestamp_unavailable_entries"]),
                "scored_entries": int(totals["entries"]),
            }
        )
    return values


class CTCTimestampMetrics(BaseMetrics):
    """Micro-aggregate timestamp metrics with speaker-permuted word alignment."""

    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        super().__init__(compute_no_answer=compute_no_answer)
        if max_k != 1:
            raise ValueError("CTCTimestampMetrics supports one greedy generation only")
        self.max_k = 1

    def reset(self):
        super().reset()
        self.tolerance_seconds: float | None = None
        self.corpus_totals = _empty_totals()
        self.subset_totals: dict[str, dict[str, int | float]] = defaultdict(_empty_totals)

    def _get_score_dict(self, prediction):
        return {
            "timestamp_exact": prediction.get("timestamp_true_positives")
            == 2 * prediction.get("reference_words_count", 0)
            and prediction.get("wer_errors") == 0
        }

    def get_incorrect_sample(self, prediction):
        prediction = prediction.copy()
        references = int(prediction.get("reference_words_count", 0))
        prediction.update(
            wer_errors=references,
            wer_substitutions=0,
            wer_insertions=0,
            wer_deletions=references,
            hypothesis_words_count=0,
            timestamp_true_positives=0,
            timestamped_word_matches=0,
            start_error_seconds_sum=0.0,
            end_error_seconds_sum=0.0,
            malformed_timestamp_words=0,
        )
        return prediction

    def update(self, predictions):
        if len(predictions) != 1:
            raise ValueError("CTCTimestampMetrics expects exactly one prediction per sample")
        prediction = predictions[0]
        tolerance_seconds = float(prediction["timestamp_tolerance_seconds"])
        if self.tolerance_seconds is None:
            self.tolerance_seconds = tolerance_seconds
        elif not math.isclose(self.tolerance_seconds, tolerance_seconds):
            raise ValueError("CTCTimestampMetrics cannot aggregate predictions scored with different tolerances")
        super().update(predictions)
        _accumulate(self.corpus_totals, prediction)
        _accumulate(self.subset_totals[str(prediction.get("timestamp_subset", "unknown"))], prediction)

    def get_metrics(self):
        if self.total <= 0 or int(self.corpus_totals["reference_words"]) <= 0:
            raise ValueError("CTC timestamp evaluation has no reference words to score")
        assert self.tolerance_seconds is not None
        values = _reported_values(self.corpus_totals, include_counts=True)
        values["timestamp_tolerance_ms"] = round(1000.0 * self.tolerance_seconds, 3)
        for subset, totals in sorted(self.subset_totals.items()):
            for metric, value in _reported_values(totals, include_counts=False).items():
                values[f"{metric}::{subset}"] = value
        self.update_common_metrics(values)
        return {"pass@1": values}

    def evaluations_to_print(self):
        return ["pass@1"]

    def metrics_to_print(self):
        return {
            "wer": as_percentage,
            "timestamp_precision": as_percentage,
            "timestamp_recall": as_percentage,
            "timestamp_f1": as_percentage,
            "timestamp_tolerance_ms": as_float,
            "start_mae_ms": as_float,
            "end_mae_ms": as_float,
            "wer_errors": as_int,
            "wer_substitutions": as_int,
            "wer_insertions": as_int,
            "wer_deletions": as_int,
            "reference_words": as_int,
            "hypothesis_words": as_int,
            "timestamp_true_positives": as_int,
            "timestamp_reference_events": as_int,
            "timestamp_hypothesis_events": as_int,
            "timestamped_word_matches": as_int,
            "malformed_timestamp_words": as_int,
            "timestamp_unavailable_entries": as_int,
            "scored_entries": as_int,
            "num_entries": as_int,
        }
