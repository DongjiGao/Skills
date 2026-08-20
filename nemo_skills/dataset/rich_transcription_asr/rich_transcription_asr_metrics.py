# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Corpus metrics for Rich Transcription ASR token scopes."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from nemo_skills.evaluation.metrics.audio_metrics import AudioMetrics

EVENT_RE = re.compile(r"<ev:[^>]+>")
STYLE_RE = re.compile(r"<sty:(?:stt|end):[^>]+>")
SPECIAL_RE = re.compile(r"<(?:ev:[^>]+|sty:(?:stt|end):[^>]+)>")
SPACED_SPECIAL_RE = re.compile(r"<\s*(ev|sty)\s*:\s*([^<>]+?)\s*>")
IGNORED_SPECIAL_TOKENS = frozenset({"<sty:stt:flat>", "<sty:end:flat>"})
SPECIAL_SCOPES = ("event", "style", "all_special")

KNOWN_EVENT_TOKENS = frozenset(
    {
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
    }
)


def canonicalize_special_spacing(text: str) -> str:
    """Fold decoder variants such as ``< ev : breath >`` into canonical tags."""

    def replace(match: re.Match[str]) -> str:
        content = re.sub(r"\s+", "", match.group(1) + ":" + match.group(2))
        return f"<{content}>"

    return SPACED_SPECIAL_RE.sub(replace, text or "")


def extract_tokens(
    text: str,
    token_type: str = "event",
    canonicalize: bool = True,
    exclude_flat: bool = True,
) -> list[str]:
    """Extract one tag family in order of appearance."""
    if canonicalize:
        text = canonicalize_special_spacing(text)
    if token_type == "event":
        tokens = EVENT_RE.findall(text or "")
    elif token_type == "style":
        tokens = STYLE_RE.findall(text or "")
    elif token_type == "all_special":
        tokens = SPECIAL_RE.findall(text or "")
    else:
        raise ValueError(
            f"Unknown token_type {token_type!r}; expected one of {SPECIAL_SCOPES}"
        )
    if exclude_flat:
        tokens = [token for token in tokens if token not in IGNORED_SPECIAL_TOKENS]
    return tokens


@dataclass
class EditCounts:
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.insertions + self.deletions

    def __add__(self, other: "EditCounts") -> "EditCounts":
        return EditCounts(
            substitutions=self.substitutions + other.substitutions,
            insertions=self.insertions + other.insertions,
            deletions=self.deletions + other.deletions,
        )


@dataclass
class TokenStats:
    ref_tokens: int = 0
    hyp_tokens: int = 0
    correct: int = 0
    substitutions_as_ref: int = 0
    substitutions_as_hyp: int = 0
    deletions: int = 0
    insertions: int = 0

    @property
    def precision(self) -> float | None:
        return None if self.hyp_tokens == 0 else self.correct / self.hyp_tokens

    @property
    def recall(self) -> float | None:
        return None if self.ref_tokens == 0 else self.correct / self.ref_tokens

    @property
    def f1(self) -> float | None:
        precision = self.precision
        recall = self.recall
        if precision is None or recall is None:
            return None
        return (
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )


def align_tokens(
    reference: Sequence[str], hypothesis: Sequence[str]
) -> list[tuple[str, str | None, str | None]]:
    """Align tags with the SoundFormer substitution/deletion/insertion tie-break."""
    rows, columns = len(reference) + 1, len(hypothesis) + 1
    costs = [[0] * columns for _ in range(rows)]
    backtrace: list[list[str | None]] = [[None] * columns for _ in range(rows)]

    for i in range(1, rows):
        costs[i][0], backtrace[i][0] = i, "deletion"
    for j in range(1, columns):
        costs[0][j], backtrace[0][j] = j, "insertion"

    for i in range(1, rows):
        for j in range(1, columns):
            if reference[i - 1] == hypothesis[j - 1]:
                costs[i][j], backtrace[i][j] = costs[i - 1][j - 1], "equal"
                continue
            cost, operation, _ = min(
                (
                    (costs[i - 1][j - 1] + 1, "substitution", 0),
                    (costs[i - 1][j] + 1, "deletion", 1),
                    (costs[i][j - 1] + 1, "insertion", 2),
                ),
                key=lambda item: (item[0], item[2]),
            )
            costs[i][j], backtrace[i][j] = cost, operation

    alignment = []
    i, j = len(reference), len(hypothesis)
    while i > 0 or j > 0:
        operation = backtrace[i][j]
        if operation in ("equal", "substitution"):
            alignment.append((operation, reference[i - 1], hypothesis[j - 1]))
            i, j = i - 1, j - 1
        elif operation == "deletion":
            alignment.append((operation, reference[i - 1], None))
            i -= 1
        elif operation == "insertion":
            alignment.append((operation, None, hypothesis[j - 1]))
            j -= 1
        else:
            raise RuntimeError("Invalid special-token edit backtrace state")
    alignment.reverse()
    return alignment


def edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> EditCounts:
    counts = EditCounts()
    for operation, _, _ in align_tokens(reference, hypothesis):
        if operation == "substitution":
            counts.substitutions += 1
        elif operation == "deletion":
            counts.deletions += 1
        elif operation == "insertion":
            counts.insertions += 1
    return counts


def update_token_stats(
    stats: dict[str, TokenStats], reference: Sequence[str], hypothesis: Sequence[str]
) -> None:
    for token in reference:
        stats[token].ref_tokens += 1
    for token in hypothesis:
        stats[token].hyp_tokens += 1
    for operation, ref_token, hyp_token in align_tokens(reference, hypothesis):
        if operation == "equal" and ref_token is not None:
            stats[ref_token].correct += 1
        elif operation == "substitution":
            if ref_token is not None:
                stats[ref_token].substitutions_as_ref += 1
            if hyp_token is not None:
                stats[hyp_token].substitutions_as_hyp += 1
        elif operation == "deletion" and ref_token is not None:
            stats[ref_token].deletions += 1
        elif operation == "insertion" and hyp_token is not None:
            stats[hyp_token].insertions += 1


def _token_stat_rows(stats: dict[str, TokenStats]) -> list[dict[str, Any]]:
    rows = []
    for token, token_stats in stats.items():
        rows.append(
            {
                "token": token,
                "ref_tokens": token_stats.ref_tokens,
                "hyp_tokens": token_stats.hyp_tokens,
                "correct": token_stats.correct,
                "deletions": token_stats.deletions,
                "insertions": token_stats.insertions,
                "substitutions_as_ref": token_stats.substitutions_as_ref,
                "substitutions_as_hyp": token_stats.substitutions_as_hyp,
                "precision": token_stats.precision,
                "recall": token_stats.recall,
                "f1": token_stats.f1,
            }
        )
    rows.sort(key=lambda row: (-row["ref_tokens"], row["token"]))
    return rows


@dataclass
class EventMetrics:
    """Corpus-level tag score plus complete per-token accounting."""

    wer: float | None = None
    errors: int = 0
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0
    ref_tokens: int = 0
    hyp_tokens: int = 0
    per_token: list[dict[str, Any]] = field(default_factory=list)
    unknown_tokens: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "wer": self.wer,
            "wer_percent": None if self.wer is None else 100.0 * self.wer,
            "errors": self.errors,
            "substitutions": self.substitutions,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "ref_tokens": self.ref_tokens,
            "hyp_tokens": self.hyp_tokens,
            "per_token": self.per_token,
            "unknown_tokens": self.unknown_tokens,
        }


def score_event_tokens(
    references: Sequence[str],
    hypotheses: Sequence[str],
    canonicalize: bool = True,
    token_type: str = "event",
    exclude_flat: bool = True,
) -> EventMetrics:
    """Score an event, style, or jointly ordered all-special stream."""
    if len(references) != len(hypotheses):
        raise ValueError(
            f"Got {len(references)} references but {len(hypotheses)} hypotheses"
        )

    totals = EditCounts()
    stats: dict[str, TokenStats] = defaultdict(TokenStats)
    ref_count = 0
    hyp_count = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        ref_tokens = extract_tokens(reference, token_type, canonicalize, exclude_flat)
        hyp_tokens = extract_tokens(hypothesis, token_type, canonicalize, exclude_flat)
        totals = totals + edit_counts(ref_tokens, hyp_tokens)
        ref_count += len(ref_tokens)
        hyp_count += len(hyp_tokens)
        update_token_stats(stats, ref_tokens, hyp_tokens)

    return EventMetrics(
        wer=None if ref_count == 0 else totals.errors / ref_count,
        errors=totals.errors,
        substitutions=totals.substitutions,
        insertions=totals.insertions,
        deletions=totals.deletions,
        ref_tokens=ref_count,
        hyp_tokens=hyp_count,
        per_token=_token_stat_rows(stats),
        unknown_tokens=(
            sorted(token for token in stats if token not in KNOWN_EVENT_TOKENS)
            if token_type == "event"
            else []
        ),
    )


def evaluate_special_tokens(reference: str, hypothesis: str) -> dict[str, Any]:
    """Return additive per-sample fields for all three special-token scopes."""
    updates = {}
    for scope in SPECIAL_SCOPES:
        metrics = score_event_tokens([reference], [hypothesis], token_type=scope)
        updates[f"{scope}_wer"] = metrics.wer
        updates[f"{scope}_errors"] = metrics.errors
        updates[f"{scope}_substitutions"] = metrics.substitutions
        updates[f"{scope}_insertions"] = metrics.insertions
        updates[f"{scope}_deletions"] = metrics.deletions
        updates[f"{scope}_ref_tokens"] = metrics.ref_tokens
        updates[f"{scope}_hyp_tokens"] = metrics.hyp_tokens
    return updates


class RichTranscriptionASRMetrics(AudioMetrics):
    """Aggregate corpus WER for the evaluator's selected token scope."""

    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        super().__init__(compute_no_answer=compute_no_answer, max_k=max_k)
        self.evaluation_modes: set[str] = set()

    @staticmethod
    def _ensure_scored(prediction: dict[str, Any]) -> dict[str, Any]:
        prediction = dict(prediction)
        if "wer_errors" in prediction:
            return prediction

        from importlib import import_module

        evaluator = import_module(
            "nemo_skills.dataset.rich_transcription_asr.rich_transcription_asr_eval"
        )
        prediction.update(
            evaluator.evaluate_rich_transcription_asr_sample(
                prediction,
                prediction.get("evaluation_mode", "event"),
            )
        )
        return prediction

    def update(self, predictions):
        scored_predictions = [
            self._ensure_scored(prediction) for prediction in predictions
        ]
        super().update(scored_predictions)
        self.evaluation_modes.update(
            str(prediction.get("evaluation_mode", "event"))
            for prediction in scored_predictions
        )

    def get_metrics(self):
        metrics_dict = super().get_metrics()
        if len(self.evaluation_modes) > 1:
            raise ValueError(
                f"Cannot aggregate mixed evaluation modes: {sorted(self.evaluation_modes)}"
            )
        evaluation_mode = next(iter(self.evaluation_modes), "event")
        for aggregate_metrics in metrics_dict.values():
            if self.wer_total_ref_words > 0:
                aggregate_metrics["wer_errors"] = self.wer_total_errors
            aggregate_metrics["evaluation_mode"] = evaluation_mode
        return metrics_dict


def write_token_stats_tsv(results: dict[str, EventMetrics], path: Path) -> None:
    """Write the source evaluator's complete per-token accounting table."""
    fieldnames = [
        "metric_scope",
        "token",
        "ref_tokens",
        "hyp_tokens",
        "correct",
        "deletions",
        "insertions",
        "substitutions_as_ref",
        "substitutions_as_hyp",
        "recall_percent",
        "precision_percent",
        "f1_percent",
    ]
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for scope in SPECIAL_SCOPES:
            for row in results[scope].per_token:
                writer.writerow(
                    {
                        "metric_scope": scope,
                        "token": row["token"],
                        "ref_tokens": row["ref_tokens"],
                        "hyp_tokens": row["hyp_tokens"],
                        "correct": row["correct"],
                        "deletions": row["deletions"],
                        "insertions": row["insertions"],
                        "substitutions_as_ref": row["substitutions_as_ref"],
                        "substitutions_as_hyp": row["substitutions_as_hyp"],
                        "recall_percent": (
                            None if row["recall"] is None else 100.0 * row["recall"]
                        ),
                        "precision_percent": (
                            None
                            if row["precision"] is None
                            else 100.0 * row["precision"]
                        ),
                        "f1_percent": None if row["f1"] is None else 100.0 * row["f1"],
                    }
                )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("rt", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def score_prediction_rows(
    rows: Iterable[dict[str, Any]],
    reference_key: str,
    hypothesis_key: str,
    evaluation_mode: str = "event",
) -> dict[str, Any]:
    """Score one selected token scope in a predictions JSONL."""
    transcript_totals = {
        "errors": 0,
        "substitutions": 0,
        "insertions": 0,
        "deletions": 0,
        "ref_words": 0,
    }

    from importlib import import_module

    evaluator = import_module(
        "nemo_skills.dataset.rich_transcription_asr.rich_transcription_asr_eval"
    )
    for row in rows:
        reference = str(row.get(reference_key, "") or "")
        hypothesis = str(row.get(hypothesis_key, "") or "")
        scored = evaluator.evaluate_rich_transcription_asr_text(
            reference, hypothesis, evaluation_mode
        )
        transcript_totals["errors"] += int(scored["wer_errors"])
        transcript_totals["substitutions"] += int(scored["wer_substitutions"])
        transcript_totals["insertions"] += int(scored["wer_insertions"])
        transcript_totals["deletions"] += int(scored["wer_deletions"])
        transcript_totals["ref_words"] += int(scored["wer_ref_words"])

    return {
        "evaluation_mode": evaluator.normalize_evaluation_mode(evaluation_mode),
        **transcript_totals,
        "wer": (
            None
            if transcript_totals["ref_words"] == 0
            else transcript_totals["errors"] / transcript_totals["ref_words"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a NeMo-Skills Sound Expert predictions JSONL."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--reference-key", default="expected_answer")
    parser.add_argument("--hypothesis-key", default="generation")
    parser.add_argument(
        "--evaluation-mode",
        choices=("event", "style", "event_style", "all", "1", "2", "3", "4"),
        default="event",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    output = score_prediction_rows(
        _read_jsonl(args.predictions),
        args.reference_key,
        args.hypothesis_key,
        args.evaluation_mode,
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))
    if args.json_out:
        args.json_out.write_text(
            json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
