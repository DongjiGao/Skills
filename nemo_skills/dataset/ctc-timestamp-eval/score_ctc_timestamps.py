# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score CTC word timestamps against a ctc-timestamp-eval manifest."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

scorer = importlib.import_module("nemo_skills.dataset.ctc-timestamp-eval.scorer")


def _rows(path: Path) -> list[dict]:
    with path.open("rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _row_id(row: dict) -> str:
    for key in ("sample_id", "id", "audio_filepath"):
        value = row.get(key)
        if value:
            return Path(str(value)).stem if key == "audio_filepath" else str(value)
    raise ValueError("Every row must have sample_id, id, or audio_filepath")


def score_files(
    reference_path: Path,
    prediction_path: Path,
    tolerance_seconds: float,
    scored_output: Path | None = None,
) -> dict:
    references = {_row_id(row): row for row in _rows(reference_path)}
    predictions = {_row_id(row): row for row in _rows(prediction_path)}
    if set(references) != set(predictions):
        missing = sorted(set(references) - set(predictions))
        extra = sorted(set(predictions) - set(references))
        raise ValueError(f"Reference/prediction IDs differ; missing={missing[:10]}, extra={extra[:10]}")

    metrics = scorer.CTCTimestampMetrics(max_k=1)
    scored = []
    for sample_id, reference in references.items():
        row = dict(reference)
        row.update(predictions[sample_id])
        row["sample_id"] = sample_id
        result = scorer.evaluate_ctc_timestamp_sample(row, tolerance_seconds=tolerance_seconds)
        scored_row = {**row, **result}
        scored.append(scored_row)
        metrics.update([scored_row])

    if scored_output is not None:
        scored_output.parent.mkdir(parents=True, exist_ok=True)
        with scored_output.open("wt", encoding="utf-8") as stream:
            for row in scored:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return metrics.get_metrics()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Prepared test.jsonl")
    parser.add_argument("--predictions", type=Path, required=True, help="Prediction JSONL with word timestamps")
    parser.add_argument("--tolerance-seconds", type=float, default=scorer.DEFAULT_TOLERANCE_SECONDS)
    parser.add_argument("--scored-output", type=Path)
    parser.add_argument("--metrics-output", type=Path)
    args = parser.parse_args()
    metrics = score_files(args.reference, args.predictions, args.tolerance_seconds, args.scored_output)
    rendered = json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.metrics_output is not None:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
