# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Group score aggregation for LibriSpeechMix SOT."""

from __future__ import annotations


def _aggregate_metric_counts(output: dict, metric_name: str, totals: dict) -> None:
    ref_words = totals.get("ref_words", 0)
    output[f"{metric_name}_errors"] = totals.get("errors", 0)
    output[f"{metric_name}_substitutions"] = totals.get("substitutions", 0)
    output[f"{metric_name}_insertions"] = totals.get("insertions", 0)
    output[f"{metric_name}_deletions"] = totals.get("deletions", 0)
    output[f"{metric_name}_ref_words"] = ref_words
    if ref_words > 0:
        output[metric_name] = round(100.0 * totals.get("errors", 0) / ref_words, 2)


def _aggregate(benchmarks: dict, eval_mode: str) -> dict:
    total_entries = 0
    totals_by_metric = {
        name: {"errors": 0, "substitutions": 0, "insertions": 0, "deletions": 0, "ref_words": 0}
        for name in ("cpwer", "strict_cpwer", "canonical_cpwer", "content_wer")
    }

    for benchmark_metrics in benchmarks.values():
        metrics = benchmark_metrics.get(eval_mode)
        if not metrics:
            continue
        total_entries += metrics.get("num_entries", 0)
        for metric_name, totals in totals_by_metric.items():
            totals["errors"] += metrics.get(f"{metric_name}_errors", 0)
            totals["substitutions"] += metrics.get(f"{metric_name}_substitutions", 0)
            totals["insertions"] += metrics.get(f"{metric_name}_insertions", 0)
            totals["deletions"] += metrics.get(f"{metric_name}_deletions", 0)
            totals["ref_words"] += metrics.get(f"{metric_name}_ref_words", 0)

    if total_entries == 0:
        return {}

    output = {"num_entries": total_entries}
    for metric_name, totals in totals_by_metric.items():
        _aggregate_metric_counts(output, metric_name, totals)
    return output


def compute_score(combined_metrics: dict) -> dict:
    """Aggregate cpWER overall and by duration split."""
    if not combined_metrics:
        return {}

    first_benchmark = next(iter(combined_metrics.values()))
    eval_modes = list(first_benchmark.keys())
    groups = {
        "all": combined_metrics,
        "1mix": {k: v for k, v in combined_metrics.items() if k.endswith("-1mix")},
        "2mix": {k: v for k, v in combined_metrics.items() if k.endswith("-2mix")},
        "3mix": {k: v for k, v in combined_metrics.items() if k.endswith("-3mix")},
        "4mix": {k: v for k, v in combined_metrics.items() if k.endswith("-4mix")},
        "under20s": {k: v for k, v in combined_metrics.items() if ".under20s-" in k},
        "over20s": {k: v for k, v in combined_metrics.items() if ".over20s-" in k},
        "test_clean": {k: v for k, v in combined_metrics.items() if "-test-clean-" in k},
        "dev_clean": {k: v for k, v in combined_metrics.items() if "-dev-clean-" in k},
    }

    output = {}
    for eval_mode in eval_modes:
        mode_output = {}
        for group_name, group_metrics in groups.items():
            aggregate = _aggregate(group_metrics, eval_mode)
            if aggregate:
                mode_output[group_name] = aggregate
        if mode_output:
            output[eval_mode] = mode_output
    return output
