# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Group score aggregation for multi-speaker SOT benchmarks."""

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
        "cpwer": {"errors": 0, "substitutions": 0, "insertions": 0, "deletions": 0, "ref_words": 0}
    }

    for benchmark_metrics in benchmarks.values():
        metrics = benchmark_metrics.get(eval_mode)
        if not metrics:
            continue
        total_entries += metrics.get("num_entries", 0)
        totals = totals_by_metric["cpwer"]
        totals["errors"] += metrics.get("cpwer_errors", 0)
        totals["substitutions"] += metrics.get("cpwer_substitutions", 0)
        totals["insertions"] += metrics.get("cpwer_insertions", 0)
        totals["deletions"] += metrics.get("cpwer_deletions", 0)
        totals["ref_words"] += metrics.get("cpwer_ref_words", 0)

    if total_entries == 0:
        return {}

    output = {"num_entries": total_entries}
    for metric_name, totals in totals_by_metric.items():
        _aggregate_metric_counts(output, metric_name, totals)
    return output


def _select_eval_group(combined_metrics: dict, group_name: str) -> dict:
    """Select a merged group or its legacy per-set benchmark entries."""
    prefix = f"multispeaker-sot.{group_name}"
    return {
        name: metrics
        for name, metrics in combined_metrics.items()
        if name == prefix or name.startswith(f"{prefix}-")
    }


def compute_score(combined_metrics: dict) -> dict:
    """Aggregate cpWER overall and by dataset/mix, collapsing duration splits."""
    if not combined_metrics:
        return {}

    first_benchmark = next(iter(combined_metrics.values()))
    eval_modes = list(first_benchmark.keys())
    groups = {
        "all": combined_metrics,
        # Duration buckets
        "under20s": _select_eval_group(combined_metrics, "under20s"),
        "60-90s": _select_eval_group(combined_metrics, "60-90s"),
        "120-180s": _select_eval_group(combined_metrics, "120-180s"),
        "full": _select_eval_group(combined_metrics, "full"),
        "lsm": _select_eval_group(combined_metrics, "lsm"),
        "turing-diarization-v1": _select_eval_group(combined_metrics, "turing-diarization-v1"),
        "mlc-slm-2-dev-13L": _select_eval_group(combined_metrics, "mlc-slm-2-dev-13L"),
        "notsofar1-eval-ihm": _select_eval_group(combined_metrics, "notsofar1-eval-ihm"),
        "notsofar1-eval-sc": _select_eval_group(combined_metrics, "notsofar1-eval-sc"),
        # Per-corpus buckets (collapsed across duration splits)
        "ami-ihm-dev": {k: v for k, v in combined_metrics.items() if "-ami-ihm-dev" in k},
        "ami-ihm-test": {k: v for k, v in combined_metrics.items() if "-ami-ihm-test" in k},
        "ami-sdm-test": {k: v for k, v in combined_metrics.items() if "-ami-sdm-test" in k},
        "chaes-ch109-test": {k: v for k, v in combined_metrics.items() if "-chaes-ch109-test" in k},
        "libri-am-dev": {k: v for k, v in combined_metrics.items() if "-libri-am-dev" in k},
        "lsm-test-clean": {k: v for k, v in combined_metrics.items() if "lsm-test-clean-" in k},
        "mixer6-ch04-test": {k: v for k, v in combined_metrics.items() if "-mixer6-ch04-test" in k},
        # LibriSpeechMix speaker-count buckets
        "1mix": {k: v for k, v in combined_metrics.items() if "-1mix" in k},
        "2mix": {k: v for k, v in combined_metrics.items() if "-2mix" in k},
        "3mix": {k: v for k, v in combined_metrics.items() if "-3mix" in k},
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
