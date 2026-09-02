# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Convenience aggregation for the four official Spoken-MQA leaf scores."""

LEAVES = {
    "short_digit",
    "long_digit",
    "single_step_reasoning",
    "multi_step_reasoning",
}


def compute_score(combined_metrics: dict) -> dict:
    """Entry-weight the official per-leaf accuracies into a convenience total."""
    benchmarks = {name: values for name, values in combined_metrics.items() if name.split(".")[-1] in LEAVES}
    if not benchmarks:
        return {}

    eval_modes = next(iter(benchmarks.values())).keys()
    aggregated = {}
    for eval_mode in eval_modes:
        total_entries = 0
        weighted = {key: 0.0 for key in ("accuracy", "success_rate", "avg_tokens", "no_answer")}
        seen = {key: False for key in weighted}
        gen_seconds = 0
        gen_seconds_seen = False
        for benchmark in benchmarks.values():
            metrics = benchmark.get(eval_mode)
            if not metrics:
                continue
            entries = metrics.get("num_entries", 0)
            total_entries += entries
            for key in weighted:
                if key in metrics:
                    weighted[key] += metrics[key] * entries
                    seen[key] = True
            if "gen_seconds" in metrics:
                gen_seconds += metrics["gen_seconds"]
                gen_seconds_seen = True

        if not total_entries:
            continue
        result = {"num_entries": total_entries}
        for key, value in weighted.items():
            if seen[key]:
                mean = value / total_entries
                result[key] = int(mean) if key == "avg_tokens" else mean
        if gen_seconds_seen:
            result["gen_seconds"] = gen_seconds
        aggregated[eval_mode] = result

    return aggregated
