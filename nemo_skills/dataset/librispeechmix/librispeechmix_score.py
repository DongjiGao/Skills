# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


def _aggregate_wer_counts(aggregated: dict, metrics_name: str, totals: dict) -> None:
    ref_words = totals.get("ref_words", 0)
    if ref_words <= 0:
        return
    errors = totals.get("errors", 0)
    aggregated[metrics_name] = round(100.0 * errors / ref_words, 2)
    aggregated[f"{metrics_name}_errors"] = errors
    aggregated[f"{metrics_name}_substitutions"] = totals.get("substitutions", 0)
    aggregated[f"{metrics_name}_insertions"] = totals.get("insertions", 0)
    aggregated[f"{metrics_name}_deletions"] = totals.get("deletions", 0)
    aggregated[f"{metrics_name}_ref_words"] = ref_words


def _aggregate_bucket(benchmarks: dict, eval_mode: str) -> dict:
    total_entries = 0
    total_ref_words = 0
    total_substitutions = 0
    total_insertions = 0
    total_deletions = 0
    weighted_success = 0.0
    weighted_no_answer = 0.0
    weighted_tokens = 0.0
    total_gen_seconds = 0
    structured_totals = {
        name: {"errors": 0, "ref_words": 0, "substitutions": 0, "insertions": 0, "deletions": 0}
        for name in ("strict_wer", "canonical_wer", "content_wer")
    }

    for benchmark_data in benchmarks.values():
        metrics = benchmark_data.get(eval_mode)
        if not metrics:
            continue

        num_entries = metrics.get("num_entries", 0)
        if num_entries <= 0:
            continue

        total_entries += num_entries
        weighted_success += metrics.get("success_rate", 0.0) * num_entries
        weighted_no_answer += metrics.get("no_answer", 0.0) * num_entries
        weighted_tokens += metrics.get("avg_tokens", 0.0) * num_entries
        total_gen_seconds += metrics.get("gen_seconds", 0)

        total_ref_words += metrics.get("ref_words", 0)
        total_substitutions += metrics.get("substitutions", 0)
        total_insertions += metrics.get("insertions", 0)
        total_deletions += metrics.get("deletions", 0)

        for metric_name, totals in structured_totals.items():
            totals["errors"] += metrics.get(f"{metric_name}_errors", 0)
            totals["ref_words"] += metrics.get(f"{metric_name}_ref_words", 0)
            totals["substitutions"] += metrics.get(f"{metric_name}_substitutions", 0)
            totals["insertions"] += metrics.get(f"{metric_name}_insertions", 0)
            totals["deletions"] += metrics.get(f"{metric_name}_deletions", 0)

    if total_entries == 0:
        return {}

    aggregated = {
        "avg_tokens": int(weighted_tokens / total_entries),
        "gen_seconds": total_gen_seconds,
        "success_rate": weighted_success / total_entries,
        "no_answer": weighted_no_answer / total_entries,
        "num_entries": total_entries,
    }

    if total_ref_words > 0:
        total_errors = total_substitutions + total_insertions + total_deletions
        aggregated["substitutions"] = total_substitutions
        aggregated["insertions"] = total_insertions
        aggregated["deletions"] = total_deletions
        aggregated["ref_words"] = total_ref_words
        aggregated["wer"] = round(100.0 * total_errors / total_ref_words, 2)

    for metric_name, totals in structured_totals.items():
        _aggregate_wer_counts(aggregated, metric_name, totals)

    return aggregated


def compute_score(combined_metrics: dict) -> dict:
    """Aggregate LibriSpeechMix metrics across all sub-benchmarks and by mode."""
    if not combined_metrics:
        return {}

    first_benchmark = next(iter(combined_metrics.values()))
    eval_modes = list(first_benchmark.keys())
    grouped = {
        "all": combined_metrics,
        "asr": {name: value for name, value in combined_metrics.items() if ".asr-" in name},
        "sa-asr": {name: value for name, value in combined_metrics.items() if ".sa-asr-" in name},
    }

    aggregated = {}
    for eval_mode in eval_modes:
        overall = _aggregate_bucket(grouped["all"], eval_mode)
        if not overall:
            continue

        asr_bucket = _aggregate_bucket(grouped["asr"], eval_mode)
        sa_asr_bucket = _aggregate_bucket(grouped["sa-asr"], eval_mode)

        if asr_bucket.get("wer") is not None:
            overall["asr_wer"] = asr_bucket["wer"]
        if sa_asr_bucket.get("wer") is not None:
            overall["sa_asr_wer"] = sa_asr_bucket["wer"]
        for metric_name in ("strict_wer", "canonical_wer", "content_wer"):
            if asr_bucket.get(metric_name) is not None:
                overall[f"asr_{metric_name}"] = asr_bucket[metric_name]
            if sa_asr_bucket.get(metric_name) is not None:
                overall[f"sa_asr_{metric_name}"] = sa_asr_bucket[metric_name]
        if asr_bucket.get("num_entries") is not None:
            overall["asr_num_entries"] = asr_bucket["num_entries"]
        if sa_asr_bucket.get("num_entries") is not None:
            overall["sa_asr_num_entries"] = sa_asr_bucket["num_entries"]

        aggregated[eval_mode] = overall

    return aggregated
