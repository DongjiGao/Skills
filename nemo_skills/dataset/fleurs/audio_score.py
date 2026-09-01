# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


def _subset_scores(metrics: dict, metric: str) -> dict[str, float]:
    prefix = f"{metric}::"
    count_suffixes = (
        "_errors",
        "_ref_words",
        "_ref_chars",
        "_substitutions",
        "_insertions",
        "_deletions",
        "_num_entries",
    )
    return {
        name.removeprefix(prefix): float(value)
        for name, value in metrics.items()
        if name.startswith(prefix) and not name.endswith(count_suffixes)
    }


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 2) if values else None


def compute_score(combined_metrics: dict) -> dict:
    """Aggregate metrics from audio benchmark groups with asr and st sub-benchmarks.

    Existing row-weighted fields are retained for backward compatibility.
    Explicit language-/locale-macro ASR and pair-macro ST fields are computed
    from corpus metrics within each ``subset_for_metrics`` group. Per-subset
    values are copied through so a headline never hides its contributing set.
    """
    main_names = ["asr", "st"]
    benchmarks = {k: v for k, v in combined_metrics.items() if k.split(".")[-1] in main_names}

    if not benchmarks:
        return {}

    weighted_metrics = ["wer", "wer_macro", "cer", "cer_macro", "bleu", "comet"]
    summed_metrics = ["substitutions", "insertions", "deletions", "ref_words"]

    first_benchmark = next(iter(benchmarks.values()))
    eval_modes = list(first_benchmark.keys())

    aggregated = {}
    for eval_mode in eval_modes:
        total_entries = 0
        total_gen_seconds = 0
        weighted_success = 0.0
        weighted_tokens = 0.0
        weighted_no_answer = 0.0
        weighted_sums = {m: 0.0 for m in weighted_metrics}
        weighted_counts = {m: 0 for m in weighted_metrics}
        sums = {m: 0 for m in summed_metrics}

        language_wer_scores: dict[str, float] = {}
        language_cer_scores: dict[str, float] = {}
        pair_bleu_scores: dict[str, float] = {}

        for benchmark_name, benchmark_data in benchmarks.items():
            if eval_mode not in benchmark_data:
                continue

            metrics = benchmark_data[eval_mode]
            num_entries = metrics.get("num_entries", 0)
            if num_entries == 0:
                continue

            total_entries += num_entries
            total_gen_seconds += metrics.get("gen_seconds", 0)
            weighted_success += metrics.get("success_rate", 0.0) * num_entries
            weighted_tokens += metrics.get("avg_tokens", 0.0) * num_entries
            weighted_no_answer += metrics.get("no_answer", 0.0) * num_entries

            for m in weighted_metrics:
                if m in metrics:
                    weighted_sums[m] += metrics[m] * num_entries
                    weighted_counts[m] += num_entries

            for m in summed_metrics:
                if m in metrics:
                    sums[m] += metrics[m]

            if benchmark_name.endswith(".asr"):
                language_wer_scores.update(_subset_scores(metrics, "wer"))
                language_cer_scores.update(_subset_scores(metrics, "cer"))
            elif benchmark_name.endswith(".st"):
                pair_bleu_scores.update(_subset_scores(metrics, "bleu"))

        if total_entries == 0:
            continue

        agg = {
            "avg_tokens": int(weighted_tokens / total_entries),
            "gen_seconds": total_gen_seconds,
            "success_rate": weighted_success / total_entries,
            "no_answer": weighted_no_answer / total_entries,
            "num_entries": total_entries,
        }

        for m in weighted_metrics:
            if weighted_counts[m] > 0:
                agg[m] = round(weighted_sums[m] / weighted_counts[m], 2)

        for m in summed_metrics:
            if sums[m]:
                agg[m] = sums[m]

        for subset, value in sorted(language_wer_scores.items()):
            agg[f"wer::{subset}"] = value
        for subset, value in sorted(language_cer_scores.items()):
            agg[f"cer::{subset}"] = value
        for pair, value in sorted(pair_bleu_scores.items()):
            agg[f"bleu::{pair}"] = value

        if (score := _mean(list(language_wer_scores.values()))) is not None:
            agg["wer_language_macro"] = score
            agg["wer_language_count"] = len(language_wer_scores)
        if (score := _mean(list(language_cer_scores.values()))) is not None:
            agg["cer_language_macro"] = score
            agg["cer_language_count"] = len(language_cer_scores)
        if (score := _mean(list(pair_bleu_scores.values()))) is not None:
            agg["bleu_pair_macro"] = score
            agg["bleu_pair_count"] = len(pair_bleu_scores)

        en_to_x = []
        x_to_en = []
        for pair, value in pair_bleu_scores.items():
            source, separator, target = pair.partition("->")
            if not separator:
                continue
            if source in {"en", "en_us"}:
                en_to_x.append(value)
            if target in {"en", "en_us"}:
                x_to_en.append(value)
        if (score := _mean(en_to_x)) is not None:
            agg["bleu_en_to_x_macro"] = score
            agg["bleu_en_to_x_count"] = len(en_to_x)
        if (score := _mean(x_to_en)) is not None:
            agg["bleu_x_to_en_macro"] = score
            agg["bleu_x_to_en_count"] = len(x_to_en)

        aggregated[eval_mode] = agg

    return aggregated
