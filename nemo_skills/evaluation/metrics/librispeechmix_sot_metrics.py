# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Metrics aggregation for LibriSpeechMix SOT cpWER."""

from __future__ import annotations

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_float, as_int, as_percentage


class LibriSpeechMixSOTMetrics(BaseMetrics):
    """Aggregate LibriSpeechMix SOT cpWER from raw corpus counts."""

    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        super().__init__(compute_no_answer=compute_no_answer)
        self.max_k = max_k

    def _get_score_dict(self, prediction: dict) -> dict[str, bool]:
        return {"correct": bool(prediction.get("is_correct", False))}

    def get_incorrect_sample(self, prediction: dict) -> dict:
        prediction = prediction.copy()
        prediction["is_correct"] = False
        ref_words = prediction.get("cpwer_ref_words", 0)
        for metric_name in ("cpwer", "strict_cpwer", "canonical_cpwer"):
            prediction[f"{metric_name}_errors"] = prediction.get(f"{metric_name}_ref_words", ref_words)
            prediction[f"{metric_name}_deletions"] = prediction.get(f"{metric_name}_ref_words", ref_words)
            prediction[f"{metric_name}_insertions"] = 0
            prediction[f"{metric_name}_substitutions"] = 0
        if "content_wer_ref_words" in prediction:
            prediction["content_wer_errors"] = prediction.get("content_wer_ref_words", 0)
            prediction["content_wer_deletions"] = prediction.get("content_wer_ref_words", 0)
            prediction["content_wer_insertions"] = 0
            prediction["content_wer_substitutions"] = 0
        return prediction

    def update(self, predictions: list[dict]) -> None:
        super().update(predictions)
        predicted_answers = [p.get("generation") or p.get("pred_text") or None for p in predictions]
        self._compute_pass_at_k(predictions=predictions, predicted_answers=predicted_answers)
        if len(predictions) > 1:
            self._compute_majority_at_k(predictions=predictions, predicted_answers=predicted_answers)

        first = predictions[0]
        raw_metric_names = ["cpwer", "strict_cpwer", "canonical_cpwer", "content_wer"]
        raw_suffixes = ["errors", "substitutions", "insertions", "deletions", "ref_words"]
        for mode in ["pass@1", "pass@1[avg-of-1]"]:
            for metric_name in raw_metric_names:
                for suffix in raw_suffixes:
                    key = f"{metric_name}_{suffix}"
                    self.eval_dict[mode][key] += first.get(key, 0)

    def get_metrics(self) -> dict:
        metrics_dict = super().get_metrics()
        raw_metric_names = ["cpwer", "strict_cpwer", "canonical_cpwer", "content_wer"]
        raw_suffixes = ["errors", "substitutions", "insertions", "deletions", "ref_words"]
        for mode, metrics in metrics_dict.items():
            raw_metrics = self.eval_dict.get(mode, {})
            for metric_name in raw_metric_names:
                for suffix in raw_suffixes:
                    key = f"{metric_name}_{suffix}"
                    metrics[key] = int(raw_metrics.get(key, 0))
                ref_words = metrics.get(f"{metric_name}_ref_words", 0)
                errors = metrics.get(f"{metric_name}_errors", 0)
                if ref_words > 0:
                    metrics[metric_name] = round(100.0 * errors / ref_words, 2)
        return metrics_dict

    def evaluations_to_print(self) -> list[str]:
        return [f"pass@{self.max_k}"]

    def metrics_to_print(self) -> dict:
        metrics = {
            "correct": as_percentage,
            "cpwer": as_float,
            "cpwer_errors": as_int,
            "cpwer_substitutions": as_int,
            "cpwer_insertions": as_int,
            "cpwer_deletions": as_int,
            "cpwer_ref_words": as_int,
            "strict_cpwer": as_float,
            "strict_cpwer_errors": as_int,
            "strict_cpwer_substitutions": as_int,
            "strict_cpwer_insertions": as_int,
            "strict_cpwer_deletions": as_int,
            "strict_cpwer_ref_words": as_int,
            "canonical_cpwer": as_float,
            "canonical_cpwer_errors": as_int,
            "canonical_cpwer_substitutions": as_int,
            "canonical_cpwer_insertions": as_int,
            "canonical_cpwer_deletions": as_int,
            "canonical_cpwer_ref_words": as_int,
            "content_wer": as_float,
            "content_wer_errors": as_int,
            "content_wer_substitutions": as_int,
            "content_wer_insertions": as_int,
            "content_wer_deletions": as_int,
            "content_wer_ref_words": as_int,
            "num_entries": as_int,
        }
        if self.compute_no_answer:
            metrics["no_answer"] = as_percentage
        return metrics
