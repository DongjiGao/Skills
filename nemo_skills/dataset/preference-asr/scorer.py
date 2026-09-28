# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Preference-aware corpus WER for the private seven-corpus release.

Shares the preference-asr-public normalizer and edit-distance helper.
That implementation is parity-checked against this release's author code; its
four-corpus manifest and public dataset validation are deliberately not reused.
"""

from collections import defaultdict
from importlib import import_module

from nemo_skills.evaluation.evaluator.base import BaseEvaluator, BaseEvaluatorConfig
from nemo_skills.evaluation.metrics.base import BaseMetrics, as_int, as_percentage
from nemo_skills.utils import nested_dataclass

preference_normalizer = import_module("nemo_skills.dataset.preference-asr-public.normalizer").preference_normalizer
word_error_counts = import_module("nemo_skills.dataset.preference-asr-public.scorer").word_error_counts
SCORING_VERSION = "private-preference-aware-v1"
PREFERENCE_TYPES = {"normalization", "entities", "disfluencies", "case", "standard"}


def evaluate_sample(sample):
    if sample.get("preference_type") not in PREFERENCE_TYPES:
        raise ValueError(f"Unknown preference_type: {sample.get('preference_type')!r}")
    normalizer = preference_normalizer()
    reference = normalizer.normalize_entry(sample["expected_answer"], sample)
    hypothesis = normalizer.normalize_entry(sample["generation"], sample)
    counts = word_error_counts(reference.split(), hypothesis.split())
    return {
        "preference_scoring_version": SCORING_VERSION,
        "text": reference,
        "pred_text": hypothesis,
        "wer": counts["errors"] / counts["ref_words"] if counts["ref_words"] else float(counts["errors"]),
        **{f"wer_{key}": value for key, value in counts.items()},
        "is_correct": counts["errors"] == 0,
    }


@nested_dataclass(kw_only=True)
class PreferenceASREvaluatorConfig(BaseEvaluatorConfig):
    pass


class PreferenceASREvaluator(BaseEvaluator):
    async def eval_single(self, data_point):
        return evaluate_sample(data_point)


class PreferenceASRMetrics(BaseMetrics):
    def __init__(self, compute_no_answer=True, max_k=1):
        if max_k != 1:
            raise ValueError("Preference ASR supports one prediction per example")
        self.max_k = 1
        super().__init__(compute_no_answer=compute_no_answer)

    def reset(self):
        super().reset()
        self.counts = defaultdict(lambda: defaultdict(int))

    def _get_score_dict(self, prediction):
        return {"preference_exact": prediction["wer_errors"] == 0}

    def get_incorrect_sample(self, prediction):
        # Preserve a valid scoring schema while simulating an unanswered sample.
        sample = {**prediction, "generation": ""}
        return {**sample, **evaluate_sample(sample)}

    def update(self, predictions):
        if len(predictions) != 1:
            raise ValueError("Preference ASR expects exactly one prediction")
        prediction = predictions[0]
        if prediction.get("preference_scoring_version") != SCORING_VERSION:
            raise ValueError("Re-score with the preference-aware evaluator; legacy ContextASR counts are incompatible")
        super().update(predictions)
        for group in (
            "overall",
            f"type:{prediction['preference_type']}",
            f"dataset:{prediction['subset_for_metrics']}",
        ):
            target = self.counts[group]
            for key in ("errors", "ref_words", "substitutions", "insertions", "deletions"):
                target[key] += prediction[f"wer_{key}"]
            target["entries"] += 1

    def get_metrics(self):
        counts = self.counts["overall"]
        if counts["ref_words"] <= 0:
            raise ValueError("Preference ASR aggregate has no reference words")
        values = {
            "wer": round(100 * counts["errors"] / counts["ref_words"], 2),
            "preference_scoring_version": SCORING_VERSION,
            **{f"wer_{key}": value for key, value in counts.items()},
        }
        for group, count in self.counts.items():
            if group != "overall" and count["ref_words"]:
                values[f"wer_{group.replace(':', '_')}"] = round(100 * count["errors"] / count["ref_words"], 2)
        self.update_common_metrics(values)
        return {"pass@1": values}

    def evaluations_to_print(self):
        return ["pass@1"]

    def metrics_to_print(self):
        return {"wer": as_percentage, "wer_errors": as_int, "wer_ref_words": as_int, "num_entries": as_int}
