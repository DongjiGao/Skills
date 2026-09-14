# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Termination metrics for LoGenBench."""

from collections import Counter

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_int, as_percentage


class LoGenBenchMetrics(BaseMetrics):
    """Report how often a request exhausts its generation budget."""

    def __init__(self, compute_no_answer: bool = False, max_k: int = 1):
        if max_k != 1:
            raise ValueError("LoGenBench supports one deterministic generation per sample")
        super().__init__(compute_no_answer=compute_no_answer)
        self.max_k = 1
        self.unbounded = 0
        self.finish_reasons = Counter()

    def reset(self):
        super().reset()
        self.unbounded = 0
        self.finish_reasons = Counter()

    def _get_score_dict(self, prediction):
        return {"bounded": not self._is_unbounded(prediction)}

    @staticmethod
    def _finish_reasons(prediction):
        reasons = prediction.get("chunk_finish_reasons")
        if isinstance(reasons, list) and reasons:
            return [str(reason) for reason in reasons]
        reason = prediction.get("finish_reason")
        return [] if reason is None else [str(reason)]

    @classmethod
    def _is_unbounded(cls, prediction):
        return "length" in cls._finish_reasons(prediction)

    def update(self, predictions):
        if len(predictions) != 1:
            raise ValueError("LoGenBench expects exactly one prediction per sample")
        super().update(predictions)
        reasons = self._finish_reasons(predictions[0])
        self.finish_reasons.update(reasons or ["missing"])
        self.unbounded += int("length" in reasons)

    def get_metrics(self):
        values = {}
        self.update_common_metrics(values)
        values["unbounded_generation"] = round(100.0 * self.unbounded / self.total, 4) if self.total else 0.0
        values["unbounded_examples"] = self.unbounded
        values["bounded_examples"] = self.total - self.unbounded
        for reason, count in sorted(self.finish_reasons.items()):
            values[f"finish_reason_{reason}"] = count
        return {"pass@1": values}

    def get_incorrect_sample(self, prediction):
        return {**prediction, "finish_reason": "length"}

    def metrics_to_print(self):
        return {
            "num_entries": as_int,
            "unbounded_generation": as_percentage,
            "unbounded_examples": as_int,
            "bounded_examples": as_int,
            "avg_tokens": as_int,
            "gen_seconds": as_int,
        }

    def evaluations_to_print(self):
        return ["pass@1"]
