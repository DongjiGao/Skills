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

"""Metrics aggregation for HALAS."""

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_float, as_int, as_percentage


class HALASMetrics(BaseMetrics):
    """Corpus-level WER plus insertion-based hallucination proxy metrics."""

    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        super().__init__(compute_no_answer=compute_no_answer)
        if max_k != 1:
            raise ValueError("HALASMetrics supports only max_k=1. Run HALAS with one greedy generation.")
        self.max_k = 1
        self.wer_total_errors = 0
        self.wer_total_ref_words = 0
        self.substitutions = 0
        self.insertions = 0
        self.deletions = 0
        self.insertion_samples = 0
        self.pred_words = 0
        self.audio_seconds = 0.0

    def reset(self):
        super().reset()
        self.wer_total_errors = 0
        self.wer_total_ref_words = 0
        self.substitutions = 0
        self.insertions = 0
        self.deletions = 0
        self.insertion_samples = 0
        self.pred_words = 0
        self.audio_seconds = 0.0

    def _get_score_dict(self, prediction):
        return {"correct": bool(prediction.get("is_correct"))}

    def get_incorrect_sample(self, prediction):
        prediction = prediction.copy()
        prediction["is_correct"] = False
        if not prediction.get("generation", "").strip():
            prediction["generation"] = None
        return prediction

    def update_common_metrics(self, agg_dict):
        agg_dict["num_entries"] = self.total
        agg_dict["avg_tokens"] = int(self.avg_tokens / self.total) if self.total > 0 else 0
        if self.max_end_time > float("-inf") and self.min_start_time < float("inf"):
            agg_dict["gen_seconds"] = int(self.max_end_time - self.min_start_time)

    def update(self, predictions):
        if len(predictions) != 1:
            raise ValueError("HALASMetrics expects exactly one prediction per sample.")
        super().update(predictions)
        pred = predictions[0]

        self.wer_total_errors += int(pred.get("wer_errors") or 0)
        self.wer_total_ref_words += int(pred.get("wer_ref_words") or 0)
        self.substitutions += int(pred.get("wer_substitutions") or 0)
        self.insertions += int(pred.get("wer_insertions") or 0)
        self.deletions += int(pred.get("wer_deletions") or 0)
        self.insertion_samples += int(bool(pred.get("has_insertion")))
        self.pred_words += int(pred.get("pred_words") or 0)
        self.audio_seconds += float(pred.get("audio_duration") or 0.0)

        predicted_answers = [(pred.get("generation") or "").strip() or None]
        self._compute_pass_at_k(predictions=predictions, predicted_answers=predicted_answers)
        self._compute_majority_at_k(predictions=predictions, predicted_answers=predicted_answers)

    def get_metrics(self):
        metrics_dict = super().get_metrics()
        for _agg_mode, agg_metrics in metrics_dict.items():
            if "correct" in agg_metrics:
                agg_metrics["success_rate"] = agg_metrics["correct"]
            if "no_answer" in agg_metrics:
                agg_metrics["no_answer"] = agg_metrics["no_answer"] / 2.0

            if self.wer_total_ref_words > 0:
                agg_metrics["wer"] = round(100.0 * self.wer_total_errors / self.wer_total_ref_words, 2)
                agg_metrics["insertion_rate"] = round(100.0 * self.insertions / self.wer_total_ref_words, 2)
            if self.total > 0:
                agg_metrics["insertion_sample_rate"] = round(100.0 * self.insertion_samples / self.total, 2)
                agg_metrics["hallucination_proxy_rate"] = agg_metrics["insertion_sample_rate"]
            if self.audio_seconds > 0:
                agg_metrics["pred_words_per_minute"] = round(self.pred_words / (self.audio_seconds / 60.0), 2)

            agg_metrics["substitutions"] = self.substitutions
            agg_metrics["insertions"] = self.insertions
            agg_metrics["deletions"] = self.deletions
            agg_metrics["ref_words"] = self.wer_total_ref_words

        return metrics_dict

    def evaluations_to_print(self):
        return ["pass@1"]

    def metrics_to_print(self):
        metrics = {
            "avg_tokens": as_int,
            "gen_seconds": as_int,
            "success_rate": as_percentage,
        }
        if self.compute_no_answer:
            metrics["no_answer"] = as_percentage
        if self.wer_total_ref_words > 0:
            metrics["wer"] = as_percentage
            metrics["insertion_rate"] = as_percentage
        metrics["insertion_sample_rate"] = as_percentage
        metrics["hallucination_proxy_rate"] = as_percentage
        if self.audio_seconds > 0:
            metrics["pred_words_per_minute"] = as_float
        metrics["insertions"] = as_int
        metrics["substitutions"] = as_int
        metrics["deletions"] = as_int
        metrics["ref_words"] = as_int
        metrics["num_entries"] = as_int
        return metrics
