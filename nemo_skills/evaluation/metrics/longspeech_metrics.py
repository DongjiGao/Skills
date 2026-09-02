# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

"""Officially described deterministic metrics for Marco-LongSpeech."""

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_float, as_int, as_percentage


class LongSpeechMetrics(BaseMetrics):
    """Top-1 metrics for one homogeneous LongSpeech task leaf."""

    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        super().__init__(compute_no_answer=compute_no_answer)
        if max_k != 1:
            raise ValueError("LongSpeechMetrics supports one deterministic generation per sample")
        self.max_k = 1

    def reset(self):
        super().reset()
        self.task = None
        self.no_answer = 0
        self.overall_cer_errors = 0
        self.overall_cer_ref_chars = 0
        self.cjk_cer_errors = 0
        self.cjk_cer_ref_chars = 0
        self.non_cjk_wer_errors = 0
        self.non_cjk_wer_ref_words = 0
        self.num_cjk = 0
        self.num_non_cjk = 0
        self.translation_references = []
        self.translation_hypotheses = []
        self.rouge1_f1 = 0.0
        self.rouge2_f1 = 0.0
        self.rougeL_f1 = 0.0
        self.numeric_parsed = 0
        self.numeric_correct = 0
        self.language_parsed = 0
        self.language_correct = 0

    def _get_score_dict(self, prediction):
        return {"correct": bool(prediction.get("is_correct"))}

    def get_incorrect_sample(self, prediction):
        from nemo_skills.evaluation.evaluator.longspeech import evaluate_longspeech_sample

        return {**prediction, **evaluate_longspeech_sample({**prediction, "generation": ""})}

    def update(self, predictions):
        if len(predictions) != 1:
            raise ValueError("LongSpeechMetrics expects exactly one prediction per sample")
        super().update(predictions)
        prediction = predictions[0]
        task = prediction["longspeech_task"]
        if self.task is None:
            self.task = task
        elif task != self.task:
            raise ValueError(f"LongSpeech metric stream mixed {self.task!r} and {task!r}")

        self.no_answer += int(bool(prediction.get("longspeech_no_answer")))
        if task == "asr":
            errors = int(prediction["asr_cer_errors"])
            ref_chars = int(prediction["asr_cer_ref_chars"])
            self.overall_cer_errors += errors
            self.overall_cer_ref_chars += ref_chars
            if prediction["asr_is_cjk"]:
                self.num_cjk += 1
                self.cjk_cer_errors += errors
                self.cjk_cer_ref_chars += ref_chars
            else:
                self.num_non_cjk += 1
                self.non_cjk_wer_errors += int(prediction["asr_wer_errors"])
                self.non_cjk_wer_ref_words += int(prediction["asr_wer_ref_words"])
        elif task == "translation":
            self.translation_references.append(prediction["translation_reference"])
            self.translation_hypotheses.append(prediction["translation_hypothesis"])
        elif task == "summary":
            self.rouge1_f1 += float(prediction["rouge1_f1"])
            self.rouge2_f1 += float(prediction["rouge2_f1"])
            self.rougeL_f1 += float(prediction["rougeL_f1"])
        elif task in {"content-separation", "speaker-count"}:
            self.numeric_parsed += int(bool(prediction["numeric_parsed"]))
            self.numeric_correct += int(bool(prediction["numeric_correct"]))
        elif task == "language-detection":
            self.language_parsed += int(bool(prediction["language_parsed"]))
            self.language_correct += int(bool(prediction["language_correct"]))
        else:
            raise ValueError(f"Unsupported LongSpeech metric task: {task!r}")

    @staticmethod
    def _rate(numerator: int | float, denominator: int | float) -> float:
        return round(100.0 * numerator / denominator, 2) if denominator else 0.0

    def get_metrics(self):
        values = {}
        self.update_common_metrics(values)
        if self.compute_no_answer:
            values["no_answer"] = self._rate(self.no_answer, self.total)

        if self.task == "asr":
            values["non_cjk_wer"] = self._rate(
                self.non_cjk_wer_errors,
                self.non_cjk_wer_ref_words,
            )
            values["cjk_cer"] = self._rate(self.cjk_cer_errors, self.cjk_cer_ref_chars)
            values["overall_cer"] = self._rate(
                self.overall_cer_errors,
                self.overall_cer_ref_chars,
            )
            values["num_non_cjk"] = self.num_non_cjk
            values["num_cjk"] = self.num_cjk
        elif self.task == "translation":
            from sacrebleu import corpus_bleu

            values["bleu"] = round(
                corpus_bleu(
                    self.translation_hypotheses,
                    [self.translation_references],
                    lowercase=True,
                    tokenize="13a",
                ).score,
                2,
            )
        elif self.task == "summary":
            values["rouge1_f1"] = self._rate(self.rouge1_f1, self.total)
            values["rouge2_f1"] = self._rate(self.rouge2_f1, self.total)
            values["rougeL_f1"] = self._rate(self.rougeL_f1, self.total)
        elif self.task in {"content-separation", "speaker-count"}:
            values["numeric_accuracy"] = self._rate(self.numeric_correct, self.total)
            values["parsability_rate"] = self._rate(self.numeric_parsed, self.total)
            values["post_parsing_precision"] = self._rate(
                self.numeric_correct,
                self.numeric_parsed,
            )
            values["misunderstanding_rate"] = self._rate(
                self.total - self.numeric_parsed,
                self.total,
            )
        elif self.task == "language-detection":
            detection_errors = self.language_parsed - self.language_correct
            values["detection_accuracy"] = self._rate(self.language_correct, self.total)
            values["misunderstanding_rate"] = self._rate(
                self.total - self.language_parsed,
                self.total,
            )
            values["detection_error_rate"] = self._rate(detection_errors, self.total)
            values["detection_errors"] = detection_errors
        return {"pass@1": values}

    def evaluations_to_print(self):
        return ["pass@1"]

    def metrics_to_print(self):
        common = {
            "avg_tokens": as_int,
            "gen_seconds": as_int,
            "no_answer": as_percentage,
        }
        task_metrics = {
            "asr": {
                "non_cjk_wer": as_percentage,
                "cjk_cer": as_percentage,
                "overall_cer": as_percentage,
                "num_non_cjk": as_int,
                "num_cjk": as_int,
            },
            "translation": {"bleu": as_float},
            "summary": {
                "rouge1_f1": as_percentage,
                "rouge2_f1": as_percentage,
                "rougeL_f1": as_percentage,
            },
            "content-separation": {
                "numeric_accuracy": as_percentage,
                "parsability_rate": as_percentage,
                "post_parsing_precision": as_percentage,
                "misunderstanding_rate": as_percentage,
            },
            "speaker-count": {
                "numeric_accuracy": as_percentage,
                "parsability_rate": as_percentage,
                "post_parsing_precision": as_percentage,
                "misunderstanding_rate": as_percentage,
            },
            "language-detection": {
                "detection_accuracy": as_percentage,
                "misunderstanding_rate": as_percentage,
                "detection_error_rate": as_percentage,
                "detection_errors": as_int,
            },
        }
        return {**common, **task_metrics.get(self.task, {}), "num_entries": as_int}
