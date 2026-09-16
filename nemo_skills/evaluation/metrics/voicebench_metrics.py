# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Task-local VoiceBench metrics.  Deliberately defines no suite overall."""

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_float, as_int, as_percentage


class VoiceBenchMetrics(BaseMetrics):
    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        super().__init__(compute_no_answer=compute_no_answer)
        if max_k != 1:
            raise ValueError("VoiceBench official protocols use one model generation")
        self.max_k = 1
        self.reset()

    def reset(self):
        super().reset()
        self.task = None
        self.open_rating_sum = 0.0
        self.open_rating_count = 0
        self.qa_gpt_correct = 0
        self.qa_pedant_correct = 0
        self.qa_pedant_count = 0
        self.parse_success = 0
        self.deterministic_correct = 0
        self.parity_correct = 0
        self.refusals = 0
        self.revisions = set()

    def _get_score_dict(self, prediction):
        return {"correct": False}

    def get_incorrect_sample(self, prediction):
        prediction = prediction.copy()
        prediction["voicebench_accuracy_unparsed_wrong"] = False
        prediction["voicebench_accuracy_random_fallback_parity"] = False
        prediction["voicebench_qa_gpt_correct"] = False
        prediction["voicebench_qa_pedant_correct"] = False
        prediction["voicebench_refusal"] = False
        prediction["voicebench_open_ratings"] = [0.0, 0.0, 0.0]
        return prediction

    def update(self, predictions):
        if len(predictions) != 1:
            raise ValueError("VoiceBenchMetrics expects exactly one prediction per example")
        super().update(predictions)
        prediction = predictions[0]
        task = prediction["voicebench_task"]
        if self.task is None:
            self.task = task
        elif task != self.task:
            raise ValueError(f"VoiceBench metric aggregation cannot mix tasks: {self.task}, {task}")
        self.revisions.add(str(prediction["official_code_revision"]))

        if "voicebench_open_ratings" in prediction:
            ratings = prediction["voicebench_open_ratings"]
            self.open_rating_sum += sum(float(value) for value in ratings)
            self.open_rating_count += len(ratings)
        if "voicebench_qa_gpt_correct" in prediction:
            self.qa_gpt_correct += int(bool(prediction["voicebench_qa_gpt_correct"]))
        if "voicebench_qa_pedant_correct" in prediction:
            self.qa_pedant_correct += int(bool(prediction["voicebench_qa_pedant_correct"]))
            self.qa_pedant_count += 1
        if "voicebench_parse_success" in prediction:
            self.parse_success += int(bool(prediction["voicebench_parse_success"]))
            self.deterministic_correct += int(bool(prediction["voicebench_accuracy_unparsed_wrong"]))
            self.parity_correct += int(bool(prediction["voicebench_accuracy_random_fallback_parity"]))
        if "voicebench_refusal" in prediction:
            self.refusals += int(bool(prediction["voicebench_refusal"]))

    def get_metrics(self):
        if len(self.revisions) != 1:
            raise ValueError(f"VoiceBench metrics mixed official revisions: {sorted(self.revisions)}")
        values = {"num_entries": self.total, "official_code_revision": next(iter(self.revisions))}
        if self.open_rating_count:
            values["gpt"] = self.open_rating_sum / self.open_rating_count
        elif self.task == "sd-qa":
            values["gpt"] = 100.0 * self.qa_gpt_correct / self.total
            if self.qa_pedant_count:
                values["panda"] = 100.0 * self.qa_pedant_correct / self.qa_pedant_count
        elif self.task in {"openbookqa", "mmsu", "bbh"}:
            values["accuracy_unparsed_wrong"] = 100.0 * self.deterministic_correct / self.total
            values["parse_rate"] = 100.0 * self.parse_success / self.total
            values["unparsed_rate"] = 100.0 * (self.total - self.parse_success) / self.total
            values["accuracy_random_fallback_parity"] = 100.0 * self.parity_correct / self.total
        elif self.task == "advbench":
            # Official HarmEvaluator returns this as a 0..1 fraction, unlike other VoiceBench percentages.
            values["refusal_rate"] = self.refusals / self.total
        elif self.task == "mtbench":
            values["protocol_supported"] = False
        self.update_common_metrics(values)
        return {"pass@1": values}

    def evaluations_to_print(self):
        return ["pass@1"]

    def metrics_to_print(self):
        return {
            "gpt": as_float,
            "panda": as_percentage,
            "accuracy_unparsed_wrong": as_percentage,
            "parse_rate": as_percentage,
            "unparsed_rate": as_percentage,
            "accuracy_random_fallback_parity": as_percentage,
            "refusal_rate": as_float,
            "num_entries": as_int,
        }
