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

"""HALAS evaluator."""

from typing import Any

from nemo_skills.evaluation.evaluator.audio import evaluate_asr, extract_asr_text, preprocess_asr_text
from nemo_skills.evaluation.evaluator.base import BaseEvaluator, BaseEvaluatorConfig
from nemo_skills.utils import nested_dataclass


def evaluate_halas_sample(data_point: dict[str, Any]) -> dict[str, Any]:
    reference = data_point["expected_answer"]
    generation = extract_asr_text((data_point.get("generation") or "").strip())

    result = evaluate_asr(reference, generation, normalization_mode="standard")
    duration = float(data_point.get("audio_duration") or 0.0)
    insertions = int(result.get("wer_insertions") or 0)
    ref_words = int(result.get("wer_ref_words") or 0)
    pred_text = result.get("pred_text") or preprocess_asr_text(generation, mode="standard")
    pred_words = len(pred_text.split()) if pred_text else 0

    result["generation"] = generation
    result["audio_duration"] = duration
    result["pred_words"] = pred_words
    result["insertion_rate"] = insertions / ref_words if ref_words else None
    result["has_insertion"] = insertions > 0
    result["hallucination_proxy"] = insertions > 0
    result["pred_words_per_minute"] = (pred_words / (duration / 60.0)) if duration > 0 else None
    return result


@nested_dataclass(kw_only=True)
class HALASEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for HALAS evaluation."""

    pass


class HALASEvaluator(BaseEvaluator):
    """Evaluate HALAS ASR outputs with WER and insertion-based proxy metrics."""

    async def eval_single(self, data_point: dict[str, Any]) -> dict[str, Any]:
        return evaluate_halas_sample(data_point)
