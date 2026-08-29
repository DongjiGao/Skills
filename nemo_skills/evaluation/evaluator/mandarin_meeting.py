# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speaker-attributed Mandarin meeting evaluation with official cpCER."""

from __future__ import annotations

from typing import Any

from nemo_skills.dataset.mandarin_meeting_common import cpcer_token_string
from nemo_skills.evaluation.evaluator.audio import parse_sot_speaker_streams, resolve_speaker_tag_style
from nemo_skills.evaluation.evaluator.base import BaseEvaluator, BaseEvaluatorConfig
from nemo_skills.evaluation.metrics.cpwer_utils import session_cpwer_counts
from nemo_skills.utils import nested_dataclass


def evaluate_mandarin_meeting_sample(
    data_point: dict[str, Any], speaker_token: str = "<spk:*>"
) -> dict[str, Any]:
    reference = str(data_point["expected_answer"])
    generation = str(data_point.get("generation") or "").strip()
    tag_style = resolve_speaker_tag_style(speaker_token)
    reference_streams = parse_sot_speaker_streams(reference, mode="strict", strict_tag_style=tag_style)
    hypothesis_streams = parse_sot_speaker_streams(generation, mode="canonical", strict_tag_style=tag_style)
    normalized_reference = {speaker: cpcer_token_string(text) for speaker, text in reference_streams.items()}
    normalized_hypothesis = {speaker: cpcer_token_string(text) for speaker, text in hypothesis_streams.items()}
    if not normalized_reference or not any(normalized_reference.values()):
        raise ValueError("Mandarin meeting reference is empty after official normalization")
    counts = session_cpwer_counts(normalized_reference, normalized_hypothesis)
    return {
        "generation": generation,
        "text": " ".join(normalized_reference.values()),
        "pred_text": " ".join(normalized_hypothesis.values()),
        "cpcer": counts["cpwer"],
        "cpcer_errors": counts["errors"],
        "cpcer_substitutions": counts["substitutions"],
        "cpcer_insertions": counts["insertions"],
        "cpcer_deletions": counts["deletions"],
        "cpcer_ref_chars": counts["ref_words"],
        "speaker_assignment": counts["assignment"],
        "reference_streams": normalized_reference,
        "hypothesis_streams": normalized_hypothesis,
        "is_correct": counts["errors"] == 0,
    }


@nested_dataclass(kw_only=True)
class MandarinMeetingEvaluatorConfig(BaseEvaluatorConfig):
    speaker_token: str = "<spk:*>"


class MandarinMeetingEvaluator(BaseEvaluator):
    def __init__(self, config: dict, num_parallel_requests: int = 10):
        super().__init__(config, num_parallel_requests)
        self.eval_config = MandarinMeetingEvaluatorConfig(**self.config)

    async def eval_single(self, data_point: dict[str, Any]) -> dict[str, Any]:
        return evaluate_mandarin_meeting_sample(data_point, speaker_token=self.eval_config.speaker_token)
