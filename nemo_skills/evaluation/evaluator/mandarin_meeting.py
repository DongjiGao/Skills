# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speaker-attributed Mandarin meeting evaluation with full-session cpCER."""

from __future__ import annotations

from typing import Any

from nemo_skills.dataset.mandarin_meeting_common import cpcer_token_string
from nemo_skills.evaluation.evaluator.audio import parse_sot_speaker_streams, resolve_speaker_tag_style
from nemo_skills.evaluation.evaluator.base import BaseEvaluator, BaseEvaluatorConfig
from nemo_skills.evaluation.metrics.cpwer_utils import session_cpwer_counts
from nemo_skills.utils import nested_dataclass


def _resolve_corpus(data_point: dict[str, Any]) -> str:
    corpus = data_point.get("mandarin_corpus")
    if corpus:
        return str(corpus)
    extra_fields = data_point.get("extra_fields") or {}
    if extra_fields.get("source_repo") == "AISHELL/AISHELL-4":
        return "aishell4-sot"
    if "AliMeeting" in str(extra_fields.get("source_url", "")) or "Test_Ali" in str(
        extra_fields.get("source_split", "")
    ):
        return "alimeeting-sot"
    raise ValueError("Mandarin meeting row is missing a recognized corpus provenance field")


def evaluate_mandarin_meeting_sample(
    data_point: dict[str, Any],
    speaker_token: str = "<spk:*>",
    normalization: str | None = None,
) -> dict[str, Any]:
    reference = str(data_point["expected_answer"])
    generation = str(data_point.get("generation") or "").strip()
    tag_style = resolve_speaker_tag_style(speaker_token)
    reference_streams = parse_sot_speaker_streams(reference, mode="strict", strict_tag_style=tag_style)
    hypothesis_streams = parse_sot_speaker_streams(generation, mode="canonical", strict_tag_style=tag_style)
    corpus = normalization or _resolve_corpus(data_point)
    normalized_reference = {speaker: cpcer_token_string(text, corpus) for speaker, text in reference_streams.items()}
    normalized_hypothesis = {speaker: cpcer_token_string(text, corpus) for speaker, text in hypothesis_streams.items()}
    if not normalized_reference or not any(normalized_reference.values()):
        raise ValueError("Mandarin meeting reference is empty after corpus-specific normalization")
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
        "mandarin_corpus": corpus,
        "is_correct": counts["errors"] == 0,
    }


@nested_dataclass(kw_only=True)
class MandarinMeetingEvaluatorConfig(BaseEvaluatorConfig):
    speaker_token: str = "<spk:*>"
    normalization: str | None = None


class MandarinMeetingEvaluator(BaseEvaluator):
    def __init__(self, config: dict, num_parallel_requests: int = 10):
        super().__init__(config, num_parallel_requests)
        self.eval_config = MandarinMeetingEvaluatorConfig(**self.config)

    async def eval_single(self, data_point: dict[str, Any]) -> dict[str, Any]:
        return evaluate_mandarin_meeting_sample(
            data_point,
            speaker_token=self.eval_config.speaker_token,
            normalization=self.eval_config.normalization,
        )
