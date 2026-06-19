# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Evaluator for LibriSpeechMix SOT cpWER."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import tqdm

from nemo_skills.evaluation.evaluator.base import BaseEvaluator, BaseEvaluatorConfig
from nemo_skills.evaluation.metrics.librispeechmix_sot_utils import content_wer, cpwer, write_meeteval_artifacts
from nemo_skills.utils import nested_dataclass


@nested_dataclass(kw_only=True)
class LibriSpeechMixSOTEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for LibriSpeechMix SOT evaluation."""

    prediction_keys: list[str] | None = None
    write_meeteval: bool = True
    reference_tag_style: str = "spk"
    hypothesis_strict_tag_style: str | None = None


class LibriSpeechMixSOTEvaluator(BaseEvaluator):
    """Compute per-record cpWER and optional MeetEval artifacts."""

    def __init__(self, config: Dict[str, Any], num_parallel_requests=10):
        super().__init__(LibriSpeechMixSOTEvaluatorConfig(_init_nested=True, **config), num_parallel_requests)

    def _get_hypothesis(self, data_point: dict[str, Any]) -> str:
        keys = self.config.prediction_keys or ["pred_text", "generation"]
        for key in keys:
            value = data_point.get(key)
            if value is not None:
                return str(value)
        return ""

    async def eval_single(self, data_point: dict[str, Any]) -> dict[str, Any]:
        hypothesis = self._get_hypothesis(data_point)
        strict_result = cpwer(
            data_point["text"],
            hypothesis,
            hypothesis_mode="strict",
            reference_tag_style=self.config.reference_tag_style,
            hypothesis_strict_tag_style=self.config.hypothesis_strict_tag_style,
        )
        canonical_result = cpwer(
            data_point["text"],
            hypothesis,
            hypothesis_mode="canonical",
            reference_tag_style=self.config.reference_tag_style,
            hypothesis_strict_tag_style=self.config.hypothesis_strict_tag_style,
        )
        content_result = content_wer(data_point["text"], hypothesis)
        return {
            "pred_text": hypothesis,
            "cpwer": strict_result["cpwer"],
            "cpwer_errors": strict_result["errors"],
            "cpwer_substitutions": strict_result["substitutions"],
            "cpwer_insertions": strict_result["insertions"],
            "cpwer_deletions": strict_result["deletions"],
            "cpwer_ref_words": strict_result["ref_words"],
            "strict_cpwer": strict_result["cpwer"],
            "strict_cpwer_errors": strict_result["errors"],
            "strict_cpwer_substitutions": strict_result["substitutions"],
            "strict_cpwer_insertions": strict_result["insertions"],
            "strict_cpwer_deletions": strict_result["deletions"],
            "strict_cpwer_ref_words": strict_result["ref_words"],
            "canonical_cpwer": canonical_result["cpwer"],
            "canonical_cpwer_errors": canonical_result["errors"],
            "canonical_cpwer_substitutions": canonical_result["substitutions"],
            "canonical_cpwer_insertions": canonical_result["insertions"],
            "canonical_cpwer_deletions": canonical_result["deletions"],
            "canonical_cpwer_ref_words": canonical_result["ref_words"],
            "content_wer": content_result["wer"],
            "content_wer_errors": content_result["errors"],
            "content_wer_substitutions": content_result["substitutions"],
            "content_wer_insertions": content_result["insertions"],
            "content_wer_deletions": content_result["deletions"],
            "content_wer_ref_words": content_result["ref_words"],
            "speaker_assignment": canonical_result["assignment"],
            "strict_speaker_assignment": strict_result["assignment"],
            "format_parse_status": canonical_result["format_parse_status"],
            "format_warnings": canonical_result["format_warnings"],
            "canonical_prediction": canonical_result["canonical_prediction"],
            "canonical_reference": canonical_result["canonical_reference"],
            "speaker_mapping": canonical_result["speaker_mapping"],
            "is_correct": strict_result["errors"] == 0,
        }

    async def eval_full(self) -> None:
        records = []
        with open(self.config.input_file, "rt", encoding="utf-8") as fin:
            for line in tqdm.tqdm(fin, desc=f"Evaluating {os.path.basename(self.config.input_file)}"):
                record = json.loads(line)
                record.update(await self.eval_single(record))
                if "generation" not in record:
                    record["generation"] = record["pred_text"]
                records.append(record)

        temp_file = self.config.input_file + "-tmp"
        with open(temp_file, "wt", encoding="utf-8") as fout:
            for record in records:
                fout.write(json.dumps(record) + "\n")
        os.replace(temp_file, self.config.input_file)

        if self.config.write_meeteval:
            artifacts = write_meeteval_artifacts(records, self.config.input_file)
            sidecar = self.config.input_file + ".meeteval_artifacts.json"
            with open(sidecar, "wt", encoding="utf-8") as fout:
                json.dump(artifacts, fout, indent=2)
