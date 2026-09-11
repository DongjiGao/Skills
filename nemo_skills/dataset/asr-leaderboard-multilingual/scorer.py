# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Mixed WER/OIWER evaluator for the official multilingual leaderboard."""

import importlib

from nemo_skills.evaluation.evaluator.audio import AudioEvaluator


class OpenASRMultilingualEvaluator(AudioEvaluator):
    """Use standard multilingual WER except for Hindi Monsoon lattice rows."""

    async def eval_single(self, data_point: dict) -> dict:
        if data_point.get("monsoon_language") != "hi-in":
            return await super().eval_single(data_point)

        scorer = importlib.import_module("nemo_skills.dataset.voicearena-monsoon.scorer")
        result = scorer.evaluate_monsoon_sample(data_point)
        errors = int(result["monsoon_errors"])
        ref_words = int(result["monsoon_ref_words"])
        result.update(
            wer=errors / ref_words,
            wer_errors=errors,
            wer_ref_words=ref_words,
            wer_substitutions=int(result["monsoon_substitutions"]),
            wer_insertions=int(result["monsoon_insertions"]),
            wer_deletions=int(result["monsoon_deletions"]),
            predicted_answer=result["generation"],
            is_correct=errors == 0,
        )
        return result
