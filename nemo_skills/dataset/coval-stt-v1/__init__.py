# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Coval STT v1 frozen public quality profile.

This adapter evaluates only the licensed, fixed 50-utterance ``stt-v1``
LibriSpeech test-clean profile. Coval's per-observation WER is averaged across
utterances for the official quality score. A corpus-micro WER is reported only
as a diagnostic; this profile makes no streaming-latency claim.

Source: https://github.com/coval-ai/benchmarks
Dataset license: CC-BY-4.0
"""

REQUIRES_DATA_DIR = True
METRICS_TYPE = "nemo_skills.dataset.coval-stt-v1.scorer::CovalSTTV1Metrics"
EVAL_ARGS = "++eval_type=nemo_skills.dataset.coval-stt-v1.scorer::CovalSTTV1Evaluator"
GENERATION_ARGS = (
    "++prompt_format=openai ++enable_audio=true ++server.enable_soft_fail=true ++inference.temperature=0.0"
)
