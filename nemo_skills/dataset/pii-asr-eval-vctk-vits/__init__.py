# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Synthetic VCTK-VITS PII redaction benchmark."""

REQUIRES_DATA_DIR = True
DEFAULT_SPLIT = "test"
METRICS_TYPE = "nemo_skills.dataset.pii-asr-eval-vctk-vits.scorer::PIIASRRedactionMetrics"
EVAL_ARGS = "++eval_type=nemo_skills.dataset.pii-asr-eval-vctk-vits.scorer::PIIASRRedactionEvaluator"
GENERATION_ARGS = (
    "++prompt_format=openai ++enable_audio=true ++server.enable_soft_fail=true ++inference.temperature=0.0"
)
