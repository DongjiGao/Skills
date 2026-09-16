# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Two-stage brief PII-aware synthetic VCTK-VITS redaction benchmark."""

REQUIRES_DATA_DIR = True
DEFAULT_SPLIT = "test"
METRICS_TYPE = "nemo_skills.dataset.pii-asr-eval-vctk-vits-two-stage.scorer::PIIASRTwoStageMetrics"
EVAL_ARGS = "++eval_type=nemo_skills.dataset.pii-asr-eval-vctk-vits-two-stage.scorer::PIIASRTwoStageEvaluator"
GENERATION_MODULE = "nemo_skills.inference.eval.pii_asr_two_stage"
GENERATION_ARGS = (
    "++prompt_format=openai ++enable_audio=true ++server.enable_soft_fail=true ++inference.temperature=0.0"
)
