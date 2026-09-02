# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

REQUIRES_DATA_DIR = True
METRICS_TYPE = "nemo_skills.dataset.preference-asr-public.scorer::PreferenceASRMetrics"
EVAL_ARGS = "++eval_type=nemo_skills.dataset.preference-asr-public.scorer::PreferenceASREvaluator"
GENERATION_ARGS = (
    "++prompt_format=openai ++enable_audio=true ++server.enable_soft_fail=true ++inference.temperature=0.0"
)
