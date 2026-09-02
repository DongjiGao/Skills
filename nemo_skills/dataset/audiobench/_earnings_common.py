# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Shared AudioBench long-form ASR evaluation settings."""

REQUIRES_DATA_DIR = True
METRICS_TYPE = "audio"
EVAL_ARGS = "++eval_type=audio ++eval_config.normalization_mode=audiobench_earnings"
GENERATION_ARGS = (
    "++prompt_format=openai ++enable_audio=true ++server.enable_soft_fail=true "
    "++server.context_limit_retry_strategy=reduce_generation"
)
EVAL_SPLIT = "test"
