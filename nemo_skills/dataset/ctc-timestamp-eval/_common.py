# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared configuration for ctc-timestamp-eval leaves."""

REQUIRES_DATA_DIR = True
EVAL_SPLIT = "test"
METRICS_TYPE = "nemo_skills.dataset.ctc-timestamp-eval.scorer::CTCTimestampMetrics"
EVAL_ARGS = "++eval_type=nemo_skills.dataset.ctc-timestamp-eval.scorer::CTCTimestampEvaluator"
GENERATION_ARGS = "++prompt_format=openai ++enable_audio=true"
