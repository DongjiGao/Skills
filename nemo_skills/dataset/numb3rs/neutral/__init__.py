# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numb3rs neutral-transcription prompt leaf."""

from nemo_skills.dataset.numb3rs import (
    DATASET_GROUP,
    EVAL_ARGS,
    GENERATION_ARGS,
    METRICS_TYPE,
)

DEFAULT_SPLIT = "test"
EVAL_SPLIT = "test"

__all__ = [
    "DATASET_GROUP",
    "DEFAULT_SPLIT",
    "EVAL_ARGS",
    "EVAL_SPLIT",
    "GENERATION_ARGS",
    "METRICS_TYPE",
]
