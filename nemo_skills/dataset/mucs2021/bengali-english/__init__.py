# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

from nemo_skills.dataset.mucs2021 import EVAL_ARGS, GENERATION_ARGS, METRICS_TYPE

REQUIRES_DATA_DIR = True
DEFAULT_SPLIT = "test"
EVAL_SPLIT = "test"

__all__ = ["EVAL_ARGS", "GENERATION_ARGS", "METRICS_TYPE"]
