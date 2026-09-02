# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

"""Instruction-following MMAU-Pro non-music profile.

All evaluator, metric, and generation settings are imported from
``mmau-pro`` so this filtered profile reuses the exact official AIF constraint
scorer port.
"""

from importlib import import_module

_FULL_PROFILE = import_module("nemo_skills.dataset.mmau-pro.instruction_following")

METRICS_TYPE = _FULL_PROFILE.METRICS_TYPE
SCORE_MODULE = _FULL_PROFILE.SCORE_MODULE
EVAL_ARGS = _FULL_PROFILE.EVAL_ARGS
GENERATION_ARGS = _FULL_PROFILE.GENERATION_ARGS
