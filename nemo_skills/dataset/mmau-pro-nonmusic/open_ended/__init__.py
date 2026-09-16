# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

"""Open-ended MMAU-Pro non-music profile.

All scoring, judge, and generation settings are imported from ``mmau-pro`` so
this filtered data profile reuses the exact Qwen judge path of the full
benchmark.
"""

from importlib import import_module

_FULL_PROFILE = import_module("nemo_skills.dataset.mmau-pro.open_ended")

METRICS_TYPE = _FULL_PROFILE.METRICS_TYPE
SCORE_MODULE = _FULL_PROFILE.SCORE_MODULE
GENERATION_ARGS = _FULL_PROFILE.GENERATION_ARGS
JUDGE_PIPELINE_ARGS = _FULL_PROFILE.JUDGE_PIPELINE_ARGS
JUDGE_ARGS = _FULL_PROFILE.JUDGE_ARGS
