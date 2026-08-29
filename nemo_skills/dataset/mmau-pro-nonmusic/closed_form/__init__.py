# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

"""Closed-form MMAU-Pro non-music profile.

All scoring and generation settings are imported from ``mmau-pro`` so this
filtered data profile cannot silently drift from the full benchmark's
official NV-Embed scorer path.
"""

from importlib import import_module

_FULL_PROFILE = import_module("nemo_skills.dataset.mmau-pro.closed_form")

METRICS_TYPE = _FULL_PROFILE.METRICS_TYPE
SCORE_MODULE = _FULL_PROFILE.SCORE_MODULE
GENERATION_ARGS = _FULL_PROFILE.GENERATION_ARGS
JUDGE_PIPELINE_ARGS = _FULL_PROFILE.JUDGE_PIPELINE_ARGS
