# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Official VoxLingua107 development-set language identification.

The public development archive contains 1,609 utterances in 33 languages.
Predictions are one exact canonical ISO label; headline metrics are top-1
micro accuracy and an unweighted macro over the 33 labels.

Source: https://bark.phon.ioc.ee/voxlingua107/
License: CC BY 4.0
"""

REQUIRES_DATA_DIR = True
DEFAULT_SPLIT = "test"
EVAL_SPLIT = "test"
METRICS_TYPE = "language_id"
EVAL_ARGS = "++eval_type=language_id"
GENERATION_ARGS = (
    "++prompt_format=openai ++enable_audio=true "
    "++server.enable_soft_fail=true ++inference.temperature=0.0"
)
