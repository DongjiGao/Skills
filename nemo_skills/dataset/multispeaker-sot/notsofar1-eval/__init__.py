# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Merged NOTSOFAR1 eval benchmark over the IHM and SC microphone views.

Each record retains its leaf alias in ``subset_for_metrics`` so the parent
evaluation reports IHM and SC cpWER separately without duplicate registration.
"""

METRICS_TYPE = "audio"
EVAL_ARGS = "++eval_type=audio ++eval_config.speaker_token=spk ++eval_config.cpwer_normalization=chime8"
GENERATION_ARGS = "++prompt_format=openai ++enable_audio=true"
GENERATION_MODULE = "nemo_skills.inference.generate"
EVAL_SPLIT = "test"

SUBSETS = [
    "notsofar1-eval-ihm",
    "notsofar1-eval-sc",
]
