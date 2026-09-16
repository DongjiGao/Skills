# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Merged MLC-SLM 2 dev benchmark over the selected 13 locale leaves.

Each record retains the matching leaf alias in ``subset_for_metrics`` so the
parent evaluation reports per-locale cpWER without separate generation jobs.
"""

METRICS_TYPE = "audio"
EVAL_ARGS = "++eval_type=audio ++eval_config.speaker_token=spk ++eval_config.cpwer_normalization=chime8"
GENERATION_ARGS = "++prompt_format=openai ++enable_audio=true"
GENERATION_MODULE = "nemo_skills.inference.generate"
EVAL_SPLIT = "test"

SUBSETS = [
    "mlc-slm-2-dev-13L-de",
    "mlc-slm-2-dev-13L-en",
    "mlc-slm-2-dev-13L-es",
    "mlc-slm-2-dev-13L-es-MX",
    "mlc-slm-2-dev-13L-fr",
    "mlc-slm-2-dev-13L-fr-CA",
    "mlc-slm-2-dev-13L-it",
    "mlc-slm-2-dev-13L-ja",
    "mlc-slm-2-dev-13L-ko",
    "mlc-slm-2-dev-13L-pt",
    "mlc-slm-2-dev-13L-pt-BR",
    "mlc-slm-2-dev-13L-ru",
    "mlc-slm-2-dev-13L-th",
]
