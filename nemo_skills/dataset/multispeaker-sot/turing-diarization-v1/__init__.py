# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Merged Turing multilingual full-conversation evaluation benchmark.

The parent manifest is the concatenation of the ten language leaves in
``SUBSETS``. Each record retains the matching leaf alias in
``subset_for_metrics`` so the parent evaluation reports per-language cpWER.
"""

METRICS_TYPE = "audio"
EVAL_ARGS = "++eval_type=audio ++eval_config.speaker_token=spk ++eval_config.cpwer_normalization=chime8"
GENERATION_ARGS = "++prompt_format=openai ++enable_audio=true"
GENERATION_MODULE = "nemo_skills.inference.generate"
EVAL_SPLIT = "test"

SUBSETS = [
    "turing-diarization-v1-ar",
    "turing-diarization-v1-el",
    "turing-diarization-v1-en",
    "turing-diarization-v1-es",
    "turing-diarization-v1-fr",
    "turing-diarization-v1-it",
    "turing-diarization-v1-ja",
    "turing-diarization-v1-ko",
    "turing-diarization-v1-pt",
    "turing-diarization-v1-ru",
]
