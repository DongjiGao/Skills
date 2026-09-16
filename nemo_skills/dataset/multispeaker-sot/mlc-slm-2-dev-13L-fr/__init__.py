# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0
"""MLC-SLM 2 French dev evaluation leaf."""
METRICS_TYPE = "audio"
EVAL_ARGS = "++eval_type=audio ++eval_config.speaker_token=spk ++eval_config.cpwer_normalization=chime8"
GENERATION_ARGS = "++prompt_format=openai ++enable_audio=true"
GENERATION_MODULE = "nemo_skills.inference.generate"
EVAL_SPLIT = "test"
