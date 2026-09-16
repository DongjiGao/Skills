# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Shared VoiceBench leaf configuration.

VoiceBench evaluates an audio-only interaction: the public text ``prompt`` is
judge metadata and must never be sent to the model.
"""

REQUIRES_DATA_DIR = True
DEFAULT_SPLIT = "test"
EVAL_SPLIT = "test"
METRICS_TYPE = "voicebench"
EVAL_ARGS = "++eval_type=voicebench"
GENERATION_ARGS = (
    "++prompt_format=openai ++enable_audio=true ++server.enable_soft_fail=true ++inference.temperature=0.0"
)

JUDGE_PIPELINE_ARGS = {
    "judge_step_fn": "nemo_skills.pipeline.judges.voicebench_judge::create_judge_tasks",
    "api_key_env": "NVIDIA_INFERENCE_HUB_API_KEY",
    "base_url": "https://inference-api.nvidia.com/v1",
    "judge_model": "openai/openai/gpt-4o-mini",
    "max_workers": 12,
}

JUDGED_EVAL_ARGS = ""
