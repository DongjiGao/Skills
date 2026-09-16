# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Official Hugging Face Open ASR Leaderboard English long-form track.

This is deliberately separate from ``audiobench.earnings21`` and
``audiobench.earnings22``.  AudioBench randomizes over nine prompts and uses
its own pinned text-normalization recipe; this benchmark uses one deterministic
ASR adapter prompt and the Open ASR English normalizer/WER/composite contract.
"""

REQUIRES_DATA_DIR = True
METRICS_TYPE = "open_asr_leaderboard"
EVAL_ARGS = "++eval_type=audio ++eval_config.normalization_mode=hf_leaderboard"
GENERATION_ARGS = (
    "++prompt_format=openai ++enable_audio=true ++server.enable_soft_fail=true "
    "++server.context_limit_retry_strategy=reduce_generation"
)
