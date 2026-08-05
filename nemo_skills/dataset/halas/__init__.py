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

"""HALAS ASR benchmark over Earnings22 audio segments.

HALAS provides human hallucination annotations for released ASR systems. New
model outputs do not have human span labels, so this adapter evaluates ASR
quality against the corrected reference transcript and reports insertion-based
hallucination proxy metrics alongside WER.

Dataset: https://huggingface.co/datasets/MatBar99/HALAS
Paper:   https://arxiv.org/abs/2606.23048
"""

REQUIRES_DATA_DIR = True
METRICS_TYPE = "halas"
EVAL_ARGS = "++eval_type=halas"
GENERATION_ARGS = "++prompt_format=openai ++enable_audio=true ++server.enable_soft_fail=true"
