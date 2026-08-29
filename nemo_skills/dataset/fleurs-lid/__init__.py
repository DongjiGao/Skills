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

"""FLEURS language identification over the merged 102-locale test set.

Predictions are exactly one canonical FLEURS locale label. Reports top-1 micro
accuracy, macro accuracy over locales, and geographic breakdowns.

Dataset: https://huggingface.co/datasets/google/fleurs (CC BY 4.0)
Paper: https://arxiv.org/abs/2205.12446
"""

REQUIRES_DATA_DIR = True
METRICS_TYPE = "fleurs_lid"
EVAL_ARGS = "++eval_type=fleurs_lid"
GENERATION_ARGS = "++prompt_format=openai ++enable_audio=true"
