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

"""Public AIR-Bench Foundation speech-category evaluation.

This profile contains the 10,973 public Foundation rows from the nine speech
tasks (11 exact task/dataset leaves). It deliberately excludes AIR-Bench Chat,
whose official protocol requires bidirectional GPT-4 judging. The public
dataset is licensed CC-BY-NC-4.0.

Dataset: https://huggingface.co/datasets/qyang1021/AIR-Bench-Dataset
Protocol: https://github.com/OFA-Sys/AIR-Bench
"""

REQUIRES_DATA_DIR = True
DEFAULT_SPLIT = "test"
EVAL_SPLIT = "test"
METRICS_TYPE = "air_bench_foundation"
EVAL_ARGS = "++eval_type=air-bench-foundation"
GENERATION_ARGS = (
    "++prompt_format=openai ++enable_audio=true ++server.enable_soft_fail=true ++inference.temperature=0.0"
)
