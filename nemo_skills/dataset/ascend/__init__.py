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

"""ASCEND Mandarin-English code-switched ASR benchmark.

Uses the official test split and official mixed-error-rate scorer: ASCII spans
are word units and non-ASCII spans are character units after Jieba segmentation.

Dataset: https://huggingface.co/datasets/CAiRE/ASCEND (CC BY-SA 4.0)
Scorer: https://github.com/HLTCHKUST/ASCEND
"""

REQUIRES_DATA_DIR = True
METRICS_TYPE = "ascend"
EVAL_ARGS = "++eval_type=ascend"
GENERATION_ARGS = "++prompt_format=openai ++enable_audio=true"
