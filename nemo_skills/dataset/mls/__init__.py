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

"""Multilingual LibriSpeech (MLS) official eight-language ASR test benchmark.

The paper's Table 6 evaluates the MLS test split in English plus the seven
non-English languages.  The benchmark reports corpus WER for each language and
uses the unweighted mean of those eight language WERs as its headline score.
"""

REQUIRES_DATA_DIR = True
METRICS_TYPE = "audio"
EVAL_ARGS = "++eval_type=audio ++eval_config.normalization_mode=multilingual"
GENERATION_ARGS = "++prompt_format=openai ++enable_audio=true"
