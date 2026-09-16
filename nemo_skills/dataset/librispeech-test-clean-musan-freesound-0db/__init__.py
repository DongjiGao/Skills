# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Frozen LibriSpeech test-clean + MUSAN FreeSound noise at 0 dB SNR.

This is an ASR robustness profile, not the MUSAN no-speech hallucination
benchmark.  Its construction is versioned in :mod:`prepare`.
"""

REQUIRES_DATA_DIR = True
METRICS_TYPE = "audio"
EVAL_ARGS = "++eval_type=audio ++eval_config.normalization_mode=standard"
GENERATION_ARGS = "++prompt_format=openai ++enable_audio=true"
