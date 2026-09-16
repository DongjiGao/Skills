# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Official NeMo Gym MUSAN hallucination benchmark.

The default preparation uses all OpenSLR-17 noise, music, and speech files.
Every reference is empty.  Four independently seeded generations are scored;
an output is a hallucination only when its character rate is strictly greater
than 1500 characters/minute.  The prompt and protocol match NeMo Gym revision
``68cb9969d656d259846c45e7369acfe6529f0e97``.

This no-speech/non-target-speech protocol is intentionally distinct from the
``librispeech-test-clean-musan-freesound-0db`` noisy-ASR benchmark.
"""

REQUIRES_DATA_DIR = True
SCORE_MODULE = "nemo_skills.evaluation.metrics.audio_metrics"
METRICS_TYPE = "audio"
NUM_SAMPLES = 4

EVAL_ARGS = "++eval_type=audio"
GENERATION_ARGS = "++prompt_format=openai ++enable_audio=true"
