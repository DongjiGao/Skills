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

"""Canary Table-6 MUSAN hallucinated-characters/minute reconstruction.

This profile uses the complete OpenSLR-17 Music and Noise categories at their
native 16 kHz rate, without additive mixing or an SNR transform. It reports
corpus generated characters per audio minute after one greedy generation.
That source slice and metric follow arXiv:2406.19674v1 Table 6. The paper does
not publish its exact manifest, prompt, decoder settings, or character
preprocessing, so this result must be labeled a versioned reconstruction rather
than an exact reproduction of the published Canary/Whisper values.

The parent ``musan`` benchmark remains the separate NeMo Gym protocol: all
three categories, four generations, and the >1500 chars/min event threshold.
"""

REQUIRES_DATA_DIR = True
SCORE_MODULE = "nemo_skills.evaluation.metrics.audio_metrics"
METRICS_TYPE = "audio"
NUM_SAMPLES = 1

EVAL_ARGS = "++eval_type=audio"
GENERATION_ARGS = "++prompt_format=openai ++enable_audio=true"
