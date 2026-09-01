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

"""VoiceCodeBench structured-token ASR evaluation.

The model receives only raw audio and a generic transcription instruction.
Format-invariant WER follows the released scorer. By default, CTEM/TSR use the
offline, annotation-only ``deterministic_annotations_v1`` verifier. It accepts
only exact canonical or expected-acoustic values after versioned formatting
normalization. These deterministic scores are reproducible but are not
interchangeable with release scores from ``openai_gpt_5_5_v1``. Use explicit
``live``/``replay`` verifier settings when official-release comparability is
required. Entity values and metadata remain scorer-only fields.

Dataset/scorer: https://huggingface.co/datasets/besimple-ai/voice-code-bench
"""

REQUIRES_DATA_DIR = True
METRICS_TYPE = "voice_code_bench"
EVAL_ARGS = (
    "++eval_type=voice_code_bench "
    "++eval_config.entity_verifier_mode=deterministic"
)
GENERATION_ARGS = (
    "++prompt_format=openai ++enable_audio=true "
    "++server.enable_soft_fail=true"
)
