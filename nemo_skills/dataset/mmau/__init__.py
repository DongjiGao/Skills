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

"""MMAU's public test-mini speech-and-sound evaluation profile.

The full 9,000-row MMAU test labels are withheld. This adapter uses only the
public 1,000-row test-mini release, further filtered to the official ``speech``
and ``sound`` tasks requested by the product evaluation contract.
"""

REQUIRES_DATA_DIR = True
DEFAULT_SPLIT = "test"
EVAL_SPLIT = "test"
METRICS_TYPE = "set_token_mcq"
EVAL_ARGS = "++eval_type=set-token-mcq ++eval_config.official_output_key=model_output"
GENERATION_ARGS = (
    "++prompt_format=openai ++enable_audio=true ++server.enable_soft_fail=true ++inference.temperature=0.0"
)
