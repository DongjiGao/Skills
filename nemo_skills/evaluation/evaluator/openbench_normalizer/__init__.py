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

"""English text normalizer vendored from Argmax OpenBench (MIT, see LICENSE).

Source: https://github.com/argmaxinc/OpenBench (commit e13108070b96, 2026-09-24),
``src/openbench/metric/word_error_metrics/{text_normalizer,english_abbreviations}.py``.
Used to score Contextual Earnings-22 exactly like the OpenBench leaderboard.
"""

from .text_normalizer import EnglishTextNormalizer

__all__ = ["EnglishTextNormalizer"]
