# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public reconstruction proxy for the NFA word-timestamp protocol.

The known public protocol uses LibriSpeech ``test-other`` and Mozilla Common
Voice 16.1 English, Spanish, French, and German; 80 ms timestamp tokens; and a
strict 240 ms start/end tolerance. The authors did not publish their CTMs,
alignment artifacts, or alignment/configuration pins. Consequently this
benchmark deliberately requires a caller-pinned local alignment artifact and
labels every result as a reconstruction proxy, not a paper reproduction.
"""

REQUIRES_DATA_DIR = True
NUM_SAMPLES = 1
EVAL_SPLIT = "test"
METRICS_TYPE = "nemo_skills.dataset.nfa-timestamp-reconstruction.scorer::NFATimestampReconstructionMetrics"
EVAL_ARGS = "++eval_type=nemo_skills.dataset.nfa-timestamp-reconstruction.scorer::NFATimestampReconstructionEvaluator"
GENERATION_ARGS = "++prompt_format=openai ++enable_audio=true"
