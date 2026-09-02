# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Public VoiceArena Monsoon ASR self-scoring splits.

These profiles intentionally exclude the speaker-disjoint private halves and
the separately licensed hosted VoiceArena leaderboard corpus.
"""

REQUIRES_DATA_DIR = True
IS_BENCHMARK_GROUP = True
SCORE_MODULE = "nemo_skills.dataset.voicearena-monsoon.score"

BENCHMARKS = {
    "voicearena-monsoon.en-in.public": {},
    "voicearena-monsoon.hi-in.public": {},
}
