# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Friendly singleton group for the Rich Transcription D8 test benchmark.

The canonical leaf remains ``rich_transcription_asr.test.d8`` so existing
prepared data and result tooling keep one source of truth.  This group exposes
the user-facing ``rich_transcription_asr.d8_test`` spelling without copying the
manifest or reimplementing the event-token scorer.
"""

IS_BENCHMARK_GROUP = True
REQUIRES_DATA_DIR = True
SCORE_MODULE = "nemo_skills.dataset.rich_transcription_asr.rich_transcription_asr_score"

BENCHMARKS = {
    "rich_transcription_asr.test.d8": {},
}
