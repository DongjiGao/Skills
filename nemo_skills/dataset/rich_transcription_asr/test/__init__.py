# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Rich Transcription ASR test benchmark group."""

IS_BENCHMARK_GROUP = True
REQUIRES_DATA_DIR = True
SCORE_MODULE = "nemo_skills.dataset.rich_transcription_asr.rich_transcription_asr_score"

BENCHMARKS = {
    "rich_transcription_asr.test.d12a": {},
    "rich_transcription_asr.test.d8": {},
}
