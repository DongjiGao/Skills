# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""LoGenBench: audio requests previously observed to terminate by length."""

REQUIRES_DATA_DIR = True
METRICS_TYPE = "nemo_skills.dataset.logenbench.metrics::LoGenBenchMetrics"
GENERATION_ARGS = (
    "++prompt_format=openai ++enable_audio=true ++server.enable_soft_fail=true "
    "++enable_audio_chunking=true ++chunk_audio_threshold_sec=3300 "
    "++inference_overrides_key=inference_overrides"
)
