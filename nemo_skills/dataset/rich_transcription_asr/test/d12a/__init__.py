# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

REQUIRES_DATA_DIR = True
DEFAULT_SPLIT = "test"
EVAL_SPLIT = "test"
METRICS_TYPE = "nemo_skills.dataset.rich_transcription_asr.rich_transcription_asr_metrics::RichTranscriptionASRMetrics"
EVAL_ARGS = (
    "++eval_type=nemo_skills.dataset.rich_transcription_asr.rich_transcription_asr_eval::RichTranscriptionASREvaluator "
    "++eval_config.evaluation_mode=event"
)
GENERATION_ARGS = "++prompt_format=openai ++enable_audio=true ++enable_audio_chunking=false"
