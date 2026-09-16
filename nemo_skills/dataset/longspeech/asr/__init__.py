# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

REQUIRES_DATA_DIR = True
METRICS_TYPE = "nemo_skills.evaluation.metrics.longspeech_metrics::LongSpeechMetrics"
EVAL_ARGS = "++eval_type=nemo_skills.evaluation.evaluator.longspeech::LongSpeechEvaluator"
GENERATION_ARGS = "++prompt_format=openai ++enable_audio=true ++server.enable_soft_fail=true"
