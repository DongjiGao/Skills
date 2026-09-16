# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""MUCS 2021 Subtask 2 blind-test code-switched ASR benchmark.

The two official language tracks are intentionally reported independently.
The challenge release defines Normal WER and Transliterated WER per language;
it does not define a Hindi/Bengali overall score.

Source: https://www.openslr.org/104/
"""

REQUIRES_DATA_DIR = True
IS_BENCHMARK_GROUP = True
SCORE_MODULE = "nemo_skills.dataset.mucs2021.mucs2021_score"
BENCHMARKS = {
    "mucs2021.hindi-english": {},
    "mucs2021.bengali-english": {},
}

METRICS_TYPE = "nemo_skills.dataset.mucs2021.scorer::MUCS2021Metrics"
EVAL_ARGS = "++eval_type=nemo_skills.dataset.mucs2021.scorer::MUCS2021Evaluator"
GENERATION_ARGS = "++prompt_format=openai ++enable_audio=true"
