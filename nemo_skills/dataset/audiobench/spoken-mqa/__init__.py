# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Official AudioBench Spoken-MQA Speech Instruction leaves.

The upstream benchmark reports one accuracy for each leaf. The group score
also emits an entry-weighted overall accuracy as a convenience, not as an
official AudioBench aggregate.
"""

REQUIRES_DATA_DIR = True
IS_BENCHMARK_GROUP = True
SCORE_MODULE = "nemo_skills.dataset.audiobench.spoken-mqa.spoken_mqa_score"
BENCHMARKS = {
    f"audiobench.spoken-mqa.{leaf}": {}
    for leaf in (
        "short_digit",
        "long_digit",
        "single_step_reasoning",
        "multi_step_reasoning",
    )
}
