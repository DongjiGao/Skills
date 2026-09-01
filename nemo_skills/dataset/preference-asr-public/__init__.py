# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Pinned public Preference-ASR release, evaluated with and without instructions."""

REQUIRES_DATA_DIR = True
IS_BENCHMARK_GROUP = True
SCORE_MODULE = "nemo_skills.dataset.preference-asr-public.score"

# Keep the paper's D/I conditions separate. Comparing these leaves measures the
# effect of the request-scoped instruction without inventing a suite scalar.
BENCHMARKS = {
    "preference-asr-public.default": {},
    "preference-asr-public.instructed": {},
}
