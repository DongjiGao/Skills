# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Word-timestamp evaluation datasets for CTC-aligned speech models."""

IS_BENCHMARK_GROUP = True
REQUIRES_DATA_DIR = True

BENCHMARKS = {
    "ctc-timestamp-eval.ch109": {},
    "ctc-timestamp-eval.ami-ihm-test": {},
    "ctc-timestamp-eval.lsm": {},
    **{
        f"ctc-timestamp-eval.d12a-{language}": {}
        for language in (
            "ar",
            "de",
            "en",
            "es",
            "fr",
            "hi",
            "it",
            "ja",
            "ko",
            "pt",
            "ru",
            "th",
            "tr",
            "vi",
            "zh",
        )
    },
}
