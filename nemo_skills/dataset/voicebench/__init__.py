# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""All public VoiceBench task/config leaves; no cross-task overall is defined."""

REQUIRES_DATA_DIR = True
IS_BENCHMARK_GROUP = True
BENCHMARKS = {
    f"voicebench.{name}": {}
    for name in (
        "advbench",
        "alpacaeval",
        "alpacaeval-full",
        "alpacaeval-speaker",
        "bbh",
        "commoneval",
        "ifeval",
        "mmsu",
        "openbookqa",
        "sd-qa",
        "wildvoice",
    )
}

# Public audio/metadata is prepared, but the pinned official repo provides no
# working two-turn inference implementation or scorer.  It is not in the runnable group.
PUBLIC_DATA_ONLY_BENCHMARKS = {"voicebench.mtbench": {"reason": "official protocol unavailable"}}
