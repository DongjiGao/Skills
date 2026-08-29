# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

from nemo_skills.dataset.voicebench.common import (
    DEFAULT_SPLIT,
    EVAL_SPLIT,
    REQUIRES_DATA_DIR,
)
from nemo_skills.dataset.voicebench.common import (
    GENERATION_ARGS as COMMON_GENERATION_ARGS,
)

METRICS_TYPE = "if"
EVAL_ARGS = "++eval_type=voicebench-ifeval"
GENERATION_ARGS = f"{COMMON_GENERATION_ARGS} ++generation_key=response"
