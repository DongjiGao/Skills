# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Merged single-benchmark view of the multispeaker-sot `full` eval sets.

See ``multispeaker-sot/under20s/__init__.py`` for the rationale (one merged
benchmark -> one vLLM job + one summarize job, per-set cpWER via
``subset_for_metrics``). ``test.jsonl`` is built by build_group_manifests.py.
"""

METRICS_TYPE = "audio"
# NOTE: bare token `spk` (not '<spk:*>') -- see under20s/__init__.py. '<spk:*>' reaching
# the generated shell is parsed as `<spk:*` redirection+glob and kills the job.
EVAL_ARGS = "++eval_type=audio ++eval_config.speaker_token=spk ++eval_config.cpwer_normalization=chime8"
GENERATION_ARGS = "++prompt_format=openai ++enable_audio=true"
GENERATION_MODULE = "nemo_skills.inference.generate"
EVAL_SPLIT = "test"

SUBSETS = [
    "full-ami-ihm-test",
    "full-ami-sdm-test",
    "full-chaes-ch109-test",
    "full-mixer6-ch04-test",
]
