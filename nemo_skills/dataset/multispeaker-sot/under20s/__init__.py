# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Merged single-benchmark view of the multispeaker-sot `under20s` eval sets.

This is intentionally a *single* benchmark (not IS_BENCHMARK_GROUP): its
``test.jsonl`` is the concatenation of every member set listed in ``SUBSETS``,
built on the cluster by ``build_group_manifests.py``. Running the group as one
benchmark yields exactly one generation (vLLM) job and one summarize job while
keeping the log/job name short (``...-multispeaker-sot.under20s``).

Per-set cpWER is still reported: every record carries a ``subset_for_metrics``
tag, and ``nemo_skills.evaluation.metrics.audio_metrics.AudioMetrics`` buckets
cpWER by that tag, so the single metrics.json includes both the overall cpWER
and one ``cpwer::<set>`` entry per member set.
"""

METRICS_TYPE = "audio"
# NOTE: speaker_token is the bare token `spk` (NOT '<spk:*>'). resolve_speaker_tag_style()
# maps `spk` to the exact same "spk" tag style, but '<spk:*>' reaching the generated
# shell script is parsed by bash as `<spk:*` (input redirection + glob) -> job dies with
# "spk:*: No such file or directory". `spk` has no shell-hostile characters.
EVAL_ARGS = "++eval_type=audio ++eval_config.speaker_token=spk ++eval_config.cpwer_normalization=chime8"
GENERATION_ARGS = "++prompt_format=openai ++enable_audio=true"
GENERATION_MODULE = "nemo_skills.inference.generate"
EVAL_SPLIT = "test"

# Member eval-set directory names merged into this group's test.jsonl. Single
# source of truth for build_group_manifests.py.
SUBSETS = [
    "under20s-ami-ihm-dev",
    "under20s-ami-ihm-test",
    "under20s-ami-sdm-test",
    "under20s-chaes-ch109-test",
    "under20s-libri-am-dev",
    "under20s-mixer6-ch04-test",
]
