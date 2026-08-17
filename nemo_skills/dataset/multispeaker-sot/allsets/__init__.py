# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Single merged benchmark spanning ALL multispeaker-sot eval sets.

This exists so an ENTIRE multispeaker-sot sweep runs as ONE benchmark per
checkpoint -> one convert + one vLLM generation job + one summarize job (3 jobs
per step). Its ``test.jsonl`` is the concatenation of the five already-merged,
audio-path-fixed GROUP manifests listed in ``SUBSETS`` (under20s / 60-90s /
120-180s / full / lsm), built on the cluster by
``build_group_manifests.py``. We merge the GROUP manifests (not the raw member
sets) so we reuse their fixed audio paths verbatim -- the merge is a pure
concatenation and touches no audio.

Per-set cpWER is still fully recovered: every record carries its original
``subset_for_metrics`` tag (e.g. ``under20s-ami-ihm-dev``), and AudioMetrics
buckets cpWER by that tag, so the single metrics.json has the overall cpWER plus
one ``cpwer::<set>`` entry per underlying eval set. The merged output.jsonl can
likewise be split back apart by ``subset_for_metrics`` (or the dataset name in
each record's audio path).

The launcher must use its ``full`` long-audio profile for this aggregate because
it now includes full-session recordings up to approximately 2869 seconds.
"""

METRICS_TYPE = "audio"
# speaker_token is the bare token `spk` (NOT '<spk:*>'): resolve_speaker_tag_style()
# maps `spk` to the identical tag style, but '<spk:*>' reaching the generated shell
# script is parsed by bash as redirection+glob and kills the job.
EVAL_ARGS = "++eval_type=audio ++eval_config.speaker_token=spk ++eval_config.cpwer_normalization=chime8"
GENERATION_ARGS = "++prompt_format=openai ++enable_audio=true"
GENERATION_MODULE = "nemo_skills.inference.generate"
EVAL_SPLIT = "test"

# Member directory names merged into this benchmark's test.jsonl. These are the
# already-merged GROUP dirs (each has its own fixed test.jsonl), so allsets is a
# concatenation of concatenations. Single source of truth for
# build_group_manifests.py.
SUBSETS = [
    "under20s",
    "60-90s",
    "120-180s",
    "full",
    "lsm",
]
