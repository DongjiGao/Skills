# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

"""MMAU-Pro's speech/sound-focused non-music evaluation profile.

This is deliberately a different benchmark name from ``mmau-pro``. The
existing group remains the complete 5,305-row release for backward
reproducibility. By default this profile implements the PRD's literal policy:
only rows whose top-level ``category`` is exactly ``music`` are excluded.
Consequently mixed-domain categories such as ``sound_music`` remain included.

Preparation also exposes an explicit stricter ``exclude-any-music`` policy.
It excludes every top-level domain label containing the ``music`` component,
but is never selected implicitly because its 3,784-row result is not
comparable to the default 3,887-row profile.
"""

REQUIRES_DATA_DIR = True
IS_BENCHMARK_GROUP = True

# Reuse the same three official-compatible scorer paths as full MMAU-Pro.
SCORE_MODULE = "nemo_skills.dataset.mmau-pro.mmau_pro_score"

BENCHMARKS = {
    "mmau-pro-nonmusic.closed_form": {},
    "mmau-pro-nonmusic.open_ended": {},
    "mmau-pro-nonmusic.instruction_following": {},
}
