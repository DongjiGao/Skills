# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

"""Deterministic public subset of the Marco-LongSpeech benchmark.

Supported tasks use only metrics fully specified by the LongSpeech paper:
ASR, speech translation, summarization, content separation, speaker count,
and language detection. Temporal QA is intentionally excluded because the
paper does not publish its GPT-4-Turbo judge prompt. Emotion is excluded
because the published coarse-label mapping omits ``Joyful``, which occurs in
the pinned public test manifest.
"""

REQUIRES_DATA_DIR = True
IS_BENCHMARK_GROUP = True
SCORE_MODULE = "nemo_skills.dataset.longspeech.longspeech_score"

BENCHMARKS = {
    "longspeech.asr": {},
    "longspeech.translation": {},
    "longspeech.summary": {},
    "longspeech.content-separation": {},
    "longspeech.speaker-count": {},
    "longspeech.language-detection": {},
}
