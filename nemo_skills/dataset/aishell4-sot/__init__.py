# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AISHELL-4 public test set with speaker-attributed cpCER."""

REQUIRES_DATA_DIR = True
METRICS_TYPE = "mandarin_meeting"
EVAL_ARGS = "++eval_type=mandarin_meeting ++eval_config.speaker_token='<spk:*>'"
GENERATION_ARGS = "++prompt_format=openai ++enable_audio=true"
GENERATION_MODULE = "nemo_skills.inference.generate"
EVAL_SPLIT = "test"

FORMAT_EXPECTATIONS = {
    "aishell4-sot": {
        "task_type": "Mandarin-MS-ASR",
        "source_split": "official AISHELL-4 test",
        "input": "distant-array channel 0",
        "speaker_label_namespace": "ordinal_speakers",
        "preferred_output_dialect": "spk_tag",
        "metric": "cpCER",
        "tokenization": "official normalized Unicode characters",
        "repeated_speaker_turns_legal": True,
    }
}
