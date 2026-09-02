# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AISHELL-4 public test set with speaker-attributed cpCER."""

REQUIRES_DATA_DIR = True
METRICS_TYPE = "mandarin_meeting"
EVAL_ARGS = (
    # The scorer maps the bare token ``spk`` to the same tag style. Keeping
    # ``<spk:*>`` here is unsafe because nested generated shells can parse it
    # as input redirection before Hydra receives the argument.
    "++eval_type=mandarin_meeting ++eval_config.speaker_token=spk ++eval_config.normalization=aishell4-sot"
)
GENERATION_ARGS = "++prompt_format=openai ++enable_audio=true"
GENERATION_MODULE = "nemo_skills.inference.generate"
EVAL_SPLIT = "test"

FORMAT_EXPECTATIONS = {
    "aishell4-sot": {
        "task_type": "Mandarin-MS-ASR",
        "source_split": "official AISHELL-4 test",
        "input": "deliberate distant-array channel-0 SDM reproduction",
        "speaker_label_namespace": "ordinal_speakers",
        "preferred_output_dialect": "spk_tag",
        "metric": "pinned full-session cpCER reproduction",
        "tokenization": "released AISHELL-4 U+4E00--U+9FA5 characters",
        "repeated_speaker_turns_legal": True,
    }
}
