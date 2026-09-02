# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AliMeeting public far-field test set with speaker-attributed cpCER."""

REQUIRES_DATA_DIR = True
METRICS_TYPE = "mandarin_meeting"
EVAL_ARGS = (
    "++eval_type=mandarin_meeting ++eval_config.speaker_token='<spk:*>' ++eval_config.normalization=alimeeting-sot"
)
GENERATION_ARGS = "++prompt_format=openai ++enable_audio=true"
GENERATION_MODULE = "nemo_skills.inference.generate"
EVAL_SPLIT = "test"

FORMAT_EXPECTATIONS = {
    "alimeeting-sot": {
        "task_type": "Mandarin-MS-ASR",
        "source_split": "official OpenSLR-119 Test_Ali far-field",
        "input": "deliberate distant-array channel-0 SDM reproduction",
        "speaker_label_namespace": "ordinal_speakers",
        "preferred_output_dialect": "spk_tag",
        "metric": "pinned full-session cpCER reproduction",
        "tokenization": "released AliMeeting cleanup plus Unicode characters",
        "repeated_speaker_turns_legal": True,
    }
}
