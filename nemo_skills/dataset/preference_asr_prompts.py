# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Versioned Preference-ASR prompt repair; preserve every preference suffix."""

PROMPT_POLICY = "preference-asr-recording-prefix-v1"


def rewrite_instruction(instruction: str) -> str:
    """Avoid the two measured EOS-triggering openings without changing the task."""
    english = "Transcribe the speech in the input English audio"
    if instruction == english:
        return "Transcribe this English recording."
    prefix = "Transcribe the speech in the input audio."
    if instruction.startswith(prefix):
        return "Transcribe this recording." + instruction[len(prefix) :]
    return instruction
