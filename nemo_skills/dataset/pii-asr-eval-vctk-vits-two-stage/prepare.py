# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Prepare the two-stage PII-ASR benchmark."""

import importlib

if __name__ == "__main__":
    importlib.import_module("nemo_skills.dataset.pii-asr-eval-vctk-vits.prepare").main(default_scenario="two-stage")
