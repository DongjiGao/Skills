# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Non-runnable classification of the public AEGIS 2.0 audio derivative."""

from pathlib import Path

from nemo_skills.dataset.protocol_manifest import load_protocol_manifest

PROTOCOL_MANIFEST = load_protocol_manifest(Path(__file__).with_name("protocol.json"))
PROTOCOL_STATUS = PROTOCOL_MANIFEST.status
PROTOCOL_CLASSIFICATION = PROTOCOL_MANIFEST.classification
IS_RUNNABLE_BENCHMARK = False


def require_runnable_protocol() -> None:
    """Raise until the product owner approves the missing task and score contracts."""
    PROTOCOL_MANIFEST.require_runnable()
