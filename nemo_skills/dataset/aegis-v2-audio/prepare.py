# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Refuse to prepare AEGIS v2 Audio as though it had an approved protocol."""

from pathlib import Path

from nemo_skills.dataset.protocol_manifest import load_protocol_manifest

PROTOCOL_MANIFEST = load_protocol_manifest(Path(__file__).with_name("protocol.json"))


def main() -> None:
    PROTOCOL_MANIFEST.require_runnable()


if __name__ == "__main__":
    main()
