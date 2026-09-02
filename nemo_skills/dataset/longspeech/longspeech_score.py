# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

"""LongSpeech has no official scalar aggregate across its heterogeneous tasks."""


def compute_score(combined_metrics: dict) -> dict:
    """Return no synthetic cross-task score; leaf metrics remain authoritative."""
    return {}
