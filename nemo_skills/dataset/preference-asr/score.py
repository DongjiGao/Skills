# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Preserve the contextless and fine conditions as separate tables."""

LEAVES = ("contextless", "fine")


def compute_score(combined_metrics: dict) -> dict:
    selected = {
        name.rsplit("preference-asr.", 1)[-1]: values
        for name, values in combined_metrics.items()
        if name.startswith("preference-asr.")
    }
    if not selected:
        return {}
    unknown = set(selected).difference(LEAVES)
    if unknown:
        raise ValueError(f"Unexpected seven-corpus Preference-ASR profiles: {sorted(unknown)}")
    modes = sorted({mode for values in selected.values() for mode in values})
    return {
        mode: {leaf: selected[leaf][mode] for leaf in LEAVES if leaf in selected and mode in selected[leaf]}
        for mode in modes
    }
