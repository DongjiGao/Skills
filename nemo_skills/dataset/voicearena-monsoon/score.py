# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Keep the two official Monsoon metrics separate; no cross-language scalar."""

LEAVES = ("en-in.public", "hi-in.public")


def compute_score(combined_metrics: dict) -> dict:
    selected = {
        name.rsplit("voicearena-monsoon.", 1)[-1]: values
        for name, values in combined_metrics.items()
        if name.startswith("voicearena-monsoon.")
    }
    if not selected:
        return {}
    unknown = set(selected).difference(LEAVES)
    if unknown:
        raise ValueError(f"Unexpected VoiceArena Monsoon profiles: {sorted(unknown)}")
    return {
        eval_mode: {
            leaf: selected[leaf][eval_mode] for leaf in LEAVES if leaf in selected and eval_mode in selected[leaf]
        }
        for eval_mode in sorted({mode for values in selected.values() for mode in values})
    }
