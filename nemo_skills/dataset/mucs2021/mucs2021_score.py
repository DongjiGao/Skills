# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Group presentation for MUCS 2021 without an invented overall score."""

LEAVES = ("hindi-english", "bengali-english")


def compute_score(combined_metrics: dict) -> dict:
    """Return the two official language-track results side by side.

    The released challenge scorer is invoked separately for Hindi-English and
    Bengali-English. It specifies no weighting or official overall across the
    tracks, so this function deliberately does not create one.
    """
    by_leaf = {
        name.split(".")[-1]: values for name, values in combined_metrics.items() if name.split(".")[-1] in LEAVES
    }
    if not by_leaf:
        return {}

    eval_modes = {mode for values in by_leaf.values() for mode in values}
    return {
        mode: {leaf: by_leaf[leaf][mode] for leaf in LEAVES if mode in by_leaf.get(leaf, {})}
        for mode in sorted(eval_modes)
    }
