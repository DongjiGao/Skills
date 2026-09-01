# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Fail-closed classifications for public artifacts without runnable protocols.

These manifests are deliberately separate from benchmark definitions.  A public
dataset or scanner is not a runnable benchmark until its task, inference, scorer,
and aggregation contracts are all fixed.  Loading a classification is safe;
requesting preparation always raises :class:`ProtocolDecisionRequired`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ProtocolDecisionRequired(RuntimeError):
    """Raised when a public artifact still lacks an approved evaluation protocol."""


@dataclass(frozen=True)
class ProtocolManifest:
    """Validated, non-runnable protocol classification."""

    path: Path
    payload: dict[str, Any]

    @property
    def benchmark(self) -> str:
        return self.payload["benchmark"]

    @property
    def status(self) -> str:
        return self.payload["status"]

    @property
    def classification(self) -> str:
        return self.payload["classification"]

    @property
    def missing_product_decisions(self) -> tuple[dict[str, str], ...]:
        return tuple(self.payload["missing_product_decisions"])

    def require_runnable(self) -> None:
        """Stop before data preparation, inference, or scoring can be implied."""
        decision_ids = ", ".join(decision["id"] for decision in self.missing_product_decisions)
        raise ProtocolDecisionRequired(
            f"{self.benchmark} is classified as {self.classification!r} and is not a runnable, "
            f"comparable benchmark. Product decisions required: {decision_ids}. "
            f"See {self.path}."
        )


_REQUIRED_STRING_FIELDS = ("benchmark", "display_name", "status", "classification", "operator_action")
_REQUIRED_LIST_FIELDS = (
    "public_artifacts",
    "source_evidence",
    "missing_product_decisions",
    "prohibited_claims",
    "data_preparation_requirements",
)
_ALLOWED_STATUSES = {"product-decision-required"}
_ALLOWED_CLASSIFICATIONS = {
    "public-artifact-no-comparable-speechlm-protocol",
    "scanner-no-fixed-product-benchmark",
}


def load_protocol_manifest(path: str | Path) -> ProtocolManifest:
    """Load and strictly validate a non-runnable protocol manifest."""
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load protocol manifest {manifest_path}") from error

    if not isinstance(payload, dict):
        raise TypeError(f"Protocol manifest {manifest_path} must contain a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError(f"Protocol manifest {manifest_path} has an unsupported schema version")

    for field in _REQUIRED_STRING_FIELDS:
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise ValueError(f"Protocol manifest {manifest_path} requires a non-empty {field!r}")
    for field in _REQUIRED_LIST_FIELDS:
        if not isinstance(payload.get(field), list) or not payload[field]:
            raise ValueError(f"Protocol manifest {manifest_path} requires a non-empty {field!r} list")

    if payload["status"] not in _ALLOWED_STATUSES:
        raise ValueError(f"Protocol manifest {manifest_path} has an unsupported status")
    if payload["classification"] not in _ALLOWED_CLASSIFICATIONS:
        raise ValueError(f"Protocol manifest {manifest_path} has an unsupported classification")
    if payload.get("runnable") is not False or payload.get("comparable") is not False:
        raise ValueError(f"Protocol manifest {manifest_path} must remain explicitly non-runnable and non-comparable")

    decision_ids: set[str] = set()
    for decision in payload["missing_product_decisions"]:
        if not isinstance(decision, dict) or set(decision) != {"id", "question"}:
            raise ValueError(f"Protocol manifest {manifest_path} has a malformed product decision")
        if not all(isinstance(decision[key], str) and decision[key].strip() for key in ("id", "question")):
            raise ValueError(f"Protocol manifest {manifest_path} has an empty product decision")
        if decision["id"] in decision_ids:
            raise ValueError(f"Protocol manifest {manifest_path} repeats product decision {decision['id']!r}")
        decision_ids.add(decision["id"])

    return ProtocolManifest(path=manifest_path.resolve(), payload=payload)
