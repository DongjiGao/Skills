# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import zipfile

import pytest

from nemo_skills.evaluation.evaluator.language_id import evaluate_language_id_sample
from nemo_skills.evaluation.metrics.language_id_metrics import LanguageIDMetrics

prepare = importlib.import_module("nemo_skills.dataset.voxlingua107.prepare")


def _evaluated(label: str, generation: str) -> dict:
    sample = {"expected_answer": label, "generation": generation}
    return {**sample, **evaluate_language_id_sample(sample, prepare.CANONICAL_LABELS)}


def test_official_archive_identity_and_label_cardinalities_are_frozen():
    assert prepare.SOURCE_SHA256 == "54c352d2ecbcbc03a4bb24151750a287573a5c2738636387a2240700bb89d2d2"
    assert prepare.EXPECTED_ROWS == 1609
    assert len(prepare.CANONICAL_LABELS) == 33
    assert prepare.EXPECTED_LABEL_COUNTS["ar"] == 100
    assert prepare.EXPECTED_LABEL_COUNTS["pt"] == 1
    assert prepare.EXPECTED_LABEL_COUNTS["zh"] == 23


def test_prompt_exposes_only_the_canonical_label_space():
    record = prepare._record("en", "sample.wav", 1.5, "/data/voxlingua107/audio")
    assert record["expected_answer"] == "en"
    assert record["messages"][0]["content"] == prepare.PROMPT
    assert record["messages"][0]["audio"]["path"].endswith("/en/sample.wav")
    for label in prepare.CANONICAL_LABELS:
        assert label in prepare.PROMPT


def test_archive_member_validation_rejects_traversal_and_unknown_labels():
    with pytest.raises(ValueError, match="Unsafe"):
        prepare._member_identity("../en/sample.wav")
    with pytest.raises(ValueError, match="Unexpected"):
        prepare._member_identity("xx/sample.wav")
    with pytest.raises(ValueError, match="Unexpected"):
        prepare._member_identity("en/sample.txt")


def test_inventory_requires_the_exact_official_cardinality(tmp_path):
    archive_path = tmp_path / "small.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("en/sample.wav", b"RIFF")
    with zipfile.ZipFile(archive_path) as archive, pytest.raises(ValueError, match="cardinalities changed"):
        prepare._inventory(archive)


def test_exact_label_and_micro_macro_scoring():
    metrics = LanguageIDMetrics(max_k=1)
    for prediction in [
        _evaluated("en", "en"),
        _evaluated("en", "EN"),
        _evaluated("fr", "fr"),
        _evaluated("de", "fr"),
    ]:
        metrics.update([prediction])
    result = metrics.get_metrics()["pass@1"]
    assert result["accuracy_micro"] == 50.0
    assert result["accuracy_macro"] == 50.0
    assert result["accuracy::en"] == 50.0
    assert result["accuracy::fr"] == 100.0
    assert result["accuracy::de"] == 0.0
