# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import importlib
import json

import pytest

from nemo_skills.dataset.protocol_manifest import ProtocolDecisionRequired, load_protocol_manifest


@pytest.mark.parametrize(
    "dataset_name,classification",
    [
        ("aegis-v2-audio", "public-artifact-no-comparable-speechlm-protocol"),
        ("garak", "scanner-no-fixed-product-benchmark"),
    ],
)
def test_unresolved_safety_protocols_are_discoverable_but_not_runnable(dataset_name, classification):
    module = importlib.import_module(f"nemo_skills.dataset.{dataset_name}")

    assert module.PROTOCOL_STATUS == "product-decision-required"
    assert module.PROTOCOL_CLASSIFICATION == classification
    assert module.IS_RUNNABLE_BENCHMARK is False
    assert not hasattr(module, "METRICS_TYPE")
    assert not hasattr(module, "SCORE_MODULE")
    assert not hasattr(module, "BENCHMARKS")
    with pytest.raises(ProtocolDecisionRequired, match="Product decisions required"):
        module.require_runnable_protocol()


@pytest.mark.parametrize("dataset_name", ["aegis-v2-audio", "garak"])
def test_prepare_entry_points_fail_before_materializing_a_fake_benchmark(dataset_name):
    prepare = importlib.import_module(f"nemo_skills.dataset.{dataset_name}.prepare")

    with pytest.raises(ProtocolDecisionRequired, match=dataset_name):
        prepare.main()


def test_aegis_audio_public_artifact_is_pinned_without_inventing_a_score():
    module = importlib.import_module("nemo_skills.dataset.aegis-v2-audio")
    payload = module.PROTOCOL_MANIFEST.payload
    artifact = payload["public_artifacts"][0]

    assert artifact == {
        "kind": "huggingface_dataset",
        "repository": "nvidia/Nemotron-Content-Safety-Audio-Dataset",
        "revision": "8b90c949f88f9d90d5157025bc7695229d512206",
        "license": "cc-by-4.0",
        "split": "test",
        "metadata_path": "metadata.csv",
        "metadata_sha256": "ff80a362bd88cd1f2d0113f03509f53cf984ee434e909383caa42b20b2c011f0",
        "metadata_bytes": 1330539,
        "rows": 1928,
        "audio_files": 1928,
        "repository_storage_bytes": 2951859829,
        "prompt_label_counts": {"safe": 889, "unsafe": 1039},
        "speaker_native_language_counts": {"English": 648, "Spanish": 642, "French": 638},
    }
    assert {decision["id"] for decision in payload["missing_product_decisions"]} == {
        "task_contract",
        "target_policy",
        "prompt_contract",
        "scorer_contract",
        "aggregation_contract",
        "prd_mapping",
    }
    assert payload["runnable"] is False
    assert payload["comparable"] is False


def test_garak_release_is_pinned_but_scanner_choices_remain_explicit():
    module = importlib.import_module("nemo_skills.dataset.garak")
    payload = module.PROTOCOL_MANIFEST.payload
    code, audio = payload["public_artifacts"]

    assert code["tag"] == "v0.16.0"
    assert code["revision"] == "dbe4515d12664f2e34ac2cea295f055c22fe82b4"
    assert audio["observed_revision"] == "2d08e441a61f584510886ccfa15bf5b94f37480f"
    assert audio["upstream_loader_revision"] is None
    assert {decision["id"] for decision in payload["missing_product_decisions"]} == {
        "target_policy",
        "speechlm_adapter",
        "probe_allowlist",
        "detector_allowlist",
        "run_contract",
        "audio_payload",
        "reporting_contract",
    }
    assert any("cross-probe" in claim for claim in payload["prohibited_claims"])


def test_protocol_manifest_validation_cannot_be_flipped_to_runnable(tmp_path):
    source = importlib.import_module("nemo_skills.dataset.garak").PROTOCOL_MANIFEST.payload
    payload = json.loads(json.dumps(source))
    payload["runnable"] = True
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="non-runnable and non-comparable"):
        load_protocol_manifest(path)


def test_protocol_manifest_requires_unique_explicit_product_decisions(tmp_path):
    source = importlib.import_module("nemo_skills.dataset.aegis-v2-audio").PROTOCOL_MANIFEST.payload
    payload = json.loads(json.dumps(source))
    payload["missing_product_decisions"].append(payload["missing_product_decisions"][0])
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="repeats product decision"):
        load_protocol_manifest(path)
