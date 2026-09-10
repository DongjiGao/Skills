# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import asyncio
import hashlib
import importlib
import io
import json

import numpy as np
import pytest
import soundfile as sf

from nemo_skills.evaluation.evaluator import get_evaluator_class
from nemo_skills.evaluation.metrics.map_metrics import get_metrics
from nemo_skills.inference.generate import GenerationTask

dataset = importlib.import_module("nemo_skills.dataset.pii-asr-eval-vctk-vits")
prepare = importlib.import_module("nemo_skills.dataset.pii-asr-eval-vctk-vits.prepare")
scorer = importlib.import_module("nemo_skills.dataset.pii-asr-eval-vctk-vits.scorer")
two_stage_dataset = importlib.import_module("nemo_skills.dataset.pii-asr-eval-vctk-vits-two-stage")
two_stage_scorer = importlib.import_module("nemo_skills.dataset.pii-asr-eval-vctk-vits-two-stage.scorer")
two_stage_generation = importlib.import_module("nemo_skills.inference.eval.pii_asr_two_stage")


def _wav_bytes():
    output = io.BytesIO()
    sf.write(output, np.zeros(22050, dtype=np.float32), 22050, format="WAV", subtype="PCM_16")
    return output.getvalue()


def _row(payload):
    transcript = "Use alpha one as my test PIN."
    start = transcript.index("alpha")
    return {
        "id": "pii-eval-pin-000",
        "audio_filepath": "/irrelevant/audio/pii-eval-pin-000.wav",
        "duration": 1.0,
        "sample_rate": 22050,
        "speaker_id": "p225",
        "category": "pin",
        "transcript": transcript,
        "redacted_text": "Use <pin> as my test PIN.",
        "entities": [
            {
                "start": start,
                "end": start + len("alpha one"),
                "text": "alpha one",
                "label": "pin",
                "placeholder": "<pin>",
            }
        ],
        "audio_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _mini_contract(monkeypatch, metadata, row):
    monkeypatch.setattr(prepare, "SOURCE_METADATA_SHA256", hashlib.sha256(metadata.read_bytes()).hexdigest())
    monkeypatch.setattr(prepare, "EXPECTED_ROWS", 1)
    monkeypatch.setattr(prepare, "EXPECTED_ENTITIES", 1)
    monkeypatch.setattr(prepare, "EXPECTED_CATEGORIES", 1)
    monkeypatch.setattr(prepare, "EXPECTED_ROWS_PER_CATEGORY", 1)
    monkeypatch.setattr(prepare, "EXPECTED_SPEAKERS", 1)
    monkeypatch.setattr(prepare, "EXPECTED_DURATION_SECONDS", row["duration"])


def test_fixed_source_contract():
    assert prepare.SOURCE_METADATA_SHA256 == "c4eaf9a8926377a583c5a221551e79c92fa81a54ba55a267575db37925b53ce0"
    assert (prepare.EXPECTED_ROWS, prepare.EXPECTED_ENTITIES, prepare.EXPECTED_CATEGORIES) == (1000, 1080, 25)
    assert prepare.EXPECTED_SAMPLE_RATE == 22050


def test_prepare_publishes_only_minimal_closure(tmp_path, monkeypatch):
    source = tmp_path / "source"
    audio = source / "audio"
    audio.mkdir(parents=True)
    payload = _wav_bytes()
    row = _row(payload)
    (audio / f"{row['id']}.wav").write_bytes(payload)
    metadata = source / "metadata.jsonl"
    metadata.write_text(json.dumps(row) + "\n", encoding="utf-8")
    _mini_contract(monkeypatch, metadata, row)
    output = tmp_path / "prepared"
    manifest = prepare.prepare_dataset(source, output, "/data/pii-asr-eval-vctk-vits")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    receipt = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert sorted(path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()) == [
        "audio/pii-eval-pin-000.wav",
        "provenance.json",
        "test.jsonl",
    ]
    assert record["expected_answer"] == "Use <pin> as my test PIN."
    assert record["expected_placeholders"] == ["<pin>"]
    assert record["messages"][1]["audio"] == {
        "path": "/data/pii-asr-eval-vctk-vits/audio/pii-eval-pin-000.wav",
        "duration": 1.0,
    }
    assert receipt["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_source_and_audio_hashes_fail_closed(tmp_path, monkeypatch):
    source = tmp_path / "source"
    audio = source / "audio"
    audio.mkdir(parents=True)
    payload = _wav_bytes()
    row = _row(payload)
    (audio / f"{row['id']}.wav").write_bytes(b"corrupt")
    metadata = source / "metadata.jsonl"
    metadata.write_text(json.dumps(row) + "\n", encoding="utf-8")
    _mini_contract(monkeypatch, metadata, row)
    with pytest.raises(ValueError, match="audio hash mismatch"):
        prepare.prepare_dataset(source, tmp_path / "prepared", "/data/pii-asr-eval-vctk-vits")
    monkeypatch.setattr(prepare, "SOURCE_METADATA_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="metadata hash drift"):
        prepare.prepare_dataset(source, tmp_path / "prepared", "/data/pii-asr-eval-vctk-vits")


def _evaluated(reference, generation):
    row = {
        "expected_answer": reference,
        "generation": generation,
        "expected_placeholders": ["<email>"],
        "pii_entity_texts": ["name at example dot com"],
    }
    return {**row, **scorer.evaluate_sample(row)}


def test_placeholder_metrics_and_conservative_entity_leakage():
    metrics = scorer.PIIASRRedactionMetrics(max_k=1)
    metrics.update([_evaluated("Email is <email>.", "Email is <email>.")])
    metrics.update([_evaluated("Email is <email>.", "Email is name at example dot com.")])
    values = metrics.get_metrics()["pass@1"]
    assert (values["placeholder_precision"], values["placeholder_recall"], values["placeholder_f1"]) == (
        100.0,
        50.0,
        66.67,
    )
    assert values["placeholder_exact_match"] == 50.0
    assert values["entity_leakage_rate"] == 50.0
    assert values["samples_with_entity_leakage"] == 1


def test_wrong_placeholder_type_is_penalized():
    prediction = _evaluated("Email is <email>.", "Email is <phone-number>.")
    assert scorer.normalize_with_placeholders("Email is <email>.").split()[-1] == "<email>"
    assert (prediction["placeholder_tp"], prediction["placeholder_fp"], prediction["placeholder_fn"]) == (0, 1, 1)
    assert prediction["wer_errors"] == 1


def test_dataset_registers_custom_evaluator_and_metrics():
    assert dataset.REQUIRES_DATA_DIR is True
    assert isinstance(get_metrics(dataset.METRICS_TYPE, max_k=1), scorer.PIIASRRedactionMetrics)
    assert isinstance(
        get_evaluator_class("nemo_skills.dataset.pii-asr-eval-vctk-vits.scorer::PIIASRRedactionEvaluator", {}),
        scorer.PIIASRRedactionEvaluator,
    )


def test_prepare_two_stage_reuses_audio_and_preserves_training_turns(tmp_path, monkeypatch):
    source = tmp_path / "source"
    audio = source / "audio"
    audio.mkdir(parents=True)
    payload = _wav_bytes()
    row = _row(payload)
    (audio / f"{row['id']}.wav").write_bytes(payload)
    metadata = source / "metadata.jsonl"
    metadata.write_text(json.dumps(row) + "\n", encoding="utf-8")
    _mini_contract(monkeypatch, metadata, row)
    output = tmp_path / "prepared"
    manifest = prepare.prepare_dataset(
        source,
        output,
        "/data/pii-asr-eval-vctk-vits",
        scenario="two-stage",
        publish_audio=False,
    )
    record = json.loads(manifest.read_text(encoding="utf-8"))
    receipt = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert sorted(path.name for path in output.iterdir()) == ["provenance.json", "test.jsonl"]
    assert record["messages"][1]["content"] == prepare.TRANSCRIBE_PROMPT
    assert record["redaction_prompt"] == prepare.REDACTION_FOLLOWUP_PROMPT
    assert record["expected_asr"] == row["transcript"]
    assert record["expected_answer"] == row["redacted_text"]
    assert receipt["task"] == "model-ASR-then-model-redaction"
    assert receipt["audio_published_here"] is False


def test_prepare_two_stage_pii_aware_names_all_categories(tmp_path, monkeypatch):
    source = tmp_path / "source"
    audio = source / "audio"
    audio.mkdir(parents=True)
    payload = _wav_bytes()
    row = _row(payload)
    (audio / f"{row['id']}.wav").write_bytes(payload)
    metadata = source / "metadata.jsonl"
    metadata.write_text(json.dumps(row) + "\n", encoding="utf-8")
    _mini_contract(monkeypatch, metadata, row)
    output = tmp_path / "prepared"
    manifest = prepare.prepare_dataset(
        source, output, "/data/pii-asr-eval-vctk-vits", scenario="two-stage-pii-aware", publish_audio=False
    )
    record = json.loads(manifest.read_text(encoding="utf-8"))
    prompt = record["messages"][1]["content"]
    assert prompt == prepare.PII_AWARE_TRANSCRIBE_PROMPT
    assert "Do not redact" in prompt
    for expected in (
        "API key",
        "bank account",
        "bank routing number",
        "card security code",
        "certificate or license number",
        "credit or debit card",
        "customer ID",
        "spelled email",
        "spoken email",
        "employee ID",
        "fax number",
        "health-plan beneficiary number",
        "HTTP cookie",
        "medical-record number",
        "mixed identifiers",
        "national ID",
        "password",
        "domestic phone",
        "international phone",
        "PIN",
        "Social Security number",
        "UK street address",
        "US street address",
        "SWIFT or BIC",
        "tax ID",
    ):
        assert expected in prompt
    receipt = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert receipt["dataset"] == prepare.TWO_STAGE_PII_AWARE_BENCHMARK_NAME
    assert receipt["task"] == "PII-aware-model-ASR-then-model-redaction"


def test_prepare_two_stage_pii_brief_uses_exact_prompt(tmp_path, monkeypatch):
    source = tmp_path / "source"
    audio = source / "audio"
    audio.mkdir(parents=True)
    payload = _wav_bytes()
    row = _row(payload)
    (audio / f"{row['id']}.wav").write_bytes(payload)
    metadata = source / "metadata.jsonl"
    metadata.write_text(json.dumps(row) + "\n", encoding="utf-8")
    _mini_contract(monkeypatch, metadata, row)
    output = tmp_path / "prepared"
    manifest = prepare.prepare_dataset(
        source, output, "/data/pii-asr-eval-vctk-vits", scenario="two-stage-pii-brief", publish_audio=False
    )
    record = json.loads(manifest.read_text(encoding="utf-8"))
    assert record["messages"][1]["content"] == (
        "Produce an exact, verbatim transcript of the audio. It may contain PII categories such as API key, "
        "banking numbers, SSN, emails, IDs, phone numbers, card numbers, or PIN."
    )
    receipt = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert receipt["dataset"] == prepare.TWO_STAGE_PII_BRIEF_BENCHMARK_NAME
    assert receipt["task"] == "brief-PII-aware-model-ASR-then-model-redaction"


def test_two_stage_generation_uses_model_asr_for_followup(monkeypatch):
    calls = []
    responses = [
        {
            "generation": "Model ASR hypothesis",
            "finish_reason": "stop",
            "num_input_tokens": 10,
            "num_generated_tokens": 3,
        },
        {"generation": "Redacted result", "finish_reason": "stop", "num_input_tokens": 20, "num_generated_tokens": 4},
    ]

    async def fake_process(self, data_point, all_data, prompt_format=None):
        calls.append(data_point)
        return responses[len(calls) - 1].copy()

    monkeypatch.setattr(GenerationTask, "process_single_datapoint", fake_process)
    task = object.__new__(two_stage_generation.PIIASRTwoStageGenerationTask)
    task.cfg = type("Cfg", (), {"generation_key": "generation"})()
    task.fill_prompt = lambda data_point, all_data: data_point["messages"]
    data_point = {
        "messages": [{"role": "user", "content": "Transcribe", "audio": {"path": "/data/x.wav"}}],
        "redaction_prompt": "Redact",
    }
    result = asyncio.run(task.process_single_datapoint(data_point, []))
    assert calls[1]["messages"][-2:] == [
        {"role": "assistant", "content": "Model ASR hypothesis"},
        {"role": "user", "content": "Redact"},
    ]
    assert result["intermediate_asr"] == "Model ASR hypothesis"
    assert result["generation"] == "Redacted result"
    assert (result["num_input_tokens"], result["num_generated_tokens"]) == (30, 7)


def test_two_stage_metrics_include_first_stage_asr():
    row = {
        "expected_answer": "Use <pin>.",
        "generation": "Use <pin>.",
        "expected_placeholders": ["<pin>"],
        "pii_entity_texts": ["alpha one"],
        "expected_asr": "Use alpha one.",
        "intermediate_asr": "Use alpha two.",
    }
    prediction = {**row, **two_stage_scorer.evaluate_sample(row)}
    metrics = two_stage_scorer.PIIASRTwoStageMetrics(max_k=1)
    metrics.update([prediction])
    values = metrics.get_metrics()["pass@1"]
    assert values["placeholder_f1"] == 100.0
    assert values["first_stage_asr_wer"] == 33.33
    assert values["first_stage_missing"] == 0
    assert two_stage_dataset.GENERATION_MODULE == "nemo_skills.inference.eval.pii_asr_two_stage"
