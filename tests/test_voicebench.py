# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import asyncio
import hashlib
import importlib
import io
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_skills.dataset.voicebench.ifeval import GENERATION_ARGS as IFEVAL_GENERATION_ARGS
from nemo_skills.dataset.voicebench.prepare import (
    EXPECTED_ROWS,
    HF_REVISION,
    MMSU_EXPECTED_ROWS,
    MMSU_SPLITS,
    OFFICIAL_CODE_REVISION,
    PROFILES,
    SD_QA_SPLITS,
    SPEAKER_SPLITS,
    _audio_duration_seconds,
    format_entry,
)
from nemo_skills.evaluation.evaluator.voicebench import (
    JUDGE_MODEL,
    JUDGE_REQUEST_KWARGS,
    JUDGE_SYSTEM_PROMPT,
    OPEN_JUDGE_TEMPLATE,
    PEDANT_PORTABLE_ARTIFACT_SHA256,
    PEDANT_SOURCE_ARTIFACT_SHA256,
    PEDANT_SOURCE_REVISION,
    QA_JUDGE_TEMPLATE,
    VoiceBenchEvaluator,
    extract_open_rating,
    load_pinned_pedant,
    official_majority_vote,
)
from nemo_skills.evaluation.evaluator.voicebench_bbh_official import BBHEvaluator
from nemo_skills.evaluation.evaluator.voicebench_mcq_official import MCQEvaluator
from nemo_skills.evaluation.metrics.voicebench_metrics import VoiceBenchMetrics


def test_public_leaf_inventory_and_revisions_are_pinned():
    assert HF_REVISION == "b02edcef1330480be3a11bd6f7434ac32f05ad08"
    assert OFFICIAL_CODE_REVISION == "3c3b0d3a7a956f745305eb348f5e03ce7ec73dad"
    assert len(SPEAKER_SPLITS) == 11
    assert len(MMSU_SPLITS) == 12
    assert set(MMSU_SPLITS) == set(MMSU_EXPECTED_ROWS)
    assert {split: EXPECTED_ROWS[f"mmsu/{split}"] for split in MMSU_SPLITS} == MMSU_EXPECTED_ROWS
    assert len(SD_QA_SPLITS) == 11
    assert set(PROFILES) == {
        "advbench",
        "alpacaeval",
        "alpacaeval-full",
        "alpacaeval-speaker",
        "bbh",
        "commoneval",
        "ifeval",
        "mmsu",
        "mtbench",
        "openbookqa",
        "sd-qa",
        "wildvoice",
    }


def test_runnable_manifest_is_audio_only_and_preserves_judge_metadata():
    row = {"audio": {"bytes": b"not-json"}, "prompt": "secret transcription", "reference": "A"}
    item = format_entry(row, "openbookqa", "openbookqa", "test", ["/data/voicebench/x.wav"], [1.25])
    assert item["prompt"] == "secret transcription"
    assert item["reference"] == "A"
    assert "audio" not in item
    assert item["messages"] == [
        {"role": "user", "content": "", "audio": {"path": "/data/voicebench/x.wav", "duration": 1.25}}
    ]
    assert "secret transcription" not in str(item["messages"])


def test_duration_is_read_from_decode_disabled_audio_bytes(tmp_path):
    import soundfile

    buffer = io.BytesIO()
    soundfile.write(buffer, [0.0] * 16000, 16000, format="WAV", subtype="PCM_16")
    assert _audio_duration_seconds({"bytes": buffer.getvalue()}, tmp_path / "missing.wav") == 1.0


def test_mtbench_is_data_only_instead_of_inventing_two_turn_protocol():
    row = {"audio1": {"bytes": b"1"}, "audio2": {"bytes": b"2"}, "turns": ["one", "two"]}
    item = format_entry(row, "mtbench", "mtbench", "test", ["/a.wav", "/b.wav"], [1.0, 2.0])
    assert item["messages"] == []
    assert item["protocol_supported"] is False
    assert "no working MTBench inference path or scorer" in item["protocol_gap"]
    assert item["audio1_path"] == "/a.wav"
    assert item["audio2_path"] == "/b.wav"
    assert item["audio1_duration"] == 1.0
    assert item["audio2_duration"] == 2.0


def test_every_runnable_profile_module_is_registered():
    group = importlib.import_module("nemo_skills.dataset.voicebench")
    assert "voicebench.mtbench" not in group.BENCHMARKS
    assert "voicebench.mtbench" in group.PUBLIC_DATA_ONLY_BENCHMARKS
    for benchmark in group.BENCHMARKS:
        module = importlib.import_module(f"nemo_skills.dataset.{benchmark}")
        assert "++inference.temperature=0.0" in module.GENERATION_ARGS


def test_ifeval_writes_response_and_tolerates_legacy_or_soft_failed_rows():
    pytest.importorskip("immutabledict")
    from nemo_skills.evaluation.evaluator.voicebench_ifeval import read_prompt_to_response_dict

    assert "++generation_key=response" in IFEVAL_GENERATION_ARGS
    assert read_prompt_to_response_dict(
        [
            {"prompt": "new", "response": "<1> exact <|user|>"},
            {"prompt": "legacy", "generation": "older artifact"},
            {"prompt": "soft-fail", "error": "HTTP 500"},
        ]
    ) == {
        "new": "exact",
        "legacy": "older artifact",
        "soft-fail": "",
    }


def test_exact_judge_config_and_parsers():
    assert JUDGE_MODEL == "gpt-4o-mini"
    assert JUDGE_SYSTEM_PROMPT == "You are a helpful assistant who tries to help answer the user's question."
    assert JUDGE_REQUEST_KWARGS == {
        "model": "gpt-4o-mini",
        "max_tokens": 1024,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": None,
        "temperature": 0.5,
        "top_p": 0.95,
        "n": 3,
    }
    assert "### [Instruction]: {prompt}" in OPEN_JUDGE_TEMPLATE
    assert "### [Response]: {response}" in OPEN_JUDGE_TEMPLATE
    assert QA_JUDGE_TEMPLATE.endswith('Please only output a single "Yes" or "No". Do not output anything else.')
    assert extract_open_rating("4") == 4.0
    assert extract_open_rating("score [[5]]") == 5.0
    with pytest.raises(ValueError):
        extract_open_rating("five")
    assert official_majority_vote(["Yes", "yes", "No"]) is True


def test_portable_pedant_resources_are_in_wheel_package_data():
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert '"evaluation/evaluator/resources/voicebench_pedant/*"' in pyproject


def test_pedant_artifacts_are_pinned_and_portable_without_constructor(tmp_path, monkeypatch):
    import numpy as np

    assert PEDANT_SOURCE_REVISION == "26623c6ca3313e7a58c48fd6ce2b8579eb0c742e"
    assert set(PEDANT_SOURCE_ARTIFACT_SHA256) == {"lr_classifier.pkl", "tf-idf_vectorizer.pkl"}
    artifact_dir = tmp_path / "portable"
    artifact_dir.mkdir(parents=True)
    np.save(artifact_dir / "classes.npy", np.asarray(["correct", "incorrect"]), allow_pickle=False)
    np.save(artifact_dir / "coef.npy", np.asarray([[-1.0, 0.0, 0.0, 0.0]]), allow_pickle=False)
    np.save(artifact_dir / "idf.npy", np.asarray([1.0]), allow_pickle=False)
    np.save(artifact_dir / "intercept.npy", np.asarray([0.0]), allow_pickle=False)
    documents = {
        "model.json": {
            "classes": ["correct", "incorrect"],
            "coef_shape": [1, 4],
            "intercept_shape": [1],
            "vocabulary_size": 1,
            "idf_size": 1,
        },
        "vectorizer.json": {
            "analyzer": "word",
            "binary": False,
            "decode_error": "strict",
            "encoding": "utf-8",
            "input": "content",
            "lowercase": True,
            "ngram_range": [1, 1],
            "norm": "l2",
            "smooth_idf": True,
            "stop_words": None,
            "strip_accents": None,
            "sublinear_tf": False,
            "token_pattern": r"(?u)\b\w\w+\b",
            "use_idf": True,
        },
        "vocabulary.json": {"paris": 0},
        "parity.json": {
            "texts": ["paris"],
            "extras": [[0.0, 0.0, 0.0]],
            "decision": [-1.0],
            "prediction": ["correct"],
        },
    }
    for filename, content in documents.items():
        (artifact_dir / filename).write_text(
            json.dumps(content, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
        )
    expected = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in artifact_dir.iterdir()}
    monkeypatch.setattr(
        "nemo_skills.evaluation.evaluator.voicebench.PEDANT_PORTABLE_ARTIFACT_SHA256",
        expected,
    )

    class FakePedant:
        def __init__(self):
            raise AssertionError("legacy PEDANT constructor must not run")

    pedant = load_pinned_pedant(SimpleNamespace(PEDANT=FakePedant), portable_dir=artifact_dir)
    features = pedant.tokenizer.transform(["paris"])
    from scipy.sparse import hstack

    features = hstack([features, np.zeros((1, 3))])
    assert pedant.model.predict(features).tolist() == ["correct"]
    assert set(PEDANT_PORTABLE_ARTIFACT_SHA256) == set(documents) | {
        "classes.npy", "coef.npy", "idf.npy", "intercept.npy"
    }


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Completions:
    def __init__(self):
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return type("Completion", (), {"choices": [_Choice("5"), _Choice("4"), _Choice("5")]})()


class _FakeClient:
    def __init__(self):
        self.chat = type("Chat", (), {})()
        self.chat.completions = _Completions()


def test_open_eval_sends_exact_chat_completion_request():
    evaluator = VoiceBenchEvaluator({"judge_success_sleep_seconds": 0, "judge_error_sleep_seconds": 0})
    evaluator._client = _FakeClient()
    point = {"voicebench_task": "commoneval", "prompt": "Question", "generation": "Answer"}
    result = asyncio.run(evaluator.eval_single(point))
    request = evaluator._client.chat.completions.request
    assert {key: request[key] for key in JUDGE_REQUEST_KWARGS} == JUDGE_REQUEST_KWARGS
    assert request["messages"][0] == {"role": "system", "content": JUDGE_SYSTEM_PROMPT}
    assert result["score"] == ["5", "4", "5"]
    assert result["voicebench_open_ratings"] == [5.0, 4.0, 5.0]


@pytest.mark.parametrize(
    ("response", "answer"),
    [("The correct answer is B.", "B"), ("答案是选项c", "C"), ("d", "D"), ("unparseable", None)],
)
def test_vendored_official_mcq_parser(response, answer):
    assert MCQEvaluator().extract_answer(response) == answer


def test_mcq_exposes_deterministic_and_official_random_parity(monkeypatch):
    evaluator = VoiceBenchEvaluator({})
    monkeypatch.setattr(random, "choice", lambda choices: "A")
    point = {
        "voicebench_task": "openbookqa",
        "reference": "A",
        "generation": "unparseable",
    }
    result = asyncio.run(evaluator.eval_single(point))
    assert result["voicebench_parse_success"] is False
    assert result["voicebench_accuracy_unparsed_wrong"] is False
    assert result["voicebench_random_fallback_parity_answer"] == "A"
    assert result["voicebench_accuracy_random_fallback_parity"] is True


def test_bbh_parser_diagnostics_preserve_official_parsed_cases(monkeypatch):
    assert BBHEvaluator().extract_answer("The answer is (a)", "hyperbaton") == 0
    evaluator = VoiceBenchEvaluator({})
    monkeypatch.setattr(random, "choice", lambda choices: 1)
    point = {"voicebench_task": "bbh", "id": "hyperbaton", "reference": "(b)", "generation": "???"}
    result = asyncio.run(evaluator.eval_single(point))
    assert result["voicebench_parse_success"] is False
    assert result["voicebench_accuracy_unparsed_wrong"] is False
    assert result["voicebench_accuracy_random_fallback_parity"] is True


def test_task_metrics_do_not_create_cross_task_overall():
    metric = VoiceBenchMetrics()
    metric.update(
        [
            {
                "voicebench_task": "openbookqa",
                "official_code_revision": OFFICIAL_CODE_REVISION,
                "voicebench_parse_success": False,
                "voicebench_accuracy_unparsed_wrong": False,
                "voicebench_accuracy_random_fallback_parity": True,
            }
        ]
    )
    values = metric.get_metrics()["pass@1"]
    assert values["accuracy_unparsed_wrong"] == 0.0
    assert values["parse_rate"] == 0.0
    assert values["unparsed_rate"] == 100.0
    assert values["accuracy_random_fallback_parity"] == 100.0
    with pytest.raises(ValueError, match="cannot mix tasks"):
        metric.update([{"voicebench_task": "bbh", "official_code_revision": OFFICIAL_CODE_REVISION}])
