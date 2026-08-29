# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import asyncio
import hashlib
import importlib
import io
import random
from types import SimpleNamespace

import pytest

from nemo_skills.dataset.voicebench.prepare import (
    HF_REVISION,
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
    PEDANT_ARTIFACT_SHA256,
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
        assert module.GENERATION_ARGS.endswith("++inference.temperature=0.0")


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


def test_pedant_artifacts_are_pinned_and_network_refresh_is_disabled(tmp_path, monkeypatch):
    assert PEDANT_SOURCE_REVISION == "26623c6ca3313e7a58c48fd6ce2b8579eb0c742e"
    package_dir = tmp_path / "qa_metrics"
    artifact_dir = package_dir / "classifier"
    artifact_dir.mkdir(parents=True)
    pedant_py = package_dir / "pedant.py"
    pedant_py.write_text("# synthetic module\n", encoding="utf-8")

    contents = {
        "lr_classifier.pkl": b"classifier",
        "tf-idf_vectorizer.pkl": b"vectorizer",
    }
    expected = {}
    for filename, content in contents.items():
        (artifact_dir / filename).write_bytes(content)
        expected[filename] = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(
        "nemo_skills.evaluation.evaluator.voicebench.PEDANT_ARTIFACT_SHA256",
        expected,
    )

    module = SimpleNamespace(__file__=str(pedant_py))
    module.download_link = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("network refresh"))
    module.PEDANT = lambda: (module.download_link("file", "url", "name"), "pedant")[1]
    assert load_pinned_pedant(module) == "pedant"
    assert set(PEDANT_ARTIFACT_SHA256) == set(contents)


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
