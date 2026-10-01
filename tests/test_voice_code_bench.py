# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import typing
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_skills.evaluation.evaluator import voice_code_bench as voice_code_bench_scorer
from nemo_skills.evaluation.evaluator.voice_code_bench import (
    OFFICIAL_REVISION,
    VoiceCodeBenchEvaluator,
    _ensure_typing_not_required,
    _load_official_package,
    compute_format_invariant_wer,
)
from nemo_skills.evaluation.evaluator.voice_code_bench_deterministic import (
    DETERMINISTIC_VERIFIER_DIGEST,
    DETERMINISTIC_VERIFIER_ID,
    verify_deterministic_entity_matches,
)
from nemo_skills.evaluation.metrics.voice_code_bench_metrics import VoiceCodeBenchMetrics


def _phone_datapoint():
    return {
        "audio_id": "phone",
        "transcripts": {
            "template": "Call area code <phone_e01> tomorrow",
            "acoustic": "Call area code two one two five five five zero one zero zero tomorrow",
            "canonical": "Call area code (212) 555-0100 tomorrow",
        },
        "entities": [
            {
                "id": "phone_e01",
                "type": "phone_number",
                "role": "callback_number",
                "acoustic": "two one two five five five zero one zero zero",
                "canonical": "(212) 555-0100",
            }
        ],
    }


def _load_prepare_module():
    path = Path(__file__).parents[1] / "nemo_skills" / "dataset" / "voice-code-bench" / "prepare.py"
    spec = spec_from_file_location("voice_code_bench_prepare", path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_voice_code_bench_uses_release_verifier():
    path = Path(__file__).parents[1] / "nemo_skills" / "dataset" / "voice-code-bench" / "__init__.py"
    spec = spec_from_file_location("voice_code_bench_benchmark", path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.EVAL_ARGS.split() == [
        "++eval_type=voice_code_bench",
        "++eval_config.entity_verifier_mode=live-fill",
        "++eval_config.verifier_id=openai_gpt_5_5_v1",
        "++eval_config.verifier_provider=nvidia-inference-hub",
    ]
    evaluator = VoiceCodeBenchEvaluator({})
    assert evaluator.eval_config.entity_verifier_mode == "live-fill"
    assert evaluator.eval_config.verifier_id == "openai_gpt_5_5_v1"
    assert evaluator.eval_config.verifier_provider == "nvidia-inference-hub"


@dataclass(frozen=True)
class _VerifierConfig:
    id: str = "openai_gpt_5_5_v1"
    model: str = "gpt-5.5"
    endpoint_or_api: str = "https://api.openai.com/v1/chat/completions"

    @property
    def digest(self):
        return f"{self.id}:{self.model}:{self.endpoint_or_api}"


@pytest.mark.parametrize(
    ("hub_key", "openai_key", "provider", "expected_id", "expected_model", "expected_endpoint", "expected_key"),
    [
        (
            "hub-key", "openai-key", None, "nvidia_inference_hub_gpt_5_5_v1", "openai/openai/gpt-5.5",
            "https://inference-api.nvidia.com/v1/chat/completions", "hub-key",
        ),
        (
            "hub-key", "openai-key", "openai", "openai_gpt_5_5_v1", "gpt-5.5",
            "https://api.openai.com/v1/chat/completions", "openai-key",
        ),
    ],
)
def test_default_voice_code_bench_scores_with_pinned_live_fill_verifier(
    tmp_path, monkeypatch, hub_key, openai_key, provider, expected_id, expected_model, expected_endpoint, expected_key
):
    root = tmp_path / "voice-code-bench"
    root.mkdir()
    (root / "NEMO_SKILLS_REVISION").write_text(OFFICIAL_REVISION, encoding="utf-8")
    datapoint = _phone_datapoint()
    input_path = tmp_path / "predictions.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "audio_id": "phone",
                "generation": "Call area code 212-555-0100 tomorrow",
                "voice_code_bench_revision": OFFICIAL_REVISION,
                "voice_code_bench": datapoint,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls = []
    verifier_config = _VerifierConfig()

    def verify_entity_matches(*args, **kwargs):
        calls.append(("verify", kwargs["cache_mode"], args[2], args[3]["OPENAI_API_KEY"]))
        return [{"present": True}]

    verifier = SimpleNamespace(
        load_verifier_config=lambda verifier_id: (
            calls.append(("config", verifier_id)) or verifier_config
        ),
        VerifierCache=lambda path: (calls.append(("cache", path)) or object()),
        verify_entity_matches=verify_entity_matches,
    )
    monkeypatch.setattr(
        voice_code_bench_scorer,
        "_load_official_package",
        lambda data_dir: (None, verifier, str(root)),
    )
    monkeypatch.setenv("NVIDIA_INFERENCE_HUB_API_KEY", hub_key)
    monkeypatch.setenv("OPENAI_API_KEY", openai_key)

    config = {"input_file": str(input_path), "data_dir": str(tmp_path)}
    if provider is not None:
        config["verifier_provider"] = provider
    asyncio.run(VoiceCodeBenchEvaluator(config).eval_full())

    result = json.loads(input_path.read_text(encoding="utf-8"))
    assert calls[0] == ("config", "openai_gpt_5_5_v1")
    _, mode, config, key = calls[-1]
    assert mode == "live-fill"
    assert key == expected_key
    assert config.id == expected_id
    assert config.model == expected_model
    assert config.endpoint_or_api == expected_endpoint
    assert result["voice_code_verifier_id"] == expected_id
    assert result["voice_code_verifier_config_digest"] == config.digest
    assert result["voice_code_correct_entities"] == 1


def test_default_voice_code_bench_requires_hub_key_even_if_openai_key_exists(tmp_path, monkeypatch):
    root = tmp_path / "voice-code-bench"
    root.mkdir()
    (root / "NEMO_SKILLS_REVISION").write_text(OFFICIAL_REVISION, encoding="utf-8")
    input_path = tmp_path / "predictions.jsonl"
    input_path.write_text("", encoding="utf-8")
    verifier = SimpleNamespace(
        load_verifier_config=lambda _: _VerifierConfig(),
        VerifierCache=lambda _: object(),
    )
    monkeypatch.setattr(voice_code_bench_scorer, "_load_official_package", lambda _: (None, verifier, str(root)))
    monkeypatch.delenv("NVIDIA_INFERENCE_HUB_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    with pytest.raises(RuntimeError, match="NVIDIA_INFERENCE_HUB_API_KEY"):
        asyncio.run(VoiceCodeBenchEvaluator({"input_file": str(input_path), "data_dir": str(tmp_path)}).eval_full())


def test_format_invariant_wer_accepts_acoustic_and_canonical_entity_forms():
    datapoint = _phone_datapoint()
    acoustic = compute_format_invariant_wer(datapoint, datapoint["transcripts"]["acoustic"])
    canonical = compute_format_invariant_wer(datapoint, "Call area code 212-555-0100 tomorrow")
    assert acoustic.wer == 0
    assert canonical.wer == 0
    assert canonical.reference_words == 14
    assert canonical.hypothesis_words == 7


def test_format_invariant_wer_counts_changed_structured_value():
    result = compute_format_invariant_wer(_phone_datapoint(), "Call area code 212-555-0199 tomorrow")
    assert result.wer > 0
    assert result.substitutions + result.deletions + result.insertions > 0


def test_wer_only_evaluator_does_not_import_official_python_311_package(tmp_path):
    root = tmp_path / "voice-code-bench"
    root.mkdir()
    (root / "NEMO_SKILLS_REVISION").write_text(OFFICIAL_REVISION, encoding="utf-8")
    datapoint = _phone_datapoint()
    input_path = tmp_path / "predictions.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "audio_id": "phone",
                "generation": datapoint["transcripts"]["acoustic"],
                "voice_code_bench_revision": OFFICIAL_REVISION,
                "voice_code_bench": datapoint,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    evaluator = VoiceCodeBenchEvaluator(
        {
            "input_file": str(input_path),
            "data_dir": str(tmp_path),
            "entity_verifier_mode": "off",
        }
    )
    asyncio.run(evaluator.eval_full())

    result = json.loads(input_path.read_text(encoding="utf-8"))
    assert result["voice_code_wer_reference"] == "acoustic_or_canonical_entity_lattice"
    assert result["voice_code_wer_errors"] == 0
    assert result["voice_code_entity_verifier_status"] == "off"


def test_deterministic_verifier_accepts_canonical_and_spoken_phone_forms():
    datapoint = _phone_datapoint()
    for transcript in (
        "Call area code 212-555-0100 tomorrow",
        "Call area code two one two five five five oh one zero zero tomorrow",
    ):
        match = verify_deterministic_entity_matches(datapoint, transcript)[0]
        assert match["present"] is True
        assert match["canonical"] == "(212) 555-0100"

    changed = verify_deterministic_entity_matches(
        datapoint, "Call area code 212-555-0199 tomorrow"
    )[0]
    assert changed["present"] is False


def test_deterministic_spelled_sequence_requires_explicit_exact_letters():
    datapoint = {
        "entities": [
            {
                "id": "name_e01",
                "type": "spelled_sequence",
                "role": "requester_name_spelling",
                "acoustic": "T A R A V O S S",
                "canonical": "T A R A V O S S",
            }
        ]
    }
    assert verify_deterministic_entity_matches(
        datapoint, "The spelling is T-A-R-A-V-O-S-S."
    )[0]["present"]
    assert not verify_deterministic_entity_matches(
        datapoint, "The caller is Tara Voss."
    )[0]["present"]
    assert not verify_deterministic_entity_matches(
        datapoint, "The caller is Tara Voss; spelling T-A-K-R-A-V-O-S-S."
    )[0]["present"]


def test_deterministic_evaluator_emits_complete_ctem_inputs_without_api_key(
    tmp_path, monkeypatch
):
    root = tmp_path / "voice-code-bench"
    root.mkdir()
    (root / "NEMO_SKILLS_REVISION").write_text(OFFICIAL_REVISION, encoding="utf-8")
    datapoint = _phone_datapoint()
    input_path = tmp_path / "predictions.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "audio_id": "phone",
                "generation": "Call area code 212-555-0100 tomorrow",
                "voice_code_bench_revision": OFFICIAL_REVISION,
                "voice_code_bench": datapoint,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_INFERENCE_HUB_API_KEY", raising=False)

    evaluator = VoiceCodeBenchEvaluator(
        {
            "input_file": str(input_path),
            "data_dir": str(tmp_path),
            "entity_verifier_mode": "deterministic",
        }
    )
    asyncio.run(evaluator.eval_full())

    result = json.loads(input_path.read_text(encoding="utf-8"))
    assert result["voice_code_entity_verifier_status"] == "complete"
    assert result["voice_code_verifier_id"] == DETERMINISTIC_VERIFIER_ID
    assert result["voice_code_verifier_config_digest"] == DETERMINISTIC_VERIFIER_DIGEST
    assert result["voice_code_correct_entities"] == 1
    assert result["voice_code_total_entities"] == 1
    assert result["voice_code_task_success"] is True
    assert result["is_correct"] is True


def test_official_loader_installs_not_required_compatibility(monkeypatch):
    monkeypatch.delattr(typing, "NotRequired", raising=False)

    _ensure_typing_not_required()

    assert typing.NotRequired is not None


def test_official_loader_imports_pinned_package_on_python_310(tmp_path, monkeypatch):
    package = tmp_path / "voice-code-bench" / "scripts" / "voice_code_bench"
    package.mkdir(parents=True)
    (tmp_path / "voice-code-bench" / "NEMO_SKILLS_REVISION").write_text(
        OFFICIAL_REVISION, encoding="utf-8"
    )
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "models.py").write_text(
        "from typing import NotRequired\n", encoding="utf-8"
    )
    (package / "metrics.py").write_text(
        "from .models import NotRequired\n"
        "FORMAT_INVARIANT_WER_REFERENCE = 'acoustic_or_canonical_entity_lattice'\n",
        encoding="utf-8",
    )
    (package / "verifier.py").write_text("", encoding="utf-8")
    monkeypatch.delattr(typing, "NotRequired", raising=False)

    metrics, _, root = _load_official_package(str(tmp_path))

    assert metrics.FORMAT_INVARIANT_WER_REFERENCE == "acoustic_or_canonical_entity_lattice"
    assert root == str(tmp_path / "voice-code-bench")


def test_prepare_manifest_keeps_scorer_metadata_out_of_prompt():
    prepare = _load_prepare_module()
    row = {
        "file_name": "audio/001.wav",
        "audio_id": "contact_001",
        "duration": 12.5,
        "domain": "secret_domain",
        "scenario": "callback",
        "difficulty": "stress",
        "transcripts": {
            "template": "Call <e1>",
            "acoustic": "Call one two three",
            "canonical": "Call 123",
        },
        "entities": [
            {"id": "e1", "type": "plain_number", "role": "id", "acoustic": "one two three", "canonical": "123"}
        ],
        "entity_types": ["plain_number"],
        "entity_count": 1,
    }
    entry = prepare._format_entry(row, "/data/voice-code-bench")
    assert entry["messages"][0] == {"role": "system", "content": prepare.SYSTEM_PROMPT}
    message = entry["messages"][1]
    assert message["content"] == prepare.USER_PROMPT
    assert "secret_domain" not in message["content"]
    assert "123" not in message["content"]
    assert entry["voice_code_bench"]["entities"] == row["entities"]
    assert entry["voice_code_bench_revision"] == OFFICIAL_REVISION


def _prediction(audio_id, errors, ref_words, correct, total, success):
    return {
        "audio_id": audio_id,
        "generation": "example",
        "is_correct": success,
        "voice_code_bench_revision": OFFICIAL_REVISION,
        "voice_code_wer_errors": errors,
        "voice_code_wer_ref_words": ref_words,
        "voice_code_wer_hyp_words": ref_words - errors,
        "voice_code_wer_substitutions": errors,
        "voice_code_wer_deletions": 0,
        "voice_code_wer_insertions": 0,
        "voice_code_entity_verifier_status": "complete",
        "voice_code_verifier_id": "openai_gpt_5_5_v1",
        "voice_code_verifier_config_digest": "a" * 64,
        "voice_code_correct_entities": correct,
        "voice_code_total_entities": total,
        "voice_code_task_success": success,
        "voice_code_correct_by_type": {"phone_number": correct},
        "voice_code_total_by_type": {"phone_number": total},
    }


def test_metrics_compute_micro_wer_ctem_tsr_and_per_type():
    metrics = VoiceCodeBenchMetrics()
    metrics.update([_prediction("a", errors=1, ref_words=10, correct=2, total=2, success=True)])
    metrics.update([_prediction("b", errors=2, ref_words=20, correct=1, total=2, success=False)])
    result = metrics.get_metrics()["pass@1"]
    assert result["wer"] == pytest.approx(10.0)
    assert result["ctem"] == pytest.approx(75.0)
    assert result["tsr"] == pytest.approx(50.0)
    assert result["ctem::phone_number"] == pytest.approx(75.0)
    assert result["entity_verified_entries"] == 2


def test_metrics_reject_partial_entity_verification():
    metrics = VoiceCodeBenchMetrics()
    metrics.update([_prediction("a", errors=0, ref_words=10, correct=2, total=2, success=True)])
    diagnostic = _prediction("b", errors=0, ref_words=10, correct=0, total=0, success=False)
    diagnostic["voice_code_entity_verifier_status"] = "off"
    metrics.update([diagnostic])
    with pytest.raises(ValueError, match="Partial VoiceCodeBench entity scoring"):
        metrics.get_metrics()
