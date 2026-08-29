# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from nemo_skills.evaluation.evaluator.voice_code_bench import (
    OFFICIAL_REVISION,
    compute_format_invariant_wer,
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
    message = entry["messages"][0]
    assert message["content"] == "Transcribe the following:"
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
