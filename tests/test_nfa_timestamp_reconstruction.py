# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused protocol tests for the NFA timestamp public reconstruction."""

import hashlib
import importlib
import json
from pathlib import Path

import pytest

from nemo_skills.evaluation.evaluator import _resolve_eval_type
from nemo_skills.evaluation.metrics.map_metrics import get_metrics

benchmark = importlib.import_module("nemo_skills.dataset.nfa-timestamp-reconstruction")
prepare = importlib.import_module("nemo_skills.dataset.nfa-timestamp-reconstruction.prepare")
scorer = importlib.import_module("nemo_skills.dataset.nfa-timestamp-reconstruction.scorer")


def _point(reference_words, generation, source="librispeech-test-other"):
    return {
        "protocol_profile": scorer.PROTOCOL_VERSION,
        "subset_for_metrics": source,
        "reference_words": reference_words,
        "generation": generation,
    }


def _words(*items):
    return [{"word": word, "start_tick": start_tick, "end_tick": end_tick} for word, start_tick, end_tick in items]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _make_artifact(tmp_path: Path):
    artifact_root = tmp_path / "artifact"
    audio_root = artifact_root / "audio"
    audio_root.mkdir(parents=True)
    source_root = artifact_root / "sources"
    source_root.mkdir()
    manifest_rows = []
    source_contracts = {}
    audio_paths = []
    for index, (source_key, source_spec) in enumerate(prepare.SOURCE_SPECS.items()):
        sample_id = f"sample-{index}"
        audio_payload = f"pinned-audio-{source_key}".encode()
        audio_path = audio_root / f"{sample_id}.flac"
        audio_path.write_bytes(audio_payload)
        audio_paths.append(audio_path)
        manifest_rows.append(
            {
                "id": sample_id,
                "source": source_key,
                "audio_filepath": f"audio/{sample_id}.flac",
                "audio_sha256": _sha256_bytes(audio_payload),
                "audio_bytes": len(audio_payload),
                "duration": 0.24,
                "words": [{"word": f"word{index}", "start": 0.04, "end": 0.12}],
            }
        )
        source_manifest_payload = f"source-manifest-{source_key}".encode()
        source_manifest_path = source_root / f"{source_key}.txt"
        source_manifest_path.write_bytes(source_manifest_payload)
        source_contracts[source_key] = {
            **source_spec,
            "rows": 1,
            "source_manifest": f"sources/{source_key}.txt",
            "source_manifest_sha256": _sha256_bytes(source_manifest_payload),
        }

    manifest_path = artifact_root / "aligned.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    alignment_config = artifact_root / "alignment-config.yaml"
    alignment_config.write_bytes(b"test alignment config")
    contract = {
        "schema_version": prepare.ARTIFACT_SCHEMA,
        "protocol_version": prepare.PROTOCOL_VERSION,
        "manifest": "aligned.jsonl",
        "manifest_sha256": _sha256_bytes(manifest_path.read_bytes()),
        "manifest_rows": len(manifest_rows),
        "sources": source_contracts,
        "alignment": {
            "tool": "test-aligner",
            "revision": "test-aligner-immutable-revision",
            "config": "alignment-config.yaml",
            "config_sha256": _sha256_bytes(b"test alignment config"),
        },
    }
    contract_path = artifact_root / "contract.json"
    _write_json(contract_path, contract)
    return contract_path, _sha256_bytes(contract_path.read_bytes()), manifest_path, audio_paths


def test_benchmark_uses_dataset_local_evaluator_and_metrics_without_registry_edits():
    assert benchmark.REQUIRES_DATA_DIR is True
    assert benchmark.NUM_SAMPLES == 1
    assert "reconstruction" in benchmark.METRICS_TYPE
    metrics = get_metrics(benchmark.METRICS_TYPE)
    assert isinstance(metrics, scorer.NFATimestampReconstructionMetrics)
    eval_type = benchmark.EVAL_ARGS.removeprefix("++eval_type=")
    evaluator_type, is_class = _resolve_eval_type(eval_type)
    assert is_class is True
    assert evaluator_type is scorer.NFATimestampReconstructionEvaluator


def test_timestamp_true_positive_uses_strict_three_tick_boundary():
    evaluated = scorer.evaluate_nfa_timestamp_sample(
        _point(
            _words(("alpha", 0, 5), ("beta", 10, 15)),
            "<|2|>alpha<|7|> <|13|>beta<|18|>",
        )
    )
    assert evaluated["wer"] == 0
    assert evaluated["timestamp_true_positives"] == 1
    assert evaluated["timestamp_predictions"] == 2
    assert evaluated["timestamp_references"] == 2
    assert evaluated["timestamped_word_matches"] == 2
    assert evaluated["start_deviation_ticks_sum"] == 5
    assert evaluated["end_deviation_ticks_sum"] == 5
    assert evaluated["unparsed_timestamp_tokens"] == 0


def test_shared_boundary_tokens_and_stripped_wer_alignment():
    shared = scorer.evaluate_nfa_timestamp_sample(
        _point(
            _words(("one", 0, 5), ("two", 5, 10)),
            "<|0|>one<|5|>two<|10|>",
        )
    )
    assert shared["timestamp_true_positives"] == 2
    assert shared["timestamped_hyp_words"] == 2
    assert shared["unparsed_timestamp_tokens"] == 0

    insertion = scorer.evaluate_nfa_timestamp_sample(
        _point(
            _words(("one", 0, 1), ("two", 2, 3)),
            "<|0|>one<|1|> <|1|>extra<|2|> <|2|>two<|3|>",
        )
    )
    assert insertion["pred_text"] == "one extra two"
    assert insertion["wer_errors"] == 1
    assert insertion["wer_insertions"] == 1
    assert insertion["reference_words_count"] == 2
    assert insertion["timestamp_true_positives"] == 2
    assert insertion["timestamp_predictions"] == 3


def test_plain_words_have_zero_timestamp_credit_but_valid_stripped_wer():
    evaluated = scorer.evaluate_nfa_timestamp_sample(_point(_words(("hello", 0, 4), ("world", 5, 9)), "hello world"))
    assert evaluated["wer"] == 0
    assert evaluated["timestamp_true_positives"] == 0
    assert evaluated["timestamp_predictions"] == 2
    assert evaluated["untimestamped_hyp_words"] == 2
    assert evaluated["timestamped_word_matches"] == 0


def test_metrics_are_corpus_micro_and_report_source_breakdowns():
    predictions = [
        scorer.evaluate_nfa_timestamp_sample(
            _point(
                _words(("alpha", 0, 5), ("beta", 10, 15)),
                "<|2|>alpha<|7|> <|13|>beta<|18|>",
                "librispeech-test-other",
            )
        ),
        scorer.evaluate_nfa_timestamp_sample(
            _point(
                _words(("one", 0, 1), ("two", 2, 3)),
                "<|0|>one<|1|> <|1|>extra<|2|> <|2|>two<|3|>",
                "common-voice-16.1-en",
            )
        ),
    ]
    metrics = scorer.NFATimestampReconstructionMetrics(compute_no_answer=False)
    for prediction in predictions:
        metrics.update([prediction])
    result = metrics.get_metrics()["pass@1"]
    assert result["wer"] == 25.0
    assert result["timestamp_precision"] == 60.0
    assert result["timestamp_recall"] == 75.0
    assert result["start_deviation_ms"] == 100.0
    assert result["end_deviation_ms"] == 100.0
    assert result["timestamp_precision::librispeech-test-other"] == 50.0
    assert result["timestamp_recall::common-voice-16.1-en"] == 100.0
    assert result["comparability"] == "public-reconstruction-proxy"


def test_quantization_is_80ms_decimal_round_half_up():
    assert prepare.quantize_seconds(0.0) == 0
    assert prepare.quantize_seconds(0.039) == 0
    assert prepare.quantize_seconds(0.04) == 1
    assert prepare.quantize_seconds(0.12) == 2
    assert prepare.quantize_seconds(0.24) == 3


def test_prepare_requires_and_materializes_complete_pinned_proxy(monkeypatch, tmp_path):
    monkeypatch.delenv("NEMO_SKILLS_AUDIO_ROOT", raising=False)
    contract_path, contract_sha256, _manifest_path, _audio_paths = _make_artifact(tmp_path)
    output_dir = tmp_path / "prepared"
    output_rows = prepare.prepare_artifact(
        artifact_contract=contract_path,
        artifact_contract_sha256=contract_sha256,
        output_dir=output_dir,
    )
    assert len(output_rows) == 5
    assert {row["subset_for_metrics"] for row in output_rows} == set(prepare.SOURCE_SPECS)
    assert all(row["comparability"] == "public-reconstruction-proxy" for row in output_rows)
    assert all("not paper-equivalent" in row["comparability_limit"] for row in output_rows)
    assert output_rows[0]["expected_answer"] == "<|1|>word0<|2|>"
    assert output_rows[0]["reference_words"] == [{"word": "word0", "start_tick": 1, "end_tick": 2}]
    assert output_rows[0]["audio_filepath"].startswith("/data/nfa-timestamp-reconstruction/audio/")
    assert output_rows[0]["source_provenance"]["quantization"] == "80ms-round-half-up"
    assert output_rows[0]["source_provenance"]["true_positive_tolerance"].startswith("start_error<240ms")
    assert len((output_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()) == 5
    assert len(list((output_dir / "audio").glob("**/*.flac"))) == 5
    provenance = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["artifact_contract_sha256"] == contract_sha256
    assert provenance["source_rows"] == {source: 1 for source in prepare.SOURCE_SPECS}
    assert provenance["true_positive_tolerance_operator"] == "<"
    assert provenance["comparability"] == "public-reconstruction-proxy"


def test_prepare_fails_closed_on_unpinned_or_drifted_artifacts(tmp_path):
    contract_path, contract_sha256, manifest_path, audio_paths = _make_artifact(tmp_path)
    with pytest.raises(ValueError, match="Artifact contract hash mismatch"):
        prepare.prepare_artifact(
            artifact_contract=contract_path,
            artifact_contract_sha256="0" * 64,
            output_dir=tmp_path / "bad-contract",
        )

    manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Artifact manifest hash mismatch"):
        prepare.prepare_artifact(
            artifact_contract=contract_path,
            artifact_contract_sha256=contract_sha256,
            output_dir=tmp_path / "bad-manifest",
        )

    contract_path, contract_sha256, _manifest_path, _audio_paths = _make_artifact(tmp_path / "source-drift")
    source_manifest = contract_path.parent / "sources" / "librispeech-test-other.txt"
    source_manifest.write_bytes(b"drifted source manifest")
    with pytest.raises(ValueError, match="Source manifest hash mismatch"):
        prepare.prepare_artifact(
            artifact_contract=contract_path,
            artifact_contract_sha256=contract_sha256,
            output_dir=tmp_path / "bad-source-manifest",
        )

    contract_path, contract_sha256, _manifest_path, _audio_paths = _make_artifact(tmp_path / "config-drift")
    alignment_config = contract_path.parent / "alignment-config.yaml"
    alignment_config.write_bytes(b"drifted alignment config")
    with pytest.raises(ValueError, match="Alignment config hash mismatch"):
        prepare.prepare_artifact(
            artifact_contract=contract_path,
            artifact_contract_sha256=contract_sha256,
            output_dir=tmp_path / "bad-alignment-config",
        )

    contract_path, contract_sha256, _manifest_path, audio_paths = _make_artifact(tmp_path / "second")
    audio_paths[0].write_bytes(b"drifted-audio")
    with pytest.raises(ValueError, match="Audio (size|SHA-256) mismatch"):
        prepare.prepare_artifact(
            artifact_contract=contract_path,
            artifact_contract_sha256=contract_sha256,
            output_dir=tmp_path / "bad-audio",
        )
