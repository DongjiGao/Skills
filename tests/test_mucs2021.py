# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Network-free tests for pinned MUCS 2021 preparation and scoring."""

from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from nemo_skills.dataset.mucs2021 import BENCHMARKS, EVAL_ARGS, METRICS_TYPE
from nemo_skills.dataset.mucs2021.mucs2021_score import compute_score
from nemo_skills.dataset.mucs2021.prepare import (
    ARCHIVE_MEMBER_COUNT,
    LANGUAGE_CONTRACTS,
    SOURCE_BYTES,
    SOURCE_SHA256,
    SOURCE_URL,
    _parse_kaldi_text,
    _slice_recording,
)
from nemo_skills.dataset.mucs2021.scorer import (
    ASSET_CONTRACTS,
    ASSETS_DIR,
    MUCS2021Evaluator,
    MUCS2021Metrics,
    apply_mapping,
    evaluate_mucs2021_sample,
    load_official_mapping,
    whitespace_tokens,
)
from nemo_skills.evaluation.metrics.map_metrics import get_metrics


def _sample(reference: str, hypothesis: str, language: str = "hindi-english") -> dict:
    return {
        "expected_answer": reference,
        "generation": hypothesis,
        "extra_fields": {"mucs_language": language},
    }


def test_public_source_and_exact_release_contracts():
    assert SOURCE_URL == "https://www.openslr.org/resources/104/subtask2_blindtest_wReadme.tar.gz"
    assert SOURCE_SHA256 == "b134ba007e6a310a2ca2a89619257ea6e7de744e135043c8c517cc05a404889a"
    assert SOURCE_BYTES == 1_008_263_116
    assert ARCHIVE_MEMBER_COUNT == 88
    assert LANGUAGE_CONTRACTS["hindi-english"].rows == 4034
    assert LANGUAGE_CONTRACTS["hindi-english"].recordings == 35
    assert LANGUAGE_CONTRACTS["hindi-english"].empty_references == 8
    assert LANGUAGE_CONTRACTS["bengali-english"].rows == 3130
    assert LANGUAGE_CONTRACTS["bengali-english"].recordings == 32
    assert LANGUAGE_CONTRACTS["bengali-english"].empty_references == 4


def test_vendored_official_assets_match_pinned_hashes_and_rows():
    import hashlib

    for name, (expected_sha, expected_rows) in ASSET_CONTRACTS.items():
        payload = (ASSETS_DIR / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected_sha
        assert len(payload.decode("utf-8").splitlines()) == expected_rows


def test_official_lowercase_whitespace_and_join_mapping_semantics():
    assert whitespace_tokens("  HELLO\tworld  ") == ["hello", "world"]
    assert apply_mapping(["a", "drop", "b"], {"drop": "", "a": "x y"}) == ["x", "y", "b"]


def test_normal_wer_applies_punctuation_mapping_to_hypothesis_only():
    evaluated = evaluate_mucs2021_sample(_sample("hello ,", "HELLO comma"))
    assert evaluated["pred_text"] == "hello ,"
    assert evaluated["wer_errors"] == 0
    assert evaluated["wer"] == 0.0


def test_hindi_transliterated_wer_starts_from_punctuation_mapped_hypothesis():
    evaluated = evaluate_mucs2021_sample(_sample("acceleration ,", "अक्सेलरेशन comma"))
    assert evaluated["pred_text"] == "अक्सेलरेशन ,"
    assert evaluated["transliterated_pred_text"] == "acceleration ,"
    assert evaluated["wer_errors"] == 1
    assert evaluated["twer_errors"] == 0


def test_bengali_transliterated_wer_uses_bengali_map():
    evaluated = evaluate_mucs2021_sample(_sample("abort", "আবর্ত", "bengali-english"))
    assert evaluated["wer_errors"] == 1
    assert evaluated["twer_errors"] == 0


def test_empty_reference_is_skipped_including_hypothesis_insertions():
    evaluated = evaluate_mucs2021_sample(_sample("", "these insertions do not count"))
    assert evaluated["skipped_empty_reference"] is True
    assert evaluated["ref_words"] == 0
    assert evaluated["wer_errors"] == 0
    assert evaluated["twer_errors"] == 0
    assert evaluated["wer"] is None
    assert evaluated["twer"] is None


def test_corpus_metrics_expose_exact_wer_twer_percentage_keys():
    metrics = MUCS2021Metrics(max_k=1)
    metrics.update([{**_sample("one two", "one wrong"), **evaluate_mucs2021_sample(_sample("one two", "one wrong"))}])
    empty = _sample("", "ignored insertion")
    metrics.update([{**empty, **evaluate_mucs2021_sample(empty)}])
    output = metrics.get_metrics()["pass@1"]
    assert output["wer"] == 50.0
    assert output["twer"] == 50.0
    assert output["ref_words"] == 2
    assert output["scored_entries"] == 1
    assert output["skipped_empty_references"] == 1
    assert output["num_entries"] == 2


def test_dynamic_evaluator_and_metrics_paths_need_no_central_registration():
    assert EVAL_ARGS.endswith("nemo_skills.dataset.mucs2021.scorer::MUCS2021Evaluator")
    assert METRICS_TYPE.endswith("nemo_skills.dataset.mucs2021.scorer::MUCS2021Metrics")
    assert isinstance(get_metrics(METRICS_TYPE, max_k=1), MUCS2021Metrics)
    assert MUCS2021Evaluator.__name__ == "MUCS2021Evaluator"
    for benchmark in BENCHMARKS:
        module = import_module(f"nemo_skills.dataset.{benchmark}")
        assert module.METRICS_TYPE == METRICS_TYPE
        assert module.EVAL_ARGS == EVAL_ARGS


def test_group_reports_languages_separately_without_overall():
    combined = {
        "mucs2021.hindi-english": {"pass@1": {"wer": 10.0, "twer": 8.0, "num_entries": 4034}},
        "mucs2021.bengali-english": {"pass@1": {"wer": 20.0, "twer": 15.0, "num_entries": 3130}},
    }
    result = compute_score(combined)
    assert result == {
        "pass@1": {
            "hindi-english": combined["mucs2021.hindi-english"]["pass@1"],
            "bengali-english": combined["mucs2021.bengali-english"]["pass@1"],
        }
    }
    assert "overall" not in result["pass@1"]


def test_kaldi_text_parser_preserves_reference_and_empty_row(tmp_path: Path):
    text = tmp_path / "text"
    text.write_text("utt1 Exact देवनागरी + English\nutt2 \n", encoding="utf-8")
    assert _parse_kaldi_text(text) == {"utt1": "Exact देवनागरी + English", "utt2": ""}


def test_deterministic_segment_slice_uses_exact_source_frames(tmp_path: Path):
    source = tmp_path / "source.wav"
    output = tmp_path / "segment.wav"
    samples = np.arange(32000, dtype=np.int16)
    sf.write(source, samples, 16000, format="WAV", subtype="PCM_16")
    duration = _slice_recording(source, output, "0.5", "1.0")
    sliced, rate = sf.read(output, dtype="int16")
    assert rate == 16000
    assert duration == 0.5
    np.testing.assert_array_equal(sliced, samples[8000:16000])


def test_unknown_language_fails_closed():
    with pytest.raises(ValueError, match="mucs_language"):
        evaluate_mucs2021_sample(_sample("hello", "hello", "unknown"))


def test_official_maps_load_and_have_exact_cardinality():
    load_official_mapping.cache_clear()
    assert len(load_official_mapping("punct.mapping.txt")) == 76
    assert len(load_official_mapping("hin2engwords.txt")) == 1315
    assert len(load_official_mapping("ben2engwords.txt")) == 1177
