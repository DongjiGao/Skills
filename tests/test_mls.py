# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import importlib
import json

import pytest

prepare = importlib.import_module("nemo_skills.dataset.mls.prepare")
from nemo_skills.pipeline.summarize_results import MLS_LANGUAGE_CODES, _set_mls_language_macro_wer


def test_official_eight_language_source_contract_is_fully_pinned():
    assert prepare.PAPER_REVISION == "arxiv:2012.03411v2"
    assert prepare.LANGUAGE_CODES == ("en", "de", "nl", "fr", "es", "it", "pt", "pl")
    assert prepare.LANGUAGE_CODES == MLS_LANGUAGE_CODES
    assert prepare.EXPECTED_LANGUAGE_ROWS == {
        "en": 3769,
        "de": 3394,
        "nl": 3075,
        "fr": 2426,
        "es": 2385,
        "it": 1262,
        "pt": 871,
        "pl": 520,
    }
    assert prepare.EXPECTED_TOTAL_ROWS == 17702
    assert len({spec.parquet_path for spec in prepare.LANGUAGE_SPECS}) == 8
    for spec in prepare.LANGUAGE_SPECS:
        assert len(spec.revision) == 40
        assert len(spec.parquet_sha256) == 64
        int(spec.revision, 16)
        int(spec.parquet_sha256, 16)
        assert spec.parquet_size > 0
        assert spec.expected_rows > 0


def test_source_verification_rejects_wrong_size_or_hash(tmp_path, monkeypatch):
    source = tmp_path / "test.parquet"
    source.write_bytes(b"pinned source")
    spec = prepare.LanguageSpec(
        "xx",
        "Fixture",
        "repo/id",
        "a" * 40,
        "test.parquet",
        source.stat().st_size,
        hashlib.sha256(source.read_bytes()).hexdigest(),
        1,
    )
    monkeypatch.setattr(prepare, "hf_hub_download", lambda **_kwargs: str(source))

    assert prepare._download_and_verify_source(spec, tmp_path) == source
    with pytest.raises(ValueError, match="size mismatch"):
        prepare._download_and_verify_source(
            prepare.LanguageSpec(**{**spec.__dict__, "parquet_size": spec.parquet_size + 1}), tmp_path
        )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        prepare._download_and_verify_source(
            prepare.LanguageSpec(**{**spec.__dict__, "parquet_sha256": "0" * 64}), tmp_path
        )


def test_manifest_entry_preserves_language_scoring_and_source_provenance():
    spec = prepare.LANGUAGE_SPECS[3]
    entry = prepare.format_entry(
        {"transcript": "déjà vu"},
        spec,
        sample_id="123_456_000001",
        audio_path="/data/mls/audio/fr/123_456_000001.flac",
        duration=12.5,
    )

    assert entry["task_type"] == "Multilingual-ASR"
    assert entry["subset_for_metrics"] == "fr"
    assert entry["expected_answer"] == "déjà vu"
    assert entry["extra_fields"] == {
        "src_lang": "fr",
        "remove_diacritics": False,
        "normalize_compound": False,
        "merge_compounds": False,
        "use_cer": False,
    }
    assert entry["messages"][1]["audio"] == {
        "path": "/data/mls/audio/fr/123_456_000001.flac",
        "duration": 12.5,
    }
    provenance = entry["source_provenance"]
    assert provenance["dataset_revision"] == spec.revision
    assert provenance["source_file_sha256"] == spec.parquet_sha256
    assert provenance["expected_language_rows"] == spec.expected_rows
    assert provenance["paper_revision"] == "arxiv:2012.03411v2"


def test_prepare_publishes_complete_atomic_manifest_and_receipt(tmp_path, monkeypatch):
    fixture_specs = tuple(
        prepare.LanguageSpec(
            code,
            code,
            "repo/id",
            str(index) * 40,
            f"{code}/test.parquet",
            1,
            str(index) * 64,
            1,
        )
        for index, code in enumerate(("a", "b"), start=1)
    )
    monkeypatch.setattr(prepare, "LANGUAGE_SPECS", fixture_specs)
    monkeypatch.setattr(prepare, "LANGUAGE_CODES", ("a", "b"))
    monkeypatch.setattr(prepare, "EXPECTED_LANGUAGE_ROWS", {"a": 1, "b": 1})
    monkeypatch.setattr(prepare, "EXPECTED_TOTAL_ROWS", 2)

    def fake_prepare_language(spec, **_kwargs):
        return [
            {
                "subset_for_metrics": spec.code,
                "id": f"{spec.code}-1",
                "expected_answer": "text",
            }
        ]

    monkeypatch.setattr(prepare, "_prepare_language", fake_prepare_language)
    output_path = prepare.prepare_mls(tmp_path, no_audio=True)

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    receipt = json.loads((tmp_path / prepare.PROVENANCE_FILENAME).read_text(encoding="utf-8"))
    assert [row["subset_for_metrics"] for row in rows] == ["a", "b"]
    assert receipt["languages"] == ["a", "b"]
    assert receipt["language_rows"] == {"a": 1, "b": 1}
    assert receipt["total_rows"] == 2
    assert receipt["manifest_sha256"] == hashlib.sha256(output_path.read_bytes()).hexdigest()
    assert receipt["audio_mode"] == "metadata-only"


def _language_metrics(wers):
    metrics = {
        "_all_": {
            "pass@1": {
                "wer": 999.0,
                "wer_macro": 888.0,
                "num_entries": 17702,
            }
        }
    }
    for language, wer in zip(MLS_LANGUAGE_CODES, wers, strict=True):
        metrics[language] = {"pass@1": {"wer": wer, "num_entries": 1}}
    return metrics


def test_mls_summary_uses_explicit_language_macro_not_corpus_micro():
    metrics = _language_metrics([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    _set_mls_language_macro_wer(metrics)

    overall = metrics["_all_"]["pass@1"]
    assert overall["wer"] == 4.5
    assert overall["wer_macro"] == 4.5
    assert overall["wer_language_macro"] == 4.5
    assert overall["wer_corpus"] == 999.0
    assert overall["wer_utterance_macro"] == 888.0
    assert overall["languages_scored"] == 8


def test_mls_summary_fails_closed_when_a_language_is_missing():
    metrics = _language_metrics([1.0] * 8)
    del metrics["pl"]

    with pytest.raises(ValueError, match="complete official eight-language"):
        _set_mls_language_macro_wer(metrics)


def test_mls_multilingual_normalization_preserves_diacritics_and_word_boundaries():
    pytest.importorskip("jiwer")
    pytest.importorskip("kaldialign")
    pytest.importorskip("num2words")
    pytest.importorskip("whisper_normalizer")
    from nemo_skills.evaluation.evaluator.audio import AudioEvaluatorConfig, evaluate_sample

    config = AudioEvaluatorConfig(normalization_mode="multilingual")
    sample = {
        "task_type": "Multilingual-ASR",
        "expected_answer": "été white paper",
        "generation": "ete whitepaper",
        "extra_fields": {
            "src_lang": "fr",
            "remove_diacritics": False,
            "normalize_compound": False,
            "merge_compounds": False,
        },
    }

    result = evaluate_sample(sample, config)

    assert result["text"] == "été white paper"
    assert result["pred_text"] == "ete whitepaper"
    assert result["wer_errors"] == 3
    assert result["wer_ref_words"] == 3
