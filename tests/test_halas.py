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

import pytest

pytest.importorskip("jiwer")
pytest.importorskip("whisper_normalizer")

from nemo_skills.dataset.halas.prepare import (
    EARNINGS22_REPO_ID,
    EARNINGS22_REVISION,
    HF_REPO_ID,
    HF_REVISION,
    _default_audio_prefix,
    _default_data_dir,
    _format_entry,
    _model_label_counts,
)
from nemo_skills.evaluation.evaluator.halas import evaluate_halas_sample
from nemo_skills.evaluation.metrics.halas_metrics import HALASMetrics


def test_halas_evaluator_reports_insertion_proxy_counts():
    sample = {
        "expected_answer": "hello world",
        "generation": "hello brave world again",
        "audio_duration": 2.0,
    }
    evaluated = evaluate_halas_sample(sample)
    assert evaluated["wer_insertions"] == 2
    assert evaluated["has_insertion"] is True
    assert evaluated["hallucination_proxy"] is True

    metrics = HALASMetrics(max_k=1)
    metrics.update([{**sample, **evaluated}])
    output = metrics.get_metrics()["pass@1"]
    assert output["insertions"] == 2
    assert output["insertion_sample_rate"] == 100.0
    assert output["hallucination_proxy_rate"] == 100.0


def test_halas_prepare_counts_no_hallucination_label_separately():
    counts = _model_label_counts(
        {
            "model_a_label": "No hallucination",
            "model_b_label": "Hallucination",
            "model_c_label": "Looping",
        }
    )
    assert counts == {"hallucination_or_looping": 2, "no_hallucination": 1}


def test_halas_prepare_defaults_honor_nemo_skills_data_dir(monkeypatch):
    monkeypatch.setenv("NEMO_SKILLS_DATA_DIR", "/cluster/skills_data")

    data_dir = _default_data_dir()

    assert str(data_dir) == "/cluster/skills_data/halas"
    assert _default_audio_prefix(None, data_dir) == "/data/halas"


def test_halas_source_revisions_are_pinned_in_every_formatted_row():
    formatted = _format_entry(
        {
            "audio_id": "sample.wav",
            "audio_duration": 1.25,
            "corrected_reference_text": "hello",
            "e22_reference_text": "Hello",
            "split": "test",
        },
        "/data/halas",
    )

    assert HF_REPO_ID == "MatBar99/HALAS"
    assert EARNINGS22_REPO_ID == "distil-whisper/earnings22"
    assert formatted["extra_fields"] == {
        "source_repo": HF_REPO_ID,
        "source_revision": HF_REVISION,
        "audio_source_repo": EARNINGS22_REPO_ID,
        "audio_source_revision": EARNINGS22_REVISION,
    }
