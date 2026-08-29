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

pytest.importorskip("jieba")

from nemo_skills.dataset.ascend.prepare import HF_REVISION, OFFICIAL_SPLIT, PROMPT, _format_entry
from nemo_skills.evaluation.evaluator import is_evaluator_registered
from nemo_skills.evaluation.evaluator.ascend import (
    OFFICIAL_SCORER_REVISION,
    evaluate_ascend_sample,
    mer_tokens,
    normalize_ascend,
)
from nemo_skills.evaluation.metrics.ascend_metrics import ASCENDMetrics
from nemo_skills.evaluation.metrics.map_metrics import get_metrics


def test_ascend_official_normalization_and_mixed_units():
    normalized = normalize_ascend("HELLO, 世界！")
    assert normalized == "hello世界 "
    assert mer_tokens(normalized) == ["hello", "世", "界"]


def test_ascend_evaluator_and_corpus_metrics_report_mer_then_cer():
    sample = {"expected_answer": "hello 世界", "generation": "hello 世"}
    evaluated = evaluate_ascend_sample(sample)
    assert evaluated["mer_errors"] == 1
    assert evaluated["mer_ref_units"] == 3
    assert evaluated["cer_errors"] == 1
    assert evaluated["cer_ref_chars"] == 6

    metrics = ASCENDMetrics(max_k=1)
    metrics.update([{**sample, **evaluated}])
    output = metrics.get_metrics()["pass@1"]
    assert output["mer"] == 33.33
    assert output["cer"] == 16.67


def test_ascend_manifest_is_plain_transcript_and_pinned():
    record = _format_entry({"id": "sample/1", "transcription": "你好 hello", "duration": 1.25, "language": "mixed"}, 0)
    assert record["messages"][0]["content"] == PROMPT
    assert len(record["messages"]) == 1
    assert record["extra_fields"]["source_revision"] == HF_REVISION
    assert record["extra_fields"]["source_split"] == OFFICIAL_SPLIT


def test_ascend_registration():
    assert is_evaluator_registered("ascend")
    assert OFFICIAL_SCORER_REVISION == "8c50c2e2c65f8555eb1d7655310dd585dc1f6710"
    assert isinstance(get_metrics("ascend", max_k=1), ASCENDMetrics)
