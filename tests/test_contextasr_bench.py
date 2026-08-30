# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
import json
from importlib import import_module

import pytest

prepare = import_module("nemo_skills.dataset.contextasr-bench.prepare")


def _sample(uniq_id: str = "sample") -> dict:
    return {
        "uniq_id": uniq_id,
        "language": "English",
        "duration": 1.5,
        "audio": f"audio/ContextASR-Speech/English/{uniq_id}.wav",
        "text": "hello world",
        "domain_label": "test domain",
        "entity_list": ["hello"],
    }


def test_contextasr_source_contract_is_immutable():
    assert prepare.HF_REVISION == "7906fd98d1d093bf244df80e726e07fc8afed6a6"
    assert prepare.EXPECTED_ROWS == 15326
    assert prepare.JSONL_SHA256 == "b6bcb3de39aebe36328ab03d7d4d865899712c2dc375b6cd2b9ce156b6a952e4"


def test_contextasr_metadata_validator_rejects_hash_and_row_drift(tmp_path, monkeypatch):
    path = tmp_path / prepare.JSONL_FILENAME
    rows = [_sample("one"), _sample("two")]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(prepare, "JSONL_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(prepare, "EXPECTED_ROWS", 2)
    assert prepare._validate_metadata(path) == rows

    monkeypatch.setattr(prepare, "EXPECTED_ROWS", 3)
    with pytest.raises(ValueError, match="row-count drift"):
        prepare._validate_metadata(path)

    monkeypatch.setattr(prepare, "EXPECTED_ROWS", 2)
    monkeypatch.setattr(prepare, "JSONL_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="metadata hash drift"):
        prepare._validate_metadata(path)


def test_contextasr_record_carries_immutable_source_provenance():
    record = prepare.format_entry(_sample(), "fine", "/data/contextasr-bench")
    provenance = record["source_provenance"]
    assert provenance["dataset"] == prepare.HF_REPO_ID
    assert provenance["dataset_revision"] == prepare.HF_REVISION
    assert provenance["expected_rows"] == 15326
