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

prepare = importlib.import_module("nemo_skills.dataset.librispeech-pc.prepare")
process_split = prepare.process_split


@pytest.mark.parametrize("split", ["test-clean", "test-other"])
def test_librispeech_pc_split_benchmarks_are_independently_runnable(split):
    module = importlib.import_module(f"nemo_skills.dataset.librispeech-pc.{split}")

    assert module.REQUIRES_DATA_DIR is True
    assert module.METRICS_TYPE == "audio"
    assert module.EVAL_SPLIT == "test"
    assert "++eval_type=audio" in module.EVAL_ARGS
    assert "++enable_audio=true" in module.GENERATION_ARGS


@pytest.mark.parametrize("split", ["test-clean", "test-other"])
def test_librispeech_pc_prepare_writes_leaf_and_legacy_manifests(tmp_path, monkeypatch, split):
    relative_audio = f"{split}/1/2/1-2-3.flac"
    source = {
        "audio_filepath": relative_audio,
        "duration": 1.25,
        "text": "Hello, World!",
        "text_raw": "HELLO WORLD",
    }
    manifest = tmp_path / f"{split}.json"
    manifest.write_text(json.dumps(source) + "\n", encoding="utf-8")
    monkeypatch.setitem(
        prepare.MANIFEST_CONTRACTS,
        split,
        {
            "rows": 1,
            "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        },
    )
    audio = tmp_path / "LibriSpeech" / relative_audio
    audio.parent.mkdir(parents=True)
    audio.write_bytes(prepare.FLAC_MAGIC + b"test fixture")

    assert process_split(split, tmp_path, tmp_path, with_audio=False) == 1

    leaf = tmp_path / split / "test.jsonl"
    legacy = tmp_path / f"{split}.jsonl"
    assert leaf.read_text(encoding="utf-8") == legacy.read_text(encoding="utf-8")
    row = json.loads(leaf.read_text(encoding="utf-8"))
    assert row["id"] == "1-2-3"
    assert row["expected_answer"] == "Hello, World!"
    assert row["duration"] == 1.25
    assert row["split"] == split
    assert row["messages"][1]["audio"]["duration"] == 1.25
    assert row["messages"][1]["content"] == ("Transcribe the audio with proper punctuation and capitalization.")
