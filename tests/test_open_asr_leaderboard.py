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

import json
from importlib import import_module

import numpy as np
import pytest

pytest.importorskip("datasets")
pytest.importorskip("kaldialign")
pytest.importorskip("soundfile")
pytest.importorskip("whisper_normalizer")

import soundfile as sf

from nemo_skills.evaluation.evaluator.audio import evaluate_asr
from nemo_skills.evaluation.metrics.map_metrics import get_metrics
from nemo_skills.evaluation.metrics.open_asr_metrics import OpenASRLeaderboardMetrics

prepare = import_module("nemo_skills.dataset.asr-leaderboard.prepare")


class FakeDataset(list):
    @property
    def column_names(self):
        return list(self[0]) if self else []


def test_public_dataset_membership_matches_current_leaderboard():
    assert set(prepare.DATASET_CONFIGS) == {
        "ami",
        "earnings22_aa",
        "gigaspeech",
        "librispeech_clean",
        "librispeech_other",
        "spgispeech",
        "voxpopuli_aa",
    }
    spec = prepare.DATASET_CONFIGS["earnings22_aa"]
    assert spec.repo == "ArtificialAnalysis/Earnings22-Cleaned-AA-chunked"
    assert spec.parent_repo == "ArtificialAnalysis/Earnings22-Cleaned-AA"
    assert spec.config is None
    assert spec.split == "test"


@pytest.mark.parametrize("text", ["", " [noise] ", "ignore time segment in scoring"])
def test_reference_filter_matches_open_asr_leaderboard(text, tmp_path):
    entry = {"id": "sample", "text": text, "audio_length_s": 1.0}
    spec = prepare.DatasetSpec("dummy", "test")
    assert prepare.format_entry(entry, "dummy", tmp_path, spec, with_audio=False) is None


def test_prepare_aa_earnings22_attaches_parent_reference_and_chunk_metadata(tmp_path, monkeypatch):
    chunks = FakeDataset(
        [
            {
                "id": "chunk-0",
                "parent_id": "call-1",
                "chunk_index": 0,
                "duration": 0.01,
                "file_name": "call-1_chunk_0.wav",
            },
            {
                "id": "chunk-1",
                "parent_id": "call-1",
                "chunk_index": 1,
                "duration": 0.01,
                "file_name": "call-1_chunk_1.wav",
            },
        ]
    )
    parents = FakeDataset([{"id": "call-1", "transcript": "Hello world"}])

    def fake_load_dataset(repo, *args, **kwargs):
        if repo == "ArtificialAnalysis/Earnings22-Cleaned-AA-chunked":
            return chunks
        if repo == "ArtificialAnalysis/Earnings22-Cleaned-AA":
            return parents
        raise AssertionError(repo)

    snapshot = tmp_path / "snapshot"
    source_audio = snapshot / "audio"
    source_audio.mkdir(parents=True)
    for chunk in chunks:
        sf.write(source_audio / chunk["file_name"], np.zeros(160, dtype=np.float32), 16000)

    monkeypatch.setattr(prepare, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(prepare, "snapshot_download", lambda **kwargs: str(snapshot))

    assert prepare.prepare_dataset("earnings22_aa", tmp_path, with_audio=True) == 2

    rows = [json.loads(line) for line in (tmp_path / "earnings22_aa.jsonl").read_text().splitlines()]
    assert [row["parent_id"] for row in rows] == ["call-1", "call-1"]
    assert [row["chunk_index"] for row in rows] == [0, 1]
    assert [row["expected_answer"] for row in rows] == ["Hello world", "Hello world"]
    assert all(row["subset_for_metrics"] == "earnings22_aa" for row in rows)
    assert all(row["messages"][1]["audio"]["duration"] == pytest.approx(0.01) for row in rows)
    assert all(
        row["messages"][1]["audio"]["path"].startswith("/data/asr-leaderboard/data/earnings22_aa/") for row in rows
    )
    assert (tmp_path / "data" / "earnings22_aa" / "chunk-0.flac").is_file()
    assert (tmp_path / "data" / "earnings22_aa" / "chunk-1.flac").is_file()


def _chunk(chunk_index, generation, reference="hello world again"):
    return {
        "id": f"chunk-{chunk_index}",
        "parent_id": "call-1",
        "chunk_index": chunk_index,
        "subset_for_metrics": "earnings22_aa",
        "task_type": "ASR",
        "expected_answer": reference,
        "generation": generation,
        "num_generated_tokens": len(generation.split()),
    }


def test_chunked_predictions_are_ordered_and_scored_once_per_parent():
    metrics = OpenASRLeaderboardMetrics(max_k=1)
    metrics.update([_chunk(1, "world again")])
    metrics.update([_chunk(0, "hello")])

    result = metrics.get_metrics()["pass@1"]
    assert result["wer"] == 0.0
    assert result["wer_macro"] == 0.0
    assert result["ref_words"] == 3
    assert result["num_entries"] == 1
    assert result["avg_tokens"] == 3


def test_chunked_and_regular_rows_share_corpus_wer_accumulators():
    metrics = OpenASRLeaderboardMetrics(max_k=1)
    regular = {
        "id": "regular",
        "subset_for_metrics": "spgispeech",
        "task_type": "ASR",
        "expected_answer": "alpha beta",
        "generation": "alpha gamma",
        "num_generated_tokens": 2,
    }
    regular.update(evaluate_asr("alpha beta", "alpha gamma", normalization_mode="hf_leaderboard"))

    metrics.update([regular])
    metrics.update([_chunk(0, "hello")])
    metrics.update([_chunk(1, "world again")])

    result = metrics.get_metrics()["pass@1"]
    assert result["wer"] == 20.0
    assert result["ref_words"] == 5
    assert result["substitutions"] == 1
    assert result["num_entries"] == 2


def test_benchmark_uses_session_aware_metrics():
    benchmark = import_module("nemo_skills.dataset.asr-leaderboard")
    assert benchmark.METRICS_TYPE == "open_asr_leaderboard"
    assert isinstance(get_metrics(benchmark.METRICS_TYPE), OpenASRLeaderboardMetrics)
