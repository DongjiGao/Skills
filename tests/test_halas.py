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

import sys
import types
import wave

import numpy as np
import pytest

pytest.importorskip("jiwer")
pytest.importorskip("whisper_normalizer")

from nemo_skills.dataset.halas.prepare import (
    EARNINGS22_REPO_ID,
    EARNINGS22_REVISION,
    EXPECTED_DUPLICATE_AUDIO_COUNTS,
    EXPECTED_ROWS,
    EXPECTED_UNIQUE_AUDIO,
    HALAS_SPLIT,
    HF_REPO_ID,
    HF_REVISION,
    _decode_earnings22_audio,
    _default_audio_prefix,
    _default_data_dir,
    _extract_audio_for_rows,
    _format_entry,
    _model_label_counts,
    _validate_audio_closure,
    _validate_halas_rows,
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
            "_halas_source_row_index": 0,
        },
        "/data/halas",
    )

    assert HF_REPO_ID == "MatBar99/HALAS"
    assert HALAS_SPLIT == "test"
    assert EXPECTED_ROWS == 745
    assert EXPECTED_UNIQUE_AUDIO == 744
    assert EXPECTED_DUPLICATE_AUDIO_COUNTS == {"23_4484146.wav": 2}
    assert EARNINGS22_REPO_ID == "distil-whisper/earnings22"
    assert formatted["id"] == "halas-test-0000-sample.wav"
    assert formatted["extra_fields"] == {
        "source_repo": HF_REPO_ID,
        "source_revision": HF_REVISION,
        "source_split": "test",
        "expected_rows": 745,
        "audio_source_repo": EARNINGS22_REPO_ID,
        "audio_source_revision": EARNINGS22_REVISION,
    }


def _halas_row(audio_id: str = "sample.wav") -> dict:
    return {
        "audio_id": audio_id,
        "audio_duration": 1.25,
        "corrected_reference_text": "hello",
        "e22_reference_text": "Hello",
        "split": "test",
        "_halas_source_row_index": 0,
    }


def _write_wav(path, frames: int = 320) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * frames)


def test_halas_exact_test_census_and_unique_audio_are_required(monkeypatch):
    monkeypatch.setattr("nemo_skills.dataset.halas.prepare.EXPECTED_ROWS", 1)
    monkeypatch.setattr("nemo_skills.dataset.halas.prepare.EXPECTED_UNIQUE_AUDIO", 1)
    monkeypatch.setattr("nemo_skills.dataset.halas.prepare.EXPECTED_DUPLICATE_AUDIO_COUNTS", {})
    _validate_halas_rows([_halas_row()])

    with pytest.raises(ValueError, match="row-count drift"):
        _validate_halas_rows([])
    with pytest.raises(ValueError, match="unexpected split"):
        _validate_halas_rows([{**_halas_row(), "split": "train"}])


def test_halas_preserves_the_one_pinned_duplicate_as_unique_requests(monkeypatch):
    monkeypatch.setattr("nemo_skills.dataset.halas.prepare.EXPECTED_ROWS", 2)
    monkeypatch.setattr("nemo_skills.dataset.halas.prepare.EXPECTED_UNIQUE_AUDIO", 1)
    monkeypatch.setattr(
        "nemo_skills.dataset.halas.prepare.EXPECTED_DUPLICATE_AUDIO_COUNTS",
        {"duplicate.wav": 2},
    )
    rows = [_halas_row("duplicate.wav"), _halas_row("duplicate.wav")]

    _validate_halas_rows(rows)

    first = _format_entry(rows[0], "/data/halas")
    second = _format_entry(rows[1], "/data/halas")
    assert first["id"] == "halas-test-0000-duplicate.wav"
    assert second["id"] == "halas-test-0001-duplicate.wav"
    assert first["audio_filepath"] == second["audio_filepath"]


def test_halas_validates_complete_audio_headers(tmp_path):
    row = _halas_row()
    _write_wav(tmp_path / "audio" / row["audio_id"])
    _validate_audio_closure([row], tmp_path)

    _write_wav(tmp_path / "audio" / row["audio_id"], frames=319)
    with pytest.raises(ValueError, match="audio contract failed"):
        _validate_audio_closure([row], tmp_path)


def test_halas_missing_audio_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing or empty"):
        _validate_audio_closure([_halas_row()], tmp_path)


def test_halas_deterministically_downmixes_stereo_earnings22_audio(tmp_path):
    source = tmp_path / "stereo.wav"
    stereo = np.column_stack(
        (
            np.full(480, 8192, dtype=np.int16),
            np.full(480, -4096, dtype=np.int16),
        )
    )
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(stereo.tobytes())

    samples, sampling_rate = _decode_earnings22_audio({"bytes": source.read_bytes(), "path": None})
    assert samples.dtype == np.float32
    assert samples.shape == (480,)
    np.testing.assert_array_equal(samples, np.full(480, 0.0625, dtype=np.float32))
    assert sampling_rate == 16000


def test_halas_extracts_raw_earnings22_audio_without_torchcodec(tmp_path, monkeypatch):
    source = tmp_path / "source.wav"
    _write_wav(source, frames=480)

    class FakeAudio:
        def __init__(self, *, decode):
            self.decode = decode

    class FakeDataset:
        def __init__(self):
            self.cast = None

        def cast_column(self, name, feature):
            self.cast = (name, feature.decode)
            return self

        def __iter__(self):
            yield {
                "segment_id": "segment",
                "file_id": "clip",
                "audio": {"bytes": source.read_bytes(), "path": None},
            }

    dataset = FakeDataset()
    fake_datasets = types.SimpleNamespace(
        Audio=FakeAudio,
        load_dataset=lambda *args, **kwargs: dataset,
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    missing = _extract_audio_for_rows([_halas_row("segment_clip.wav")], tmp_path)

    assert missing == set()
    assert dataset.cast == ("audio", False)
    _validate_audio_closure([_halas_row("segment_clip.wav")], tmp_path)
