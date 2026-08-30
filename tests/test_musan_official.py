# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys

from nemo_skills.dataset import musan
from nemo_skills.dataset.musan import paper, prepare


def test_musan_is_regular_four_generation_audio_benchmark():
    assert not hasattr(musan, "IS_BENCHMARK_GROUP")
    assert not hasattr(musan, "BENCHMARKS")
    assert musan.NUM_SAMPLES == 4
    assert musan.METRICS_TYPE == "audio"
    assert "enable_audio=true" in musan.GENERATION_ARGS


def test_musan_manifest_matches_pinned_gym_protocol_for_every_category():
    for category in prepare.ALL_CATEGORIES:
        row = prepare.create_manifest_entry(
            audio_filename=f"musan_{category}_000000.wav",
            duration=2.5,
            category=category,
            sample_id=0,
            label=f"source-{category}",
        )
        assert row["messages"][0] == {"role": "system", "content": prepare.SYSTEM_PROMPT}
        assert row["messages"][1]["content"] == prepare.USER_PROMPT
        assert row["expected_answer"] == ""
        assert row["task_type"] == "Hallucination"
        assert row["subset_for_metrics"] == f"musan_{category}"
        assert row["protocol_revision"] == "68cb9969d656d259846c45e7369acfe6529f0e97"
        assert row["hallucination_char_rate_threshold"] == 1500.0
        assert row["hallucination_threshold_operator"] == ">"
        assert row["num_repeated_generations"] == 4


def test_musan_prepare_defaults_to_complete_openslr(monkeypatch, tmp_path):
    observed = {}

    def fake_load(source, output_dir):
        observed["source"] = source
        return tmp_path, source

    monkeypatch.setattr(prepare, "load_dataset_from_source", fake_load)

    def fake_process(*, category, **kwargs):
        row = prepare.create_manifest_entry(
            audio_filename=f"musan_{category}_000000.wav",
            duration=1.0,
            category=category,
            sample_id=0,
            label=f"source-{category}",
        )
        return 1, [row]

    monkeypatch.setattr(prepare, "process_category", fake_process)
    monkeypatch.setattr(sys, "argv", ["prepare", "--output-dir", str(tmp_path)])
    prepare.main()

    assert observed["source"] == "openslr"
    assert prepare.DEFAULT_SOURCE == "openslr"
    assert prepare.ALL_CATEGORIES == ("noise", "music", "speech")


def test_musan_official_inventory_counts_are_pinned():
    assert prepare.OFFICIAL_CATEGORY_FILE_COUNTS == {
        "noise": 930,
        "music": 660,
        "speech": 426,
    }
    assert prepare.OFFICIAL_TOTAL_FILES == 2016


def test_canary_paper_profile_is_one_greedy_generation_and_not_gym():
    assert paper.NUM_SAMPLES == 1
    assert paper.METRICS_TYPE == "audio"
    assert prepare.CANARY_PAPER_CATEGORIES == ("music", "noise")
    assert prepare.CANARY_PAPER_PROTOCOL == "canary-table6-reconstruction-v1"
    assert prepare.OPENSLR_MD5 == "0c472d4fc0c5141eca47ad1ffeb2a7df"


def test_canary_paper_manifest_excludes_speech_and_records_limit(monkeypatch, tmp_path):
    def fake_load(source, output_dir):
        assert source == "openslr"
        return tmp_path, source

    def fake_process(*, category, **kwargs):
        row = prepare.create_manifest_entry(
            audio_filename=f"musan_{category}_000000.wav",
            duration=60.0,
            category=category,
            sample_id=0,
            label=f"source-{category}",
        )
        return 1, [row]

    monkeypatch.setattr(prepare, "load_dataset_from_source", fake_load)
    monkeypatch.setattr(prepare, "process_category", fake_process)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare", "--output-dir", str(tmp_path)],
    )
    prepare.main()

    paper_rows = [json.loads(line) for line in (tmp_path / "paper" / "test.jsonl").read_text().splitlines()]
    assert [row["category"] for row in paper_rows] == ["noise", "music"]
    assert all(row["category"] != "speech" for row in paper_rows)
    assert all(row["messages"][0]["content"] == "" for row in paper_rows)
    assert all(row["messages"][1]["content"] == prepare.CANARY_PAPER_USER_PROMPT for row in paper_rows)
    assert all(row["protocol_profile"] == prepare.CANARY_PAPER_PROTOCOL for row in paper_rows)
    assert all(row["protocol_revision"] == "arxiv:2406.19674v1" for row in paper_rows)
    assert all(row["num_repeated_generations"] == 1 for row in paper_rows)
    assert all("does not publish" in row["comparability_limit"] for row in paper_rows)


def test_musan_prepare_fails_closed_on_partial_category(monkeypatch, tmp_path):
    monkeypatch.setattr(
        prepare,
        "load_dataset_from_source",
        lambda source, output_dir: (tmp_path, source),
    )

    def fake_process(*, category, **kwargs):
        if category == "music":
            raise RuntimeError("corrupt source")
        return 0, []

    monkeypatch.setattr(prepare, "process_category", fake_process)
    monkeypatch.setattr(sys, "argv", ["prepare", "--output-dir", str(tmp_path)])

    try:
        prepare.main()
    except RuntimeError as error:
        assert "music: corrupt source" in str(error)
    else:
        raise AssertionError("partial MUSAN preparation must fail")


def test_musan_prepare_fails_closed_when_source_load_fails(monkeypatch, tmp_path):
    def fail_load(source, output_dir):
        raise ValueError("checksum mismatch")

    monkeypatch.setattr(prepare, "load_dataset_from_source", fail_load)
    monkeypatch.setattr(sys, "argv", ["prepare", "--output-dir", str(tmp_path)])

    try:
        prepare.main()
    except RuntimeError as error:
        assert "Failed to load MUSAN dataset from openslr" in str(error)
        assert "checksum mismatch" in str(error)
    else:
        raise AssertionError("MUSAN source failures must propagate")


def test_musan_file_processing_rejects_partial_corpus(monkeypatch, tmp_path):
    category_dir = tmp_path / "source" / "music" / "album"
    category_dir.mkdir(parents=True)
    (category_dir / "broken.wav").write_bytes(b"not audio")

    def fail_read(path):
        raise RuntimeError("invalid WAV")

    monkeypatch.setattr(prepare.sf, "read", fail_read)

    try:
        prepare.process_category_from_files(
            category="music", dataset_path=tmp_path / "source", output_dir=tmp_path / "out"
        )
    except RuntimeError as error:
        assert "Failed to process 1 of 1 music files" in str(error)
    else:
        raise AssertionError("partial source processing must fail")


def test_musan_file_processing_rejects_official_inventory_drift(tmp_path):
    category_dir = tmp_path / "source" / "music" / "album"
    category_dir.mkdir(parents=True)
    (category_dir / "only-one.wav").write_bytes(b"not read when inventory is wrong")

    try:
        prepare.process_category_from_files(
            category="music",
            dataset_path=tmp_path / "source",
            output_dir=tmp_path / "out",
            expected_count=prepare.OFFICIAL_CATEGORY_FILE_COUNTS["music"],
        )
    except RuntimeError as error:
        assert "Official OpenSLR-17 music inventory drift" in str(error)
        assert "expected 660 WAV files, got 1" in str(error)
    else:
        raise AssertionError("official inventory drift must fail")


def test_musan_openslr_processing_forwards_official_inventory(monkeypatch, tmp_path):
    observed = {}

    def fake_process_category_from_files(**kwargs):
        observed.update(kwargs)
        return 1, [{}]

    monkeypatch.setattr(prepare, "process_category_from_files", fake_process_category_from_files)
    prepare.process_category(
        category="noise",
        output_dir=tmp_path / "out",
        dataset=tmp_path / "source",
        source_type="openslr",
    )
    assert observed["expected_count"] == 930

    prepare.process_category(
        category="noise",
        output_dir=tmp_path / "out",
        dataset=tmp_path / "source",
        source_type="openslr",
        max_samples=1,
    )
    assert observed["expected_count"] is None
