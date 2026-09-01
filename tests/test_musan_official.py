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

import hashlib
import io
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


class _FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, *, status: int, headers: dict[str, str]):
        super().__init__(payload)
        self.status = status
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _patch_tiny_openslr(monkeypatch, payload: bytes) -> None:
    monkeypatch.setattr(prepare, "OPENSLR_ARCHIVE_BYTES", len(payload))
    monkeypatch.setattr(
        prepare,
        "OPENSLR_MD5",
        hashlib.md5(payload, usedforsecurity=False).hexdigest(),
    )
    monkeypatch.setattr(prepare, "OPENSLR_ETAG", '"test-etag"')
    monkeypatch.setattr(prepare, "DOWNLOAD_CHUNK_BYTES", 3)


def test_musan_resumes_legacy_partial_only_after_range_and_entity_validation(monkeypatch, tmp_path):
    payload = b"authenticated archive"
    _patch_tiny_openslr(monkeypatch, payload)
    legacy_path = tmp_path / "musan.tar.gz"
    offset = 7
    legacy_path.write_bytes(payload[:offset])
    observed_request = {}

    def fake_urlopen(request):
        observed_request["range"] = request.get_header("Range")
        observed_request["if_range"] = request.get_header("If-range")
        return _FakeResponse(
            payload[offset:],
            status=206,
            headers={
                "ETag": prepare.OPENSLR_ETAG,
                "Content-Length": str(len(payload) - offset),
                "Content-Range": f"bytes {offset}-{len(payload) - 1}/{len(payload)}",
            },
        )

    monkeypatch.setattr(prepare.urllib.request, "urlopen", fake_urlopen)
    output_path = prepare._download_openslr_archive(tmp_path)
    assert output_path == legacy_path
    assert output_path.read_bytes() == payload
    assert observed_request == {"range": f"bytes={offset}-", "if_range": prepare.OPENSLR_ETAG}
    assert not (tmp_path / "musan.tar.gz.part").exists()


def test_musan_restarts_when_server_does_not_authenticate_partial(monkeypatch, tmp_path):
    payload = b"complete pinned archive"
    _patch_tiny_openslr(monkeypatch, payload)
    partial_path = tmp_path / "musan.tar.gz.part"
    partial_path.write_bytes(b"untrusted")
    responses = [
        _FakeResponse(
            payload,
            status=200,
            headers={"ETag": prepare.OPENSLR_ETAG, "Content-Length": str(len(payload))},
        ),
        _FakeResponse(
            payload,
            status=200,
            headers={"ETag": prepare.OPENSLR_ETAG, "Content-Length": str(len(payload))},
        ),
    ]

    monkeypatch.setattr(prepare.urllib.request, "urlopen", lambda _request: responses.pop(0))
    output_path = prepare._download_openslr_archive(tmp_path)
    assert output_path.read_bytes() == payload
    assert not responses
    assert not partial_path.exists()


def test_musan_short_authenticated_response_is_retained_for_resume(monkeypatch, tmp_path):
    payload = b"complete pinned archive"
    _patch_tiny_openslr(monkeypatch, payload)
    monkeypatch.setattr(
        prepare.urllib.request,
        "urlopen",
        lambda _request: _FakeResponse(
            payload[:5],
            status=200,
            headers={"ETag": prepare.OPENSLR_ETAG, "Content-Length": str(len(payload))},
        ),
    )
    try:
        prepare._download_openslr_archive(tmp_path)
    except RuntimeError as error:
        assert "retained for resume" in str(error)
    else:
        raise AssertionError("short MUSAN response must fail")
    assert (tmp_path / "musan.tar.gz.part").read_bytes() == payload[:5]


def test_musan_uses_nemo_skills_data_root_instead_of_package_directory(monkeypatch, tmp_path):
    observed = {}
    monkeypatch.setenv("NEMO_SKILLS_DATA_DIR", str(tmp_path))

    def fail_after_observing_root(_source, output_dir):
        observed["output_dir"] = output_dir
        raise RuntimeError("stop after output-root resolution")

    monkeypatch.setattr(prepare, "load_dataset_from_source", fail_after_observing_root)
    monkeypatch.setattr(sys, "argv", ["prepare"])
    try:
        prepare.main()
    except RuntimeError as error:
        assert "stop after output-root resolution" in str(error)
    else:
        raise AssertionError("test sentinel must stop MUSAN preparation")
    assert observed["output_dir"] == tmp_path / "musan"


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
