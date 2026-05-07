# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Tests for nemo_skills.dataset.audit_audio_paths.

Exercise the walker, audit logic, and CLI exit codes against fixture
filesystems built in tmp_path. Companion to the lightweight unit tests
in tests/test_audio_path_prefix.py.
"""

import json
import sys
import types
from pathlib import Path

import pytest


def _stub_audio_grader_deps(monkeypatch):
    """Stub deps pulled in transitively by nemo_skills.dataset.utils."""
    monkeypatch.setitem(
        sys.modules,
        "latex2sympy2_extended",
        types.SimpleNamespace(NormalizationConfig=object, normalize_latex=lambda value, **kwargs: value),
    )
    monkeypatch.setitem(
        sys.modules,
        "math_verify",
        types.SimpleNamespace(
            LatexExtractionConfig=object,
            StringExtractionConfig=object,
            parse=lambda *args, **kwargs: [],
            verify=lambda *args, **kwargs: False,
        ),
    )


def _make_record(audio_path):
    """Build a minimal record covering several common audio-path field shapes."""
    return {
        "expected_answer": "hello",
        "audio_filepath": audio_path,
        "audio_path": audio_path,
        "messages": [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": "Transcribe.", "audio": {"path": audio_path, "duration": 1.0}},
        ],
    }


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_walk_strings_finds_all_leaves(monkeypatch):
    _stub_audio_grader_deps(monkeypatch)
    from nemo_skills.dataset.audit_audio_paths import walk_strings

    tree = {
        "a": "x",
        "b": [1, "y", {"c": "z"}],
        "d": {"e": ["w"], "f": 5},
    }
    leaves = sorted(walk_strings(tree))
    assert leaves == ["w", "x", "y", "z"]


def test_audit_clean_jsonl(monkeypatch, tmp_path):
    """Every audio path resolves -> no issues."""
    _stub_audio_grader_deps(monkeypatch)
    from nemo_skills.dataset.audit_audio_paths import AuditStats, audit_jsonl

    bench_audio = tmp_path / "asr-leaderboard" / "data" / "librispeech_clean"
    bench_audio.mkdir(parents=True)
    (bench_audio / "001.flac").touch()
    (bench_audio / "002.flac").touch()

    jsonl = tmp_path / "asr-leaderboard" / "test.jsonl"
    _write_jsonl(
        jsonl,
        [
            _make_record("/data/asr-leaderboard/data/librispeech_clean/001.flac"),
            _make_record("/data/asr-leaderboard/data/librispeech_clean/002.flac"),
        ],
    )

    stats: AuditStats = audit_jsonl(jsonl, "/data", tmp_path)

    assert stats.rows == 2
    assert stats.audio_paths == 6  # 3 audio fields per row, 2 rows
    assert stats.missing == []
    assert stats.stale == []
    assert stats.wrong_prefix == []
    assert not stats.has_issues


def test_audit_missing_file(monkeypatch, tmp_path):
    _stub_audio_grader_deps(monkeypatch)
    from nemo_skills.dataset.audit_audio_paths import audit_jsonl

    bench_audio = tmp_path / "covost2" / "audio" / "fr" / "test"
    bench_audio.mkdir(parents=True)
    (bench_audio / "exists.wav").touch()

    jsonl = tmp_path / "covost2" / "test-asr.jsonl"
    _write_jsonl(
        jsonl,
        [
            _make_record("/data/covost2/audio/fr/test/exists.wav"),
            _make_record("/data/covost2/audio/fr/test/missing.wav"),
        ],
    )

    stats = audit_jsonl(jsonl, "/data", tmp_path)
    assert len(stats.missing) == 3  # 3 audio fields point at missing.wav
    assert all("missing.wav" in path for _, path, _ in stats.missing)
    assert stats.stale == []
    assert stats.has_issues


def test_audit_stale_path(monkeypatch, tmp_path):
    """Strings containing /dataset/ are reported even if they don't end in an audio extension."""
    _stub_audio_grader_deps(monkeypatch)
    from nemo_skills.dataset.audit_audio_paths import audit_jsonl

    jsonl = tmp_path / "fleurs" / "test-asr.jsonl"
    _write_jsonl(jsonl, [_make_record("/dataset/fleurs/audio/en_us/x.wav")])

    stats = audit_jsonl(jsonl, "/data", tmp_path)
    assert stats.stale, "expected /dataset/ substrings to be flagged"
    # Same string appears in audio_filepath, audio_path, messages[1].audio.path
    assert len(stats.stale) == 3
    # And it's also a wrong_prefix because it doesn't start with /data/
    assert len(stats.wrong_prefix) == 3
    assert stats.has_issues


def test_audit_wrong_prefix(monkeypatch, tmp_path):
    """Audio paths under an unexpected prefix are reported."""
    _stub_audio_grader_deps(monkeypatch)
    from nemo_skills.dataset.audit_audio_paths import audit_jsonl

    jsonl = tmp_path / "fleurs" / "test-asr.jsonl"
    _write_jsonl(jsonl, [_make_record("/some-other-mount/fleurs/audio/x.wav")])

    stats = audit_jsonl(jsonl, "/data", tmp_path)
    assert len(stats.wrong_prefix) == 3
    assert stats.missing == []
    assert stats.stale == []
    assert stats.has_issues


def test_audit_skips_non_audio_strings(monkeypatch, tmp_path):
    """Strings without an audio extension don't get prefix-checked even if they look path-y."""
    _stub_audio_grader_deps(monkeypatch)
    from nemo_skills.dataset.audit_audio_paths import audit_jsonl

    bench_audio = tmp_path / "asr-leaderboard" / "data" / "librispeech_clean"
    bench_audio.mkdir(parents=True)
    (bench_audio / "001.flac").touch()

    jsonl = tmp_path / "asr-leaderboard" / "test.jsonl"
    record = _make_record("/data/asr-leaderboard/data/librispeech_clean/001.flac")
    record["arbitrary_field"] = "/no_think"  # starts with / but not an audio path
    record["messages"][0]["content"] = "/sys-prompt-marker"
    _write_jsonl(jsonl, [record])

    stats = audit_jsonl(jsonl, "/data", tmp_path)
    assert not stats.has_issues


def test_audit_skips_non_absolute_audio_extension_strings(monkeypatch, tmp_path):
    """Identifier-like strings ending in audio extensions but not absolute paths are not flagged.

    Real HF datasets surface fields like ``id`` whose values are e.g.
    ``"4483338/281.wav"`` -- a relative identifier that happens to look
    like a wav filename. These must not trigger wrong_prefix because they
    are not container audio paths.
    """
    _stub_audio_grader_deps(monkeypatch)
    from nemo_skills.dataset.audit_audio_paths import audit_jsonl

    bench_audio = tmp_path / "asr-leaderboard" / "data" / "earnings22"
    bench_audio.mkdir(parents=True)
    (bench_audio / "4483338_281.flac").touch()

    jsonl = tmp_path / "asr-leaderboard" / "earnings22.jsonl"
    record = _make_record("/data/asr-leaderboard/data/earnings22/4483338_281.flac")
    # mimic the HF identifier on the prepared record -- .wav extension, no leading slash
    record["id"] = "4483338/281.wav"
    _write_jsonl(jsonl, [record])

    stats = audit_jsonl(jsonl, "/data", tmp_path)
    assert not stats.has_issues, f"Identifier 'id' field should not have been flagged; stats={stats}"


def test_audit_data_dir_returns_zero_when_clean(monkeypatch, tmp_path, capsys):
    _stub_audio_grader_deps(monkeypatch)
    from nemo_skills.dataset.audit_audio_paths import audit_data_dir

    bench_audio = tmp_path / "asr-leaderboard" / "data" / "librispeech_clean"
    bench_audio.mkdir(parents=True)
    (bench_audio / "001.flac").touch()

    jsonl = tmp_path / "asr-leaderboard" / "test.jsonl"
    _write_jsonl(jsonl, [_make_record("/data/asr-leaderboard/data/librispeech_clean/001.flac")])

    rc = audit_data_dir(tmp_path, "/data")

    assert rc == 0
    out = capsys.readouterr().out
    assert "[OK]" in out
    assert "missing=0" in out


def test_audit_data_dir_returns_one_on_issues(monkeypatch, tmp_path, capsys):
    _stub_audio_grader_deps(monkeypatch)
    from nemo_skills.dataset.audit_audio_paths import audit_data_dir

    jsonl = tmp_path / "fleurs" / "test-asr.jsonl"
    _write_jsonl(jsonl, [_make_record("/data/fleurs/audio/en_us/missing.wav")])

    rc = audit_data_dir(tmp_path, "/data")

    assert rc == 1
    out = capsys.readouterr().out
    assert "[ISSUE]" in out


def test_main_invalid_data_dir(monkeypatch, tmp_path, capsys):
    _stub_audio_grader_deps(monkeypatch)
    from nemo_skills.dataset.audit_audio_paths import main

    rc = main(["--data-dir", str(tmp_path / "does_not_exist")])

    assert rc == 2
    err = capsys.readouterr().err
    assert "is not a directory" in err


@pytest.mark.parametrize("audio_prefix_arg", ["/data", "/data/", "/dataset"])
def test_audit_data_dir_handles_trailing_slash_and_alt_prefix(monkeypatch, tmp_path, audio_prefix_arg):
    _stub_audio_grader_deps(monkeypatch)
    from nemo_skills.dataset.audit_audio_paths import audit_data_dir

    bench_audio = tmp_path / "asr-leaderboard" / "data" / "librispeech_clean"
    bench_audio.mkdir(parents=True)
    (bench_audio / "001.flac").touch()

    audio_path = audio_prefix_arg.rstrip("/") + "/asr-leaderboard/data/librispeech_clean/001.flac"
    jsonl = tmp_path / "asr-leaderboard" / "test.jsonl"
    _write_jsonl(jsonl, [_make_record(audio_path)])

    rc = audit_data_dir(tmp_path, audio_prefix_arg)
    assert rc == 0
