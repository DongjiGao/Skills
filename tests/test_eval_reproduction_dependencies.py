# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local tests for scorer normalization, hostname resolution, and packaging."""

import os
import subprocess
from types import SimpleNamespace

import pytest

from nemo_skills.evaluation.metrics.utils.chime8_normalizer import _EnglishReverseNumberNormalizer
from nemo_skills.pipeline.utils import packager
from nemo_skills.pipeline.utils.scripts.eval import _resolve_server_ipv4_once


def test_chime8_pathological_digits_survive_without_integer_conversion():
    normalizer = _EnglishReverseNumberNormalizer()
    token = "7" * 5000  # Exceeds Python's default integer-string conversion limit.
    assert normalizer(token) == token
    assert normalizer("42") == "forty two"
    assert normalizer("000") == "thousand"


def test_ipv4_resolution_deduplicates_before_client_fanout():
    command = _resolve_server_ipv4_once(["SLURM_MASTER_NODE", "SLURM_MASTER_NODE"])
    fake_dns = 'getent() { test "$1" = ahostsv4 || return 1; echo "192.0.2.7 STREAM $2"; }; '
    result = subprocess.run(
        ["bash", "-c", fake_dns + command + 'echo "$SLURM_MASTER_NODE"'],
        env={**os.environ, "SLURM_MASTER_NODE": "fake-server"},
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "192.0.2.7"
    assert command.count("getent ahostsv4") == 1


def test_ipv4_resolution_fails_before_client_when_hostname_missing():
    env = os.environ.copy()
    env.pop("SLURM_MASTER_NODE", None)
    result = subprocess.run(
        ["bash", "-c", _resolve_server_ipv4_once(["SLURM_MASTER_NODE"]) + "echo CLIENT_STARTED"],
        env=env,
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert result.returncode == 43
    assert "missing Slurm server hostname" in result.stderr
    assert "CLIENT_STARTED" not in result.stdout


def test_forced_pattern_packaging_never_scans_parent_git_repo(monkeypatch, tmp_path):
    root = tmp_path / "nemo_skills"
    extra = tmp_path / "frozen-benchmark"
    root.mkdir()
    extra.mkdir()
    monkeypatch.setenv("NEMO_SKILLS_FORCE_PATTERN_PACKAGER", "1")
    monkeypatch.setattr(
        packager,
        "EXTERNAL_REPOS",
        {
            "nemo_skills": packager.RepoMetadata("nemo_skills", root),
            "frozen": packager.RepoMetadata("frozen", extra),
        },
    )

    def forbidden(*args, **kwargs):
        pytest.fail("forced pattern mode must not discover/scan parent git repos")

    monkeypatch.setattr(packager, "get_git_repo_path", forbidden)
    monkeypatch.setattr(packager.run, "GitArchivePackager", forbidden)
    monkeypatch.setattr(packager.run, "PatternPackager", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(packager.run, "HybridPackager", lambda **kwargs: SimpleNamespace(**kwargs))
    result = packager.get_packager()
    assert result.sub_packagers["frozen"].include_pattern == [str(extra / "*")]
    assert result.sub_packagers["frozen"].relative_path == [str(extra.parent)]
