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

import os
import socket
import subprocess

import pytest

from nemo_skills.pipeline.utils import server
from nemo_skills.pipeline.utils.runtime_port import (
    PORT_COUNT,
    PORT_MIN,
    allocate_runtime_port,
    wait_for_runtime_port,
)
from nemo_skills.pipeline.utils.scripts.server import ServerScript


def _expand_port(expression: str, **environment: str) -> int:
    env = os.environ.copy()
    for name in ("SLURM_JOB_ID", "SLURM_HET_GROUP", "SLURM_ARRAY_TASK_ID"):
        env.pop(name, None)
    env.update(environment)
    return int(
        subprocess.run(
            ["bash", "-c", f"echo {expression}"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        ).stdout.strip()
    )


def test_slurm_runtime_ports_space_adjacent_eval_jobs(monkeypatch):
    monkeypatch.setenv("NEMO_SKILLS_USE_SLURM_JOB_ID_PORTS", "1")
    monkeypatch.setattr(server, "_SLURM_JOB_PORT_OFFSET", 0)

    first = server.get_free_port(strategy="random")
    second = server.get_free_port(strategy="random")
    first_port = _expand_port(first, SLURM_JOB_ID="16905972")
    second_port = _expand_port(second, SLURM_JOB_ID="16905973")

    assert 20000 <= first_port < 65000
    assert 20000 <= second_port < 65000
    assert abs(first_port - second_port) >= 1024


def test_slurm_runtime_ports_distinguish_heterogeneous_components(monkeypatch):
    monkeypatch.setenv("NEMO_SKILLS_USE_SLURM_JOB_ID_PORTS", "1")
    monkeypatch.setattr(server, "_SLURM_JOB_PORT_OFFSET", 0)

    first = server.get_free_port(strategy="random")
    second = server.get_free_port(strategy="random")
    first_port = _expand_port(first, SLURM_JOB_ID="42", SLURM_HET_GROUP="0")
    second_port = _expand_port(second, SLURM_JOB_ID="42", SLURM_HET_GROUP="1")

    assert first_port != second_port


def test_runtime_port_skips_occupied_candidate_and_preserves_client_agreement(tmp_path):
    occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    for base_port in range(32000, 32100):
        try:
            occupied.bind(("0.0.0.0", base_port))
            break
        except OSError:
            continue
    else:
        raise RuntimeError("Could not reserve a test port")

    state_file = tmp_path / "port"
    selected = allocate_runtime_port(state_file, base_port)

    assert PORT_MIN <= selected < PORT_MIN + PORT_COUNT
    assert selected != base_port
    assert wait_for_runtime_port(state_file, timeout=0.1) == selected
    occupied.close()


def test_server_script_uses_runtime_port_handoff_for_slurm_ports(monkeypatch, tmp_path):
    monkeypatch.setenv("NEMO_SKILLS_USE_SLURM_JOB_ID_PORTS", "1")
    monkeypatch.setattr(server, "_SLURM_JOB_PORT_OFFSET", 0)
    script = ServerScript(
        server_type="vllm",
        model_path="model-name",
        cluster_config={},
        num_gpus=1,
        port=None,
        allocate_port=True,
    )
    script.het_group_index = 2

    script.configure_runtime_port(str(tmp_path))

    assert isinstance(script.inline, str)
    assert "nemo_skills.pipeline.utils.runtime_port allocate" in script.inline
    assert "--base-port $((20000+(${SLURM_JOB_ID:-0}*7919+2*1543" in script.inline
    assert "-2-0" in script.inline
    assert isinstance(script.port, str)
    assert "nemo_skills.pipeline.utils.runtime_port wait" in script.port
    assert "-2-0" in script.port
    server_state = script.inline.split("--state-file ", 1)[1].split(" ", 1)[0]
    client_state = script.port.split("--state-file ", 1)[1].split(")", 1)[0]
    assert server_state == client_state


def test_runtime_port_selects_free_base_and_reuses_published_state(tmp_path):
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    for base_port in range(32100, 32200):
        try:
            probe.bind(("0.0.0.0", base_port))
            break
        except OSError:
            continue
    else:
        raise RuntimeError("Could not find a free test port")
    probe.close()

    state_file = tmp_path / "port"
    selected = allocate_runtime_port(state_file, base_port)
    assert selected == base_port

    occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupied.bind(("0.0.0.0", selected))
    try:
        assert allocate_runtime_port(state_file, base_port + 1) == selected
    finally:
        occupied.close()


def test_runtime_port_wait_times_out_without_publication(tmp_path):
    with pytest.raises(TimeoutError, match="Timed out waiting for runtime port"):
        wait_for_runtime_port(
            tmp_path / "missing-port",
            timeout=0.01,
            poll_interval=0.001,
        )
