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
import subprocess

from nemo_skills.pipeline.utils import server


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
