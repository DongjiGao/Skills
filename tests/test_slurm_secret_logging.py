import logging

from nemo_skills.pipeline.utils import cluster
from nemo_skills.pipeline.utils.exp import (
    _disable_nemorun_slurm_tracing,
    _install_nemorun_slurm_trace_guard,
)


def test_disable_nemorun_slurm_tracing_preserves_fail_fast_mode():
    script = """#!/bin/bash
set -evx
export NGC_API_KEY=secret-value
srun workload
"""

    rendered = _disable_nemorun_slurm_tracing(script)

    assert "set -evx" not in rendered
    assert "\nset -e\n" in rendered
    assert "export NGC_API_KEY=secret-value" in rendered


def test_install_nemorun_slurm_trace_guard_is_idempotent():
    from nemo_run.core.execution.slurm import SlurmBatchRequest

    _install_nemorun_slurm_trace_guard()
    first = SlurmBatchRequest.materialize
    _install_nemorun_slurm_trace_guard()

    assert SlurmBatchRequest.materialize is first
    assert first._nemo_skills_secret_safe is True


def test_placeholder_resolution_does_not_log_secret(monkeypatch, caplog):
    secret = "must-not-appear-in-logs"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    caplog.set_level(logging.INFO)
    cluster._logged_optional_env_vars.clear()

    resolved = cluster.get_env_variables(
        {"env_vars": ["OPENAI_API_KEY=$OPENAI_API_KEY"]}
    )

    assert resolved["OPENAI_API_KEY"] == secret
    assert secret not in caplog.text
    assert "Resolved environment variable OPENAI_API_KEY" in caplog.text
