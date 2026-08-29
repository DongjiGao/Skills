# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

from importlib import import_module

import pytest

judge = import_module("nemo_skills.pipeline.judges.nvembed_judge")


def _create(monkeypatch: pytest.MonkeyPatch, judge_container=None):
    captured = {}
    monkeypatch.setattr(
        judge,
        "get_remaining_jobs",
        lambda **kwargs: {None: [None]},
    )

    def fake_add_task(exp, **kwargs):
        captured.update(kwargs)
        return "judge-task"

    monkeypatch.setattr(judge, "add_task", fake_add_task)
    tasks = judge.create_judge_tasks(
        exp=object(),
        expname="eval",
        benchmark="mmau-pro-nonmusic.closed_form",
        judge_pipeline_args={
            "input_file": "/results/tmp/output.jsonl",
            "output_dir": "/results/final",
        },
        rerun_done=False,
        log_dir="/logs",
        output_dir="/unused",
        cluster_config={
            "executor": "slurm",
            "containers": {
                "nemo-skills": "skills-eval.sqsh",
                "vllm": "server.sqsh",
            },
        },
        judge_server_gpus=None,
        judge_server_nodes=None,
        partition="cpu",
        account="account",
        judge_container=judge_container,
        run_after=None,
        reuse_code_exp=None,
        reuse_code=False,
        dependent_tasks=[],
        all_tasks=[],
        _task_dependencies=[],
        installation_command="python3 -c 'import contractions'",
        skip_hf_home_check=True,
        sbatch_kwargs={},
    )
    assert tasks == ["judge-task"]
    return captured


def test_nvembed_judge_prefers_eval_image_and_baked_dependencies(monkeypatch):
    captured = _create(monkeypatch)

    assert captured["container"] == "skills-eval.sqsh"
    assert "NEMO_SKILLS_NVEMBED_DEPS_DIR" in captured["cmd"]
    assert "/opt/nemo-skills-nvembed" in captured["cmd"]
    assert 'if [[ ! -d "$NVEMBED_DEPS_DIR" ]]' in captured["cmd"]
    assert "python3 -m pip install" in captured["cmd"]


def test_nvembed_judge_explicit_container_wins(monkeypatch):
    captured = _create(monkeypatch, judge_container="explicit.sqsh")

    assert captured["container"] == "explicit.sqsh"
