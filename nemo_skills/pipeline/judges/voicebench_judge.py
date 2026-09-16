# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""CPU task creator for exact, resumable VoiceBench hosted judging."""

import hashlib
import shlex

from nemo_skills.pipeline.utils import add_task
from nemo_skills.pipeline.utils.generation import get_remaining_jobs


def create_judge_tasks(
    exp,
    expname,
    benchmark,
    judge_pipeline_args,
    rerun_done,
    log_dir,
    output_dir,
    cluster_config,
    judge_server_gpus,
    judge_server_nodes,
    partition,
    account,
    judge_container,
    run_after,
    reuse_code_exp,
    reuse_code,
    dependent_tasks,
    all_tasks,
    _task_dependencies,
    installation_command,
    skip_hf_home_check,
    sbatch_kwargs,
):
    del output_dir, judge_server_gpus, judge_server_nodes
    input_file = judge_pipeline_args.get("input_file")
    output_dir_path = judge_pipeline_args.get("output_dir")
    if not input_file or not output_dir_path:
        raise ValueError("VoiceBench judge requires non-null input_file and output_dir")

    remaining = get_remaining_jobs(
        cluster_config=cluster_config,
        output_dir=output_dir_path,
        random_seeds=[None],
        chunk_ids=[None],
        rerun_done=rerun_done,
    )
    if not remaining or all(not chunks for chunks in remaining.values()):
        return []

    api_key_env = str(judge_pipeline_args.get("api_key_env", "NVIDIA_INFERENCE_HUB_API_KEY"))
    base_url = str(judge_pipeline_args.get("base_url", "https://inference-api.nvidia.com/v1"))
    judge_model = str(judge_pipeline_args.get("judge_model", "openai/openai/gpt-4o-mini"))
    max_workers = int(judge_pipeline_args.get("max_workers", 12))
    if max_workers < 1:
        raise ValueError("VoiceBench judge max_workers must be positive")

    script_args = [
        "--input-file",
        str(input_file),
        "--output-dir",
        str(output_dir_path),
        "--api-key-env",
        api_key_env,
        "--base-url",
        base_url,
        "--judge-model",
        judge_model,
        "--max-workers",
        str(max_workers),
    ]
    run_cmd = (
        "python3 -m nemo_skills.evaluation.evaluator.voicebench_offline_judge "
        + " ".join(shlex.quote(arg) for arg in script_args)
    )

    configured_containers = cluster_config.get("containers", {})
    resolved_container = judge_container or configured_containers.get("nemo-skills")
    if resolved_container is None:
        raise ValueError("VoiceBench judge requires a nemo-skills container")

    child_sbatch = dict(sbatch_kwargs or {})
    child_sbatch.setdefault("cpus_per_task", max_workers * 2)
    child_sbatch.setdefault("mem", "32G" if benchmark.endswith("sd-qa") else "8G")
    digest = hashlib.sha256(f"{expname}|{benchmark}|voicebench-judge".encode()).hexdigest()[:12]
    task_name = f"voicebench-judge-{benchmark.rsplit('.', 1)[-1]}-{digest}"
    judge_task = add_task(
        exp,
        cmd=run_cmd,
        task_name=task_name,
        log_dir=log_dir + "/judge",
        container=resolved_container,
        cluster_config=cluster_config,
        num_gpus=0,
        num_nodes=1,
        partition=partition,
        account=account,
        run_after=run_after,
        reuse_code_exp=reuse_code_exp,
        reuse_code=reuse_code,
        task_dependencies=(
            dependent_tasks
            if cluster_config["executor"] == "slurm"
            else all_tasks + _task_dependencies
        ),
        installation_command=installation_command,
        skip_hf_home_check=skip_hf_home_check,
        sbatch_kwargs=child_sbatch,
    )
    return [judge_task]
