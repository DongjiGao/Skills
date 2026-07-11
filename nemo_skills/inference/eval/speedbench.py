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

"""SPEED-Bench speculative decoding evaluation (SGLang + vLLM, multi-turn)."""

import json
import logging
import os
import sys
from typing import Any

import hydra

from nemo_skills.inference.eval.specdec import (
    SpecdecGenerationConfig,
    SpecdecGenerationTask,
    find_sglang_metrics_file,
)
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class SpeedbenchGenerationConfig(SpecdecGenerationConfig):
    """SPEED-Bench generation config for speculative decoding evaluation.

    Extends the base speculative-decoding config with SGLang metrics file
    support.  The metrics directory is shared between the pipeline process
    (which builds the server command) and the generation worker process.
    """

    # Directory to write SGLang metrics to.  If not specified, will use a
    # temporary directory created at launch.  This is the only reliable way
    # to share the tempdir between the pipeline process (which builds the
    # server command) and this generation worker process.
    metrics_file_dir: str | None = None
    max_concurrent_requests: int = 32

    def _post_init_validate_server(self):
        super()._post_init_validate_server()
        assert self.server["server_type"] in ("sglang", "vllm"), (
            f"server_type must be 'sglang' or 'vllm' for SPEED-Bench generation, "
            f"got '{self.server['server_type']}'"
        )


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_speedbench_generation_config", node=SpeedbenchGenerationConfig)


class SpeedbenchGenerationTask(SpecdecGenerationTask):
    """Generation task for SPEED-Bench speculative decoding evaluation.

    Extends the base ``SpecdecGenerationTask`` with:
    - SGLang per-request metrics file (preferred) with Prometheus fallback
    - Multi-turn conversation handling with request-ID tracking
    - Server command injection for ``--export-metrics-to-file``
    - Acceptance stats injected into ``eval_config`` for the evaluator
    """

    _sglang_metrics_dir: str | None = None

    def __init__(self, cfg: SpeedbenchGenerationConfig):
        super().__init__(cfg)

    @classmethod
    def _ensure_sglang_metrics_dir(cls) -> str:
        """Return (and lazily create) a unique temp directory for SGLang metrics."""
        if cls._sglang_metrics_dir is None:
            import tempfile

            cls._sglang_metrics_dir = tempfile.mkdtemp(prefix="sglang-metrics-")
            LOG.info("Created SGLang metrics temp directory: %s", cls._sglang_metrics_dir)
        return cls._sglang_metrics_dir

    @classmethod
    def get_generation_default_args(cls) -> str:
        """Pass the SGLang metrics temp-directory to the generation worker."""
        metrics_dir = cls._ensure_sglang_metrics_dir()
        return f"++metrics_file_dir={metrics_dir}"

    @classmethod
    def get_server_command_fn(cls) -> callable:
        """Return a server command builder that injects SGLang metrics flags."""
        from nemo_skills.pipeline.utils import get_server_command

        metrics_dir = cls._ensure_sglang_metrics_dir()
        sglang_metrics_args = f"--enable-metrics --export-metrics-to-file --export-metrics-to-file-dir {metrics_dir}"

        def specdec_server_command(
            server_type,
            num_gpus,
            num_nodes,
            model_path,
            cluster_config,
            server_port,
            server_args="",
            server_entrypoint=None,
        ):
            if server_type == "sglang":
                server_args = f"{server_args} {sglang_metrics_args}".strip()
            return get_server_command(
                server_type=server_type,
                num_gpus=num_gpus,
                num_nodes=num_nodes,
                model_path=model_path,
                cluster_config=cluster_config,
                server_port=server_port,
                server_args=server_args,
                server_entrypoint=server_entrypoint,
            )

        return specdec_server_command

    def inject_sglang_metrics(self, metrics_file_path: str) -> dict[str, Any] | None:
        """Aggregate per-request SGLang metrics and inject them into output JSONL."""
        if not os.path.exists(metrics_file_path):
            LOG.warning("SGLang metrics file not found: %s, skipping metrics injection", metrics_file_path)
            return None

        metrics: dict[str, Any] = {}
        with open(metrics_file_path, "rt", encoding="utf-8") as fin:
            for i, line in enumerate(fin):
                try:
                    metric = json.loads(line)
                    metrics[metric["id"]] = {
                        "acceptance_length": metric["spec_accept_length"],
                        "acceptance_rate": metric["spec_accept_rate"] * 100,
                        "accepted_tokens": metric["spec_accept_token_num"],
                        "draft_tokens": metric["spec_draft_token_num"],
                        "num_drafts": metric["spec_verify_ct"],
                    }
                except json.JSONDecodeError:
                    LOG.warning(f"Failed to parse JSON line {i} for metrics injection, skipping")
                    continue

        data_points = []
        with open(self.cfg.output_file, "rt", encoding="utf-8") as fin:
            for i, line in enumerate(fin):
                try:
                    data_points.append(json.loads(line))
                except json.JSONDecodeError:
                    LOG.warning(f"Failed to parse JSON line {i} for metrics injection, skipping")
                    continue

        with open(self.cfg.output_file, "wt", encoding="utf-8") as fout:
            for data_point in data_points:
                if all(response_id in metrics for response_id in data_point["response_ids"]):
                    data_point.update(
                        {
                            "num_drafts": sum(
                                metrics[rid]["num_drafts"] for rid in data_point["response_ids"]
                            ),
                            "draft_tokens": sum(
                                metrics[rid]["draft_tokens"] for rid in data_point["response_ids"]
                            ),
                            "accepted_tokens": sum(
                                metrics[rid]["accepted_tokens"] for rid in data_point["response_ids"]
                            ),
                            "acceptance_rate": sum(
                                metrics[rid]["acceptance_rate"] for rid in data_point["response_ids"]
                            ) / len(data_point["response_ids"]),
                            "acceptance_length": sum(
                                metrics[rid]["acceptance_length"] for rid in data_point["response_ids"]
                            ) / len(data_point["response_ids"]),
                        }
                    )
                else:
                    LOG.warning(
                        "No metrics found for response_ids: %s, skipping data point", data_point["response_ids"]
                    )
                fout.write(json.dumps(data_point) + "\n")

        try:
            return {
                "num_drafts": sum(dp["num_drafts"] for dp in data_points),
                "draft_tokens": sum(dp["draft_tokens"] for dp in data_points),
                "accepted_tokens": sum(dp["accepted_tokens"] for dp in data_points),
                "acceptance_rate": sum(dp["acceptance_rate"] for dp in data_points) / len(data_points),
                "acceptance_length": sum(dp["acceptance_length"] for dp in data_points) / len(data_points),
                "per_position_acceptance_rates": [],
            }
        except KeyError:
            LOG.warning("Metrics injection failed for some data points, skipping")
            return None

    async def process_single_datapoint(self, data_point, all_data, prompt_format="openai"):
        """Handle multi-turn generation and track request IDs for SGLang metrics."""
        messages = []
        responses = []

        for message in data_point["messages"]:
            messages.append(message)
            new_data_point = {"messages": messages}
            current_response = await super().process_single_datapoint(
                new_data_point, all_data, prompt_format=prompt_format
            )
            if "response" in current_response:
                raw_response = current_response.pop("response")
                current_response["response_id"] = raw_response.id
            responses.append(current_response)
            messages.append({"role": "assistant", "content": current_response["generation"]})

        return {
            "generation": [r["generation"] for r in responses],
            "num_generated_tokens": sum(r["num_generated_tokens"] for r in responses),
            "response_ids": [r["response_id"] for r in responses],
        }

    def run_batch_evaluation(self):
        """Fetch after-metrics, compute delta, inject into eval_config for the evaluator."""
        specdec_stats: dict[str, Any] | None = None

        if self.cfg.server["server_type"] == "sglang":
            # Per-request metrics file is preferred over the Prometheus delta.
            metrics_dir = getattr(self.cfg, "metrics_file_dir", None)
            if metrics_dir:
                metrics_file = find_sglang_metrics_file(metrics_dir)
                if metrics_file:
                    specdec_stats = self.inject_sglang_metrics(metrics_file)
                else:
                    LOG.warning("Could not find SGLang metrics file in %s", metrics_dir)

        if specdec_stats is None:
            specdec_stats = self._fetch_spec_decode_delta()

        self.cfg.eval_config["specdec_stats"] = specdec_stats
        super().run_batch_evaluation()


GENERATION_TASK_CLASS = SpeedbenchGenerationTask


@hydra.main(version_base=None, config_name="base_speedbench_generation_config")
def speedbench_generation(cfg: SpeedbenchGenerationConfig):
    cfg = SpeedbenchGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = SpeedbenchGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    SpeedbenchGenerationConfig,
    server_params=server_params(),
)

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        speedbench_generation()
