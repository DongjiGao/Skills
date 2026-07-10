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

"""Generation module with vLLM MTP speculative-decoding metrics.

Generation, prompting, and evaluation are byte-for-byte identical to the
standard pipeline (``nemo_skills.inference.generate``): the prompt is the
``messages`` field of each data point, one request per data point, and the
evaluator runs interleaved per sample.  The single addition is that vLLM's
cumulative ``vllm:spec_decode_*`` Prometheus counters are snapshotted before
and after generation, and the delta is reported as acceptance rate and average
acceptance length, saved next to the output file as ``specdec_metrics.json``.

Unlike ``specdec.py`` (SPEED-Bench), this module never alters prompts,
responses, or the output schema, so evaluation results are directly comparable
with a non-speculative baseline run.  Works for any task (ASR, ST, text, etc.)
as long as the vLLM server has speculative decoding enabled.
"""

import json
import logging
import sys
from pathlib import Path

import hydra

from nemo_skills.inference.eval.specdec import (
    SpecDecodeMetricsError,
    compute_vllm_spec_decode_delta,
    fetch_vllm_spec_decode_metrics,
)
from nemo_skills.inference.generate import GenerationTask, GenerationTaskConfig
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class MTPGenerationConfig(GenerationTaskConfig):
    """Standard generation config restricted to vLLM servers.

    The acceptance metrics come from the server's global Prometheus
    ``/metrics`` endpoint, which only vLLM exposes in the expected format.
    """

    def _post_init_validate_server(self):
        super()._post_init_validate_server()
        assert self.server["server_type"] in (
            "vllm",
            "vllm_multimodal",
        ), (
            f"server_type must be 'vllm' or 'vllm_multimodal' for MTP "
            f"generation, got '{self.server['server_type']}'"
        )


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_mtp_generation_config", node=MTPGenerationConfig)


class MTPGenerationTask(GenerationTask):
    """Standard generation task that additionally reports MTP acceptance stats.

    Since Prometheus counters are cumulative, a before/after snapshot pair
    isolates the delta to this generation run.  ``average acceptance length``
    is ``1 + accepted_tokens / num_drafts`` (the "+1" is the target model's
    verified token, emitted even on a full rejection).
    """

    def __init__(self, cfg: MTPGenerationConfig):
        super().__init__(cfg)
        self._before_metrics = None

    def _server_base_address(self) -> str:
        return self.cfg.server.get("base_url") or f"http://{self.cfg.server['host']}:{self.cfg.server['port']}"

    def wait_for_server(self):
        super().wait_for_server()
        base_url = self._server_base_address()
        LOG.info("Fetching BEFORE vLLM spec-decode metrics from %s/metrics", base_url)
        try:
            self._before_metrics = fetch_vllm_spec_decode_metrics(base_url)
            LOG.info(
                "Before snapshot: drafts=%d, draft_tokens=%d, accepted=%d",
                self._before_metrics.num_drafts,
                self._before_metrics.num_draft_tokens,
                self._before_metrics.num_accepted_tokens,
            )
        except SpecDecodeMetricsError:
            LOG.warning("Spec-decode metrics unavailable (speculative decoding not enabled); skipping.")
            self._before_metrics = None

    def postprocess(self):
        super().postprocess()
        if self._before_metrics is None:
            return
        base_url = self._server_base_address()
        LOG.info("Fetching AFTER vLLM spec-decode metrics from %s/metrics", base_url)
        try:
            after_metrics = fetch_vllm_spec_decode_metrics(base_url)
        except SpecDecodeMetricsError:
            LOG.warning("Spec-decode metrics unavailable after generation; skipping.")
            return
        stats = compute_vllm_spec_decode_delta(self._before_metrics, after_metrics)
        if stats is None:
            return
        LOG.info(
            "Speculative decoding stats for this run: average acceptance_length=%.4f, "
            "acceptance_rate=%.2f%%, drafts=%d, draft_tokens=%d, accepted_tokens=%d",
            stats["acceptance_length"],
            stats["acceptance_rate"],
            stats["num_drafts"],
            stats["draft_tokens"],
            stats["accepted_tokens"],
        )
        if stats["per_position_acceptance_rates"]:
            LOG.info(
                "Per-position acceptance rates: %s",
                [f"{r:.4f}" for r in stats["per_position_acceptance_rates"]],
            )
        metrics_path = Path(self.cfg.output_file).parent / "specdec_metrics.json"
        metrics_path.write_text(json.dumps(stats, indent=2))
        LOG.info("Saved spec-decode metrics to %s", metrics_path)


GENERATION_TASK_CLASS = MTPGenerationTask


@hydra.main(version_base=None, config_name="base_mtp_generation_config")
def mtp_generation(cfg: MTPGenerationConfig):
    cfg = MTPGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = MTPGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    MTPGenerationConfig,
    server_params=server_params(),
)

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        mtp_generation()
