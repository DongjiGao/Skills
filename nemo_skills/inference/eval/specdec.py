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

import contextlib
import glob
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import requests

from nemo_skills.inference.generate import (
    GenerationTask,
    GenerationTaskConfig,
)
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


class SpecDecodeMetricsError(Exception):
    """Exception raised when fetching speculative decoding metrics fails."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


# ---------------------------------------------------------------------------
# Speculative decoding metrics from the server's Prometheus endpoint
# ---------------------------------------------------------------------------


@dataclass
class SpecDecodeMetrics:
    """Unified speculative decoding snapshot scraped from ``/metrics``."""

    # VLLM counters
    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    accepted_per_pos: dict[int, int] = field(default_factory=dict)

    # SGLang gauges and counters
    spec_accept_length: float = 0.0
    spec_accept_rate: float = 0.0
    num_requests: int = 0
    generation_tokens: int = 0


def _fetch_metrics_text(base_url: str) -> str | None:
    """Fetch raw Prometheus text from ``/metrics`` endpoint."""
    metrics_url = f"{base_url.rstrip('/')}/metrics"
    try:
        response = requests.get(metrics_url, timeout=30)
        if response.status_code != 200:
            LOG.warning("Metrics endpoint returned status %d", response.status_code)
            return None
        return response.text
    except requests.RequestException as exc:
        LOG.warning("Failed to fetch metrics from %s: %s", metrics_url, exc)
        return None


def fetch_vllm_spec_decode_metrics(base_url: str) -> SpecDecodeMetrics:
    """Fetch speculative decoding metrics from the server's ``/metrics`` endpoint.

    Parses Prometheus text-format exposition looking for VLLM
    ``vllm:spec_decode_*`` counters including per-position acceptance counts.

    Args:
        base_url: Server root URL, e.g. ``http://127.0.0.1:5000``.

    Returns:
        :class:`SpecDecodeMetrics` with the scraped counters.
    """
    text = _fetch_metrics_text(base_url)
    if text is None:
        message = "Failed to fetch metrics from the server"
        LOG.error(message)
        raise SpecDecodeMetricsError(message)

    metrics = SpecDecodeMetrics()
    found_spec_decode = False
    pos_label = 'position="'

    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("vllm:spec_decode"):
            continue

        found_spec_decode = True
        if "_created" in line:
            continue

        parts = line.split()
        if not parts:
            continue

        with contextlib.suppress(ValueError):
            metric_value = int(float(parts[-1]))
            if "num_drafts" in line:
                metrics.num_drafts += metric_value
            elif "num_draft_tokens" in line:
                metrics.num_draft_tokens += metric_value
            elif "num_accepted_tokens_per_pos" in line:
                if pos_label in line:
                    start = line.index(pos_label) + len(pos_label)
                    end = line.index('"', start)
                    pos = int(line[start:end])
                    metrics.accepted_per_pos[pos] = metrics.accepted_per_pos.get(pos, 0) + metric_value
            elif "num_accepted_tokens" in line:
                metrics.num_accepted_tokens += metric_value

    if not found_spec_decode:
        message = "No vllm:spec_decode_* metrics found on the server (speculative decoding may not be enabled)."
        LOG.error(message)
        raise SpecDecodeMetricsError(message)

    return metrics


def find_sglang_metrics_file(metrics_dir: str) -> str | None:
    """Find the most recent SGLang metrics file in the given directory.

    SGLang creates files like ``sglang-request-metrics-YYYYMMDD_HH.log``.

    Args:
        metrics_dir: Directory containing SGLang metrics files.

    Returns:
        Path to the most recent metrics file, or ``None`` if not found.
    """
    if not os.path.isdir(metrics_dir):
        LOG.warning("SGLang metrics directory does not exist: %s", metrics_dir)
        return None

    pattern = os.path.join(metrics_dir, "sglang-request-metrics-*.log")
    files = glob.glob(pattern)
    if not files:
        LOG.warning("No SGLang metrics files found matching pattern: %s", pattern)
        return None

    # Return the most recently modified file
    latest_file = max(files, key=os.path.getmtime)
    LOG.info("Using SGLang metrics file: %s", latest_file)
    return latest_file


def fetch_sglang_spec_decode_metrics(base_url: str) -> SpecDecodeMetrics:
    """Fetch speculative decoding metrics from an SGLang server's ``/metrics`` endpoint.

    Parses Prometheus text-format exposition looking for ``sglang:spec_accept_*``
    gauges and ``sglang:num_requests_total`` / ``sglang:generation_tokens_total``
    counters.

    Args:
        base_url: Server root URL, e.g. ``http://127.0.0.1:5000``.

    Returns:
        :class:`SpecDecodeMetrics` with the scraped values.
    """
    text = _fetch_metrics_text(base_url)
    if text is None:
        message = "Failed to fetch metrics from the server"
        LOG.error(message)
        raise SpecDecodeMetricsError(message)

    metrics = SpecDecodeMetrics()
    found_spec = False

    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        with contextlib.suppress(ValueError):
            if "sglang:spec_accept_length{" in line or line.startswith("sglang:spec_accept_length "):
                metrics.spec_accept_length = float(parts[-1])
                found_spec = True
            elif "sglang:spec_accept_rate{" in line or line.startswith("sglang:spec_accept_rate "):
                metrics.spec_accept_rate = float(parts[-1])
                found_spec = True
            elif "sglang:num_requests_total{" in line or line.startswith("sglang:num_requests_total "):
                metrics.num_requests = int(float(parts[-1]))
            elif "sglang:generation_tokens_total{" in line or line.startswith("sglang:generation_tokens_total "):
                metrics.generation_tokens = int(float(parts[-1]))

    if not found_spec:
        message = "No sglang:spec_accept_* metrics found on the server (speculative decoding may not be enabled)."
        LOG.error(message)
        raise SpecDecodeMetricsError(message)
    return metrics


def _build_specdec_stats(
    *,
    num_drafts: int,
    draft_tokens: int,
    accepted_tokens: int,
    acceptance_rate_fraction: float,
    acceptance_length: float,
    per_position_acceptance_rates: list[float] | None = None,
) -> dict[str, Any]:
    """Build a normalized spec-decode payload for evaluator injection."""
    return {
        "num_drafts": num_drafts,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted_tokens,
        "acceptance_rate": acceptance_rate_fraction * 100,
        "acceptance_length": acceptance_length,
        "per_position_acceptance_rates": per_position_acceptance_rates or [],
    }


def _compute_weighted_delta(
    *,
    before_avg: float,
    after_avg: float,
    before_count: int,
    after_count: int,
) -> float | None:
    """Compute benchmark-only average from cumulative weighted averages."""
    delta_count = after_count - before_count
    if delta_count <= 0:
        return None
    if before_count == 0:
        return after_avg
    weighted_after = after_avg * after_count
    weighted_before = before_avg * before_count
    return (weighted_after - weighted_before) / delta_count


def compute_sglang_spec_decode_delta(
    before: SpecDecodeMetrics,
    after: SpecDecodeMetrics,
) -> dict[str, Any] | None:
    """Compute benchmark-specific acceptance metrics from two SGLang snapshots.

    SGLang exposes ``spec_accept_length`` and ``spec_accept_rate`` as **gauges**
    (running averages from the server's perspective).  Combined with the
    ``num_requests_total`` and ``generation_tokens_total`` **counters** we can
    back out the benchmark-specific averages::

        weighted_al_after  = al_after  x n_after
        weighted_al_before = al_before x n_before
        benchmark_al = (weighted_al_after - weighted_al_before)
                       / (n_after - n_before)

    If the before snapshot had 0 requests (fresh server), the after values are
    returned as-is.

    Args:
        before: Snapshot taken before generation.
        after: Snapshot taken after generation.

    Returns:
        Dictionary with ``acceptance_length``, ``acceptance_rate``, counters,
        etc., or ``None`` if no requests were generated.
    """
    delta_requests = after.num_requests - before.num_requests
    delta_gen_tokens = after.generation_tokens - before.generation_tokens

    if delta_requests <= 0:
        LOG.warning(
            "SGLang: no new requests between before (%d) and after (%d) snapshots.",
            before.num_requests,
            after.num_requests,
        )
        return None

    acceptance_length = _compute_weighted_delta(
        before_avg=before.spec_accept_length,
        after_avg=after.spec_accept_length,
        before_count=before.num_requests,
        after_count=after.num_requests,
    )
    acceptance_rate_fraction = _compute_weighted_delta(
        before_avg=before.spec_accept_rate,
        after_avg=after.spec_accept_rate,
        before_count=before.num_requests,
        after_count=after.num_requests,
    )
    if acceptance_length is None or acceptance_rate_fraction is None:
        LOG.warning("SGLang: failed to compute weighted delta from request counters.")
        return None

    LOG.info(
        "SGLang Prometheus delta: requests=%d, gen_tokens=%d, acceptance_length=%.4f, acceptance_rate=%.2f%%",
        delta_requests,
        delta_gen_tokens,
        acceptance_length,
        acceptance_rate_fraction * 100,
    )

    return _build_specdec_stats(
        num_drafts=delta_requests,
        draft_tokens=delta_gen_tokens,
        accepted_tokens=int(delta_gen_tokens * acceptance_rate_fraction) if delta_gen_tokens > 0 else 0,
        acceptance_rate_fraction=acceptance_rate_fraction,
        acceptance_length=acceptance_length,
    )


def compute_vllm_spec_decode_delta(
    before: SpecDecodeMetrics,
    after: SpecDecodeMetrics,
) -> dict[str, Any] | None:
    """Compute the delta of speculative decoding counters between two snapshots.

    Uses the before/after pattern to isolate metrics to only the generation
    run (since Prometheus counters are cumulative).

    Computes:
        * **acceptance_rate** — ``delta_accepted / delta_draft_tokens * 100``
        * **acceptance_length** — ``1 + delta_accepted / delta_drafts``
          (the "+1" accounts for the target model's verified token that
          is always emitted even on a full rejection)
        * **per_position_acceptance_rates** — per-position acceptance
          probability, computed as ``delta_per_pos[i] / delta_drafts``

    Args:
        before: Counters snapshot taken before generation.
        after: Counters snapshot taken after generation.

    Returns:
        Dictionary with computed spec decode statistics, or ``None`` if the
        delta contains no meaningful data.
    """
    delta_drafts = after.num_drafts - before.num_drafts
    delta_draft_tokens = after.num_draft_tokens - before.num_draft_tokens
    delta_accepted = after.num_accepted_tokens - before.num_accepted_tokens

    # Compute per-position acceptance rates
    per_pos_rates: list[float] = []
    if delta_drafts > 0:
        positions = sorted(set(before.accepted_per_pos.keys()) | set(after.accepted_per_pos.keys()))
        for pos in positions:
            before_val = before.accepted_per_pos.get(pos, 0)
            after_val = after.accepted_per_pos.get(pos, 0)
            delta_pos = after_val - before_val
            per_pos_rates.append(delta_pos / delta_drafts)

    if delta_draft_tokens <= 0:
        LOG.warning(
            "No speculative decoding activity detected during generation "
            "(delta_draft_tokens=%d). Metrics will be empty.",
            delta_draft_tokens,
        )
        return None

    acceptance_rate_fraction = delta_accepted / delta_draft_tokens
    acceptance_length = 1 + delta_accepted / delta_drafts if delta_drafts > 0 else 0.0

    return _build_specdec_stats(
        num_drafts=delta_drafts,
        draft_tokens=delta_draft_tokens,
        accepted_tokens=delta_accepted,
        acceptance_rate_fraction=acceptance_rate_fraction,
        acceptance_length=acceptance_length,
        per_position_acceptance_rates=per_pos_rates,
    )


# ---------------------------------------------------------------------------
# Base speculative-decoding config & task
# ---------------------------------------------------------------------------


@nested_dataclass(kw_only=True)
class SpecdecGenerationConfig(GenerationTaskConfig):
    """Config for speculative-decoding evaluation with acceptance metrics.

    Generation is byte-for-byte identical to the standard pipeline — no prompt
    or output schema changes — so evaluation results (WER, BLEU, etc.) are
    directly comparable with a non-speculative baseline.  Works with any
    speculative method the server supports (MTP, EAGLE, ngram, draft model...).

    Restricts server type to those that expose Prometheus spec-decode counters.
    Subclasses can extend (e.g. SPEED-Bench adds an SGLang metrics dir).
    """

    def _post_init_validate_server(self):
        super()._post_init_validate_server()
        assert self.server["server_type"] in ("sglang", "vllm", "vllm_multimodal"), (
            f"server_type must be 'sglang', 'vllm', or 'vllm_multimodal' for "
            f"speculative-decoding generation, got '{self.server['server_type']}'"
        )


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_specdec_generation_config", node=SpecdecGenerationConfig)


class SpecdecGenerationTask(GenerationTask):
    """Standard generation task that additionally reports spec-decode acceptance stats.

    Captures Prometheus spec-decode counters from the server's ``/metrics``
    endpoint *before* generation starts (in ``wait_for_server``) and again in
    ``postprocess``; the delta isolates acceptance stats to the current run and
    is saved to ``specdec_metrics.json`` next to the output file.
    ``average acceptance length`` is ``1 + accepted_tokens / num_drafts``
    (the "+1" is the target model's verified token, emitted even on full rejection).

    Supports both vLLM (``vllm:spec_decode_*`` counters) and SGLang
    (``sglang:spec_accept_*`` gauges).
    """

    def __init__(self, cfg: SpecdecGenerationConfig):
        super().__init__(cfg)
        self._before_metrics: SpecDecodeMetrics | None = None

    def _get_server_base_address(self) -> str:
        return self.cfg.server.get("base_url") or f"http://{self.cfg.server['host']}:{self.cfg.server['port']}"

    def wait_for_server(self):
        """Wait for the server, then snapshot spec-decode counters (before)."""
        super().wait_for_server()
        server_type = self.cfg.server["server_type"]
        base_url = self._get_server_base_address()
        try:
            if server_type == "sglang":
                LOG.info("Fetching BEFORE SGLang spec-decode metrics from %s/metrics", base_url)
                self._before_metrics = fetch_sglang_spec_decode_metrics(base_url)
                LOG.info(
                    "SGLang before snapshot: accept_length=%.4f, accept_rate=%.4f, "
                    "num_requests=%d, gen_tokens=%d",
                    self._before_metrics.spec_accept_length,
                    self._before_metrics.spec_accept_rate,
                    self._before_metrics.num_requests,
                    self._before_metrics.generation_tokens,
                )
            else:
                LOG.info("Fetching BEFORE vLLM spec-decode metrics from %s/metrics", base_url)
                self._before_metrics = fetch_vllm_spec_decode_metrics(base_url)
                LOG.info(
                    "Before snapshot: drafts=%d, draft_tokens=%d, accepted=%d, per_pos_keys=%s",
                    self._before_metrics.num_drafts,
                    self._before_metrics.num_draft_tokens,
                    self._before_metrics.num_accepted_tokens,
                    sorted(self._before_metrics.accepted_per_pos.keys()),
                )
        except SpecDecodeMetricsError:
            LOG.warning("Spec-decode metrics unavailable (speculative decoding may not be enabled); skipping.")
            self._before_metrics = None

    def _fetch_spec_decode_delta(self) -> dict[str, Any] | None:
        """Fetch the after-snapshot and compute the delta vs the before-snapshot."""
        if self._before_metrics is None:
            return None
        base_url = self._get_server_base_address()
        server_type = self.cfg.server["server_type"]
        LOG.info("Fetching AFTER %s spec-decode metrics from %s/metrics", server_type, base_url)
        try:
            if server_type == "sglang":
                after_metrics = fetch_sglang_spec_decode_metrics(base_url)
                return compute_sglang_spec_decode_delta(self._before_metrics, after_metrics)
            after_metrics = fetch_vllm_spec_decode_metrics(base_url)
            return compute_vllm_spec_decode_delta(self._before_metrics, after_metrics)
        except SpecDecodeMetricsError:
            LOG.warning("Spec-decode metrics unavailable after generation; skipping.")
            return None

    def postprocess(self):
        super().postprocess()
        stats = self._fetch_spec_decode_delta()
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


GENERATION_TASK_CLASS = SpecdecGenerationTask


@hydra.main(version_base=None, config_name="base_specdec_generation_config")
def specdec_generation(cfg: SpecdecGenerationConfig):
    cfg = SpecdecGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = SpecdecGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    SpecdecGenerationConfig,
    server_params=server_params(),
)

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        specdec_generation()
