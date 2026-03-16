# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""vLLM ASR backend for unified_server.

Performs speech recognition using vLLM with multimodal audio support
(e.g. Nemotron-Nano-v3 + Canary-v2 ASR via the ``nemotron_nano_asr`` plugin).
Uses vLLM's optimized inference engine with PagedAttention and continuous
batching.  Supports ``tensor_parallel_size`` for multi-GPU sharding.

Prerequisites
-------------
This backend requires the ``nemotron_nano_asr`` vLLM plugin to be
installed in the environment.  The plugin registers the multimodal model
class into vLLM's model registry via the ``vllm.general_plugins``
entry point::

    pip install git+https://github.com/DongjiGao/nemotron_nano_asr.git

The plugin is activated at runtime by setting ``VLLM_PLUGINS=nemotron_nano_asr``
(handled automatically by this backend via the ``vllm_plugins`` config field).

A compatible model checkpoint is also required.  The checkpoint directory
must contain ``config.json`` with ``"model_type": "nemotron_nano_asr"`` and
``"architectures": ["NemotronNanoASRForConditionalGeneration"]``, plus
``model.safetensors`` weights.

Multi-GPU scaling
-----------------
For horizontal scaling, use ``ns generate --num_chunks N`` to launch N
independent single-GPU jobs in parallel (each running its own vLLM
instance).  This is the recommended approach for the Nemotron-Nano MoE
architecture, as DP/EP introduce synchronization overhead that negates
parallelism gains.

Example (8 GPUs, batch_size=32)::

    ns generate \\
        --num_chunks 8 \\
        --server_gpus 1 \\
        --server_type generic \\
        --server_args "--backend vllm_asr --batch_size 32 \\
            --tokenizer nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \\
            --gpu_memory_utilization 0.90" \\
        ++max_concurrent_requests=32 \\
        ++inference.tokens_to_generate=512

Throughput (32 GPUs, Nemotron-Nano-v3 + Canary-v2, A100-80GB):

    ================  ======  =======  =======
    Dataset           BS=8    BS=16    BS=32
    ================  ======  =======  =======
    librispeech_clean  6 u/s  14 u/s   31 u/s
    librispeech_other  6 u/s  14 u/s   30 u/s
    tedlium            8 u/s  17 u/s   34 u/s
    voxpopuli          7 u/s  15 u/s   32 u/s
    earnings22         6 u/s  14 u/s   30 u/s
    ami                4 u/s  11 u/s   25 u/s
    gigaspeech         2 u/s   6 u/s   15 u/s
    spgispeech         1 u/s   4 u/s   11 u/s
    ================  ======  =======  =======

WER matches NeMo checkpoint evaluation within 0.1% across all datasets
and batch sizes (corpus-level WER computed with jiwer).
"""

from __future__ import annotations

import io
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from .base import BackendConfig, GenerationRequest, GenerationResult, InferenceBackend, Modality

logger = logging.getLogger(__name__)

THINK_TAG_PATTERN = re.compile(r"^<think>.*?</think>", re.DOTALL)


@dataclass
class VLLMASRConfig(BackendConfig):
    """Configuration for vLLM ASR backend."""

    tokenizer: Optional[str] = None
    hf_overrides: Optional[Dict[str, Any]] = None
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096
    enforce_eager: bool = True
    block_size: int = 64
    prompt_template: str = (
        "<|im_start|>system\n<|im_end|>\n"
        "<|im_start|>user\n"
        "Transcribe the following: <|audio|>"
        "<|im_end|>\n<|im_start|>assistant\n"
        "<think>\n"
    )
    vllm_plugins: str = "nemotron_nano_asr"
    sampling_max_tokens: int = 256
    sampling_temperature: float = 0.0
    tensor_parallel_size: int = 1

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VLLMASRConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()} - {"extra_config"}
        return cls(
            **{k: v for k, v in d.items() if k in known},
            extra_config={k: v for k, v in d.items() if k not in known},
        )


class VLLMASRBackend(InferenceBackend):
    """Unified-server backend for ASR using vLLM with multimodal plugin."""

    @classmethod
    def get_config_class(cls) -> type:
        return VLLMASRConfig

    @property
    def name(self) -> str:
        return "vllm_asr"

    @property
    def supported_modalities(self) -> Set[Modality]:
        return {Modality.AUDIO_IN, Modality.TEXT}

    def __init__(self, config: BackendConfig):
        self.vllm_config = (
            config
            if isinstance(config, VLLMASRConfig)
            else VLLMASRConfig.from_dict(
                {
                    "model_path": config.model_path,
                    "device": config.device,
                    "dtype": config.dtype,
                    **config.extra_config,
                }
            )
        )
        super().__init__(self.vllm_config)
        self._llm = None
        self._sampling_params = None

    def load_model(self) -> None:
        os.environ["VLLM_PLUGINS"] = self.vllm_config.vllm_plugins
        from vllm import LLM, SamplingParams

        hf_overrides = self.vllm_config.hf_overrides or {
            "architectures": ["NemotronNanoASRForConditionalGeneration"],
            "model_type": "nemotron_nano_asr",
        }

        logger.info("Loading vLLM ASR model: %s", self.vllm_config.model_path)

        llm_kwargs = dict(
            model=self.vllm_config.model_path,
            hf_overrides=hf_overrides,
            trust_remote_code=True,
            dtype=self.vllm_config.dtype,
            gpu_memory_utilization=self.vllm_config.gpu_memory_utilization,
            enforce_eager=self.vllm_config.enforce_eager,
            max_model_len=self.vllm_config.max_model_len,
            block_size=self.vllm_config.block_size,
            limit_mm_per_prompt={"audio": 1},
        )
        if self.vllm_config.tokenizer:
            llm_kwargs["tokenizer"] = self.vllm_config.tokenizer
        if self.vllm_config.tensor_parallel_size > 1:
            llm_kwargs["tensor_parallel_size"] = self.vllm_config.tensor_parallel_size

        self._llm = LLM(**llm_kwargs)
        self._sampling_params = SamplingParams(
            max_tokens=self.vllm_config.sampling_max_tokens,
            temperature=self.vllm_config.sampling_temperature,
        )
        self._is_loaded = True
        logger.info("vLLM ASR model loaded")

    def _audio_bytes_to_numpy(self, audio_bytes: bytes) -> tuple:
        import numpy as np  # noqa: F401
        import soundfile as sf

        audio_arr, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if audio_arr.ndim > 1:
            audio_arr = audio_arr.mean(axis=1)
        return audio_arr, sr

    def _get_request_audio(self, request: GenerationRequest) -> bytes:
        if request.audio_bytes:
            return request.audio_bytes
        if request.audio_bytes_list:
            if len(request.audio_bytes_list) > 1:
                raise ValueError("vllm_asr backend currently supports one audio per request.")
            return request.audio_bytes_list[0]
        raise ValueError("Request must contain audio_bytes or audio_bytes_list")

    def _strip_think_tags(self, text: str) -> str:
        return THINK_TAG_PATTERN.sub("", text).strip()

    def validate_request(self, request: GenerationRequest) -> Optional[str]:
        has_audio = request.audio_bytes is not None or (
            request.audio_bytes_list is not None and len(request.audio_bytes_list) > 0
        )
        if not has_audio:
            return "vllm_asr backend requires audio input"
        unsupported = {
            "max_new_tokens": request.max_new_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "seed": request.seed,
        }
        set_fields = [k for k, v in unsupported.items() if v is not None]
        if set_fields:
            return f"vllm_asr backend does not support per-request overrides: {', '.join(set_fields)}"
        return None

    def generate(self, requests: List[GenerationRequest]) -> List[GenerationResult]:
        if not self._is_loaded:
            return [GenerationResult(error="Model not loaded", request_id=r.request_id) for r in requests]
        if not requests:
            return []

        start = time.time()
        vllm_inputs = []
        valid_indices = []
        results: List[Optional[GenerationResult]] = [None] * len(requests)

        for idx, req in enumerate(requests):
            try:
                audio_bytes = self._get_request_audio(req)
                audio_arr, sr = self._audio_bytes_to_numpy(audio_bytes)
                vllm_inputs.append(
                    {
                        "prompt": self.vllm_config.prompt_template,
                        "multi_modal_data": {"audio": (audio_arr, sr)},
                    }
                )
                valid_indices.append(idx)
            except (ValueError, TypeError, OSError) as e:
                results[idx] = GenerationResult(error=str(e), request_id=req.request_id)

        if vllm_inputs:
            outputs = self._llm.generate(vllm_inputs, self._sampling_params, use_tqdm=False)
            elapsed_ms = (time.time() - start) * 1000.0
            per_req_ms = elapsed_ms / max(len(vllm_inputs), 1)

            for out_idx, output in enumerate(outputs):
                req_idx = valid_indices[out_idx]
                text = self._strip_think_tags(output.outputs[0].text)
                results[req_idx] = GenerationResult(
                    text=text,
                    request_id=requests[req_idx].request_id,
                    generation_time_ms=per_req_ms,
                    debug_info={
                        "backend": "vllm_asr",
                        "model": self.vllm_config.model_path,
                        "batch_size": len(vllm_inputs),
                    },
                )

        return [r if r is not None else GenerationResult(error="Unknown vLLM ASR error") for r in results]
