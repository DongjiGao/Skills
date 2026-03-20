# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""vLLM NeMo Speech LM backend for unified_server.

Performs supported speech-to-text tasks such as speech recognition,
spoken question answering, speech and audio understanding, etc. using
vLLM with NeMo Speech LM multimodal models (e.g. Nemotron-Nano-v3 +
Canary-v2).  Uses vLLM's optimized inference engine with PagedAttention
and continuous batching.  Supports ``tensor_parallel_size`` for
multi-GPU sharding.

Prerequisites
-------------
This backend requires NeMo Speech (``nemo_toolkit[all]``) and a vLLM
plugin that registers NeMo Speech LM models into vLLM's model registry
via the ``vllm.general_plugins`` entry point.  The plugin is activated
at runtime by setting ``VLLM_PLUGINS`` (handled automatically by this
backend).

A compatible model checkpoint is also required.  The checkpoint
directory must contain ``config.json`` with the appropriate
``model_type`` and ``architectures`` fields, plus model weights.

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
        --server_args "--backend vllm_nemo_speechlm --batch_size 32 \\
            --tokenizer nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \\
            --gpu_memory_utilization 0.90" \\
        ++max_concurrent_requests=32 \\
        ++inference.tokens_to_generate=512

Results (32 GPUs, Nemotron-Nano-v3 + Canary-v2, A100-80GB).
WER (%) / RTFx for each batch size.  WER computed at corpus level
with jiwer + whisper EnglishTextNormalizer (hf_leaderboard mode),
same method as NeMo checkpoint evaluation:

    =================  ========  ===========  ===========  ===========
    Dataset            NeMo       BS=8         BS=16        BS=32
    =================  ========  ===========  ===========  ===========
    librispeech_clean   1.91%    1.91%/  48x  1.93%/ 107x  1.92%/ 227x
    librispeech_other   3.53%    3.53%/  41x  3.53%/  92x  3.51%/ 197x
    tedlium             3.70%    3.64%/  64x  3.64%/ 136x  3.64%/ 277x
    spgispeech          2.20%    2.21%/  12x  2.22%/  36x  2.21%/  98x
    voxpopuli           6.29%    6.32%/  67x  6.28%/ 147x  6.27%/ 306x
    gigaspeech          9.95%    9.95%/  14x  9.97%/  38x  9.96%/  98x
    earnings22         10.93%   11.03%/  46x 10.98%/ 102x 11.01%/ 215x
    ami                11.65%   11.62%/  12x 11.74%/  29x 11.65%/  66x
    =================  ========  ===========  ===========  ===========

NeMo WER recomputed with identical method matches the reported
checkpoint summary exactly.  vLLM matches NeMo within 0.1% on all
datasets; the 0.06% tedlium delta traces to one utterance where the
NeMo run produced an empty output (inference non-determinism).
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
class VLLMNeMoSpeechLMConfig(BackendConfig):
    """Configuration for vLLM NeMo Speech LM backend."""

    tokenizer: Optional[str] = None
    hf_overrides: Optional[Dict[str, Any]] = None
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096
    enforce_eager: bool = True
    block_size: int = 64
    prompt: str = "Transcribe the following:"
    sampling_max_tokens: int = 512
    sampling_temperature: float = 0.0
    tensor_parallel_size: int = 1

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VLLMNeMoSpeechLMConfig":
        """Create config from a dict, separating known fields from extra_config."""
        known = {f.name for f in cls.__dataclass_fields__.values()} - {"extra_config"}
        return cls(
            **{k: v for k, v in d.items() if k in known},
            extra_config={k: v for k, v in d.items() if k not in known},
        )


class VLLMNeMoSpeechLMBackend(InferenceBackend):
    """Unified-server backend using vLLM with NeMo Speech LM multimodal plugin."""

    @classmethod
    def get_config_class(cls) -> type:
        """Return the config class for this backend."""
        return VLLMNeMoSpeechLMConfig

    @property
    def name(self) -> str:
        """Backend identifier used in --backend flag."""
        return "vllm_nemo_speechlm"

    @property
    def supported_modalities(self) -> Set[Modality]:
        """This backend accepts audio input and produces text output."""
        return {Modality.AUDIO_IN, Modality.TEXT}

    def __init__(self, config: BackendConfig):
        """Initialize backend, converting generic config to VLLMNeMoSpeechLMConfig if needed."""
        self.vllm_config = (
            config
            if isinstance(config, VLLMNeMoSpeechLMConfig)
            else VLLMNeMoSpeechLMConfig.from_dict(
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

    _VLLM_PLUGIN = "nemo_speechlm"

    def load_model(self) -> None:
        """Load the vLLM model, configure sampling params, and build prompt template."""
        os.environ["VLLM_PLUGINS"] = self._VLLM_PLUGIN
        from vllm import LLM, SamplingParams

        hf_overrides = self.vllm_config.hf_overrides or {
            "architectures": ["NeMoSpeechLMForConditionalGeneration"],
            "model_type": "nemo_speechlm",
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

        tokenizer = self._llm.get_tokenizer()
        hf_config = self._llm.llm_engine.model_config.hf_config
        audio_tag = getattr(hf_config, "audio_locator_tag", "<|audio|>")
        prompt_with_audio = f"{self.vllm_config.prompt} {audio_tag}"
        messages = [{"role": "user", "content": prompt_with_audio}]
        self._prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        logger.info("Prompt template: %s", repr(self._prompt_text))

        self._is_loaded = True
        logger.info("vLLM NeMo Speech LM model loaded")

    def _audio_bytes_to_numpy(self, audio_bytes: bytes) -> tuple:
        """Convert raw audio bytes to a numpy array and sample rate."""
        import numpy as np  # noqa: F401
        import soundfile as sf

        audio_arr, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if audio_arr.ndim > 1:
            audio_arr = audio_arr.mean(axis=1)
        return audio_arr, sr

    def _get_request_audio(self, request: GenerationRequest) -> bytes:
        """Extract audio bytes from a request (supports audio_bytes or audio_bytes_list)."""
        if request.audio_bytes:
            return request.audio_bytes
        if request.audio_bytes_list:
            if len(request.audio_bytes_list) > 1:
                raise ValueError("vllm_nemo_speechlm backend currently supports one audio per request.")
            return request.audio_bytes_list[0]
        raise ValueError("Request must contain audio_bytes or audio_bytes_list")

    def _strip_think_tags(self, text: str) -> str:
        """Remove <think>...</think> tags from NemotronH model output."""
        return THINK_TAG_PATTERN.sub("", text).strip()

    def validate_request(self, request: GenerationRequest) -> Optional[str]:
        """Validate request has audio input. Logs warning for ignored per-request overrides."""
        has_audio = request.audio_bytes is not None or (
            request.audio_bytes_list is not None and len(request.audio_bytes_list) > 0
        )
        if not has_audio:
            return "vllm_nemo_speechlm backend requires audio input"
        ignored = {
            "max_new_tokens": request.max_new_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "seed": request.seed,
        }
        set_fields = [k for k, v in ignored.items() if v is not None]
        if set_fields:
            logger.warning("Ignoring per-request overrides (using backend defaults): %s", ", ".join(set_fields))
        return None

    def generate(self, requests: List[GenerationRequest]) -> List[GenerationResult]:
        """Generate transcriptions for a batch of audio requests via vLLM."""
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
                        "prompt": self._prompt_text,
                        "multi_modal_data": {"audio": (audio_arr, sr)},
                    }
                )
                valid_indices.append(idx)
            except (ValueError, TypeError, OSError) as e:
                results[idx] = GenerationResult(error=str(e), request_id=req.request_id)

        if vllm_inputs:
            outputs = self._llm.generate(vllm_inputs, self._sampling_params, use_tqdm=False)
            elapsed_ms = (time.time() - start) * 1000.0

            for out_idx, output in enumerate(outputs):
                req_idx = valid_indices[out_idx]
                text = self._strip_think_tags(output.outputs[0].text)
                results[req_idx] = GenerationResult(
                    text=text,
                    request_id=requests[req_idx].request_id,
                    generation_time_ms=elapsed_ms,
                    debug_info={
                        "backend": "vllm_nemo_speechlm",
                        "model": self.vllm_config.model_path,
                        "batch_size": len(vllm_inputs),
                    },
                )

        return [r if r is not None else GenerationResult(error="Unknown vLLM ASR error") for r in results]
