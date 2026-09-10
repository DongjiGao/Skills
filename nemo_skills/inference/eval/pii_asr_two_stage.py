# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Two-stage PII-ASR generation: model ASR followed by model redaction."""

import logging
import sys
from dataclasses import field

import hydra

from nemo_skills.code_execution.sandbox import sandbox_params
from nemo_skills.inference.generate import GenerationTask, GenerationTaskConfig
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class PIIASRTwoStageGenerationConfig(GenerationTaskConfig):
    prompt_format: str = "openai"
    server: dict = field(default_factory=dict)


hydra.core.config_store.ConfigStore.instance().store(
    name="base_pii_asr_two_stage_generation_config", node=PIIASRTwoStageGenerationConfig
)


class PIIASRTwoStageGenerationTask(GenerationTask):
    async def process_single_datapoint(self, data_point, all_data):
        first = await super().process_single_datapoint(data_point, all_data)
        first_generation = str(first.get(self.cfg.generation_key) or "").strip()
        messages = [
            *self.fill_prompt(data_point, all_data),
            {"role": "assistant", "content": first_generation},
            {"role": "user", "content": data_point["redaction_prompt"]},
        ]
        second = await super().process_single_datapoint({"messages": messages}, all_data)
        second["intermediate_asr"] = first_generation
        second["first_stage_finish_reason"] = first.get("finish_reason")
        second["second_stage_finish_reason"] = second.get("finish_reason")
        second["num_input_tokens_turn1"] = int(first.get("num_input_tokens") or 0)
        second["num_input_tokens_turn2"] = int(second.get("num_input_tokens") or 0)
        second["num_generated_tokens_turn1"] = int(first.get("num_generated_tokens") or 0)
        second["num_generated_tokens_turn2"] = int(second.get("num_generated_tokens") or 0)
        second["num_input_tokens"] = second["num_input_tokens_turn1"] + second["num_input_tokens_turn2"]
        second["num_generated_tokens"] = second["num_generated_tokens_turn1"] + second["num_generated_tokens_turn2"]
        return second


GENERATION_TASK_CLASS = PIIASRTwoStageGenerationTask


@hydra.main(version_base=None, config_name="base_pii_asr_two_stage_generation_config")
def generate(cfg: PIIASRTwoStageGenerationConfig):
    cfg = PIIASRTwoStageGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)
    PIIASRTwoStageGenerationTask(cfg).generate()


HELP_MESSAGE = get_help_message(
    PIIASRTwoStageGenerationConfig,
    server_params=server_params(),
    sandbox_params=sandbox_params(),
)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        generate()
