# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from nemo_skills.inference.generate import GenerationTask, GenerationTaskConfig, InferenceConfig


def test_subset_prompt_override_accepts_hydra_map_merge():
    cfg = OmegaConf.structured(
        GenerationTaskConfig(
            input_file="input.jsonl",
            output_file="output.jsonl",
            server={"server_type": "vllm_multimodal"},
            prompt_format="openai",
        )
    )
    overrides = {
        "spgispeech": {
            "system_message": "",
            "user_message": "SPGI cleanup prompt",
        }
    }

    OmegaConf.update(
        cfg,
        "prompt_overrides_by_subset",
        overrides,
        merge=True,
        force_add=True,
    )

    assert OmegaConf.to_container(cfg.prompt_overrides_by_subset) == overrides


@pytest.mark.parametrize(
    "row",
    [{}, {"subset_for_metrics": None}, {"subset_for_metrics": 7}],
)
def test_omegaconf_prompt_map_ignores_rows_without_string_subset(row):
    cfg = OmegaConf.structured(
        GenerationTaskConfig(
            input_file="input.jsonl",
            output_file="output.jsonl",
            server={"server_type": "vllm_multimodal"},
            prompt_format="openai",
            system_message="",
            user_message="benchmark fallback",
        )
    )
    OmegaConf.update(
        cfg,
        "prompt_overrides_by_subset",
        {"spgispeech": {"system_message": "", "user_message": "SPGI prompt"}},
        merge=True,
        force_add=True,
    )
    task = SimpleNamespace(cfg=cfg, prompt=None)

    assert GenerationTask._request_message_overrides(task, row) == (
        "",
        "benchmark fallback",
    )


class _CapturingLLM:
    def __init__(self):
        self.requests = []

    async def generate_async(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        return {"finish_reason": "stop", "generation": "ok", "num_generated_tokens": 1}


def _task(monkeypatch, tmp_path, **overrides):
    llm = _CapturingLLM()
    monkeypatch.setattr(GenerationTask, "setup_llm", lambda _self: llm)
    values = {
        "input_file": str(tmp_path / "input.jsonl"),
        "output_file": str(tmp_path / "output.jsonl"),
        "prompt_config": None,
        "prompt_format": "openai",
        "server": {"server_type": "vllm_multimodal"},
        "inference": InferenceConfig(
            tokens_to_generate=128,
            temperature=0,
            extra_body={"skip_special_tokens": False},
        ),
        "max_concurrent_requests": 2,
    }
    values.update(overrides)
    return GenerationTask(GenerationTaskConfig(**values)), llm


def _row(locale="de-DE", language="German", penalty=1.01):
    return {
        "messages": [
            {"role": "system", "content": "stale system"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "stale user"},
                    {"type": "audio_url", "audio_url": {"url": f"file:///data/{locale}.wav"}},
                ],
            },
        ],
        "locale": locale,
        "extra_fields": {"language_name": language},
        "decode": {
            "random_seed": 7,
            "repetition_penalty": penalty,
            "temperature": 0,
            "tokens_to_generate": 64,
            "top_k": -1,
            "top_logprobs": 0,
            "top_p": 1.0,
        },
    }


@pytest.mark.asyncio
async def test_real_generation_task_applies_row_prompts_and_decode_without_mutation(monkeypatch, tmp_path):
    task, llm = _task(
        monkeypatch,
        tmp_path,
        system_message_template="Recognition locale: {locale}",
        user_message_template="Transcribe in {language_name}. Do not translate.",
        inference_overrides_key="decode",
    )
    rows = [_row(), _row("fr-FR", "French", 1.05)]
    originals = deepcopy(rows)

    await task.process_single_datapoint(rows[0], rows)
    await task.process_single_datapoint(rows[1], rows)

    assert rows == originals
    first, second = llm.requests
    assert first["prompt"][0] == {"role": "system", "content": "Recognition locale: de-DE"}
    assert first["prompt"][1]["content"][0] == {
        "type": "text",
        "text": "Transcribe in German. Do not translate.",
    }
    assert first["prompt"][1]["content"][1] == originals[0]["messages"][1]["content"][1]
    assert second["prompt"][0]["content"] == "Recognition locale: fr-FR"
    assert second["prompt"][1]["content"][0]["text"] == "Transcribe in French. Do not translate."
    assert (first["repetition_penalty"], second["repetition_penalty"]) == (1.01, 1.05)
    assert first["tokens_to_generate"] == second["tokens_to_generate"] == 64
    assert first["extra_body"] == second["extra_body"] == {"skip_special_tokens": False}
    first["prompt"][1]["content"][0]["text"] = "mutated after capture"
    assert rows == originals


@pytest.mark.asyncio
async def test_subset_prompt_override_replaces_parent_prompt_only_for_matching_rows(
    monkeypatch, tmp_path
):
    task, llm = _task(
        monkeypatch,
        tmp_path,
        system_message="",
        user_message="parent prompt",
        prompt_overrides_by_subset={
            "spgispeech": {
                "system_message": "",
                "user_message": "SPGI cleanup prompt",
            }
        },
    )
    regular = _row()
    regular["subset_for_metrics"] = "ami"
    spgi = _row()
    spgi["subset_for_metrics"] = "spgispeech"

    await task.process_single_datapoint(regular, [regular, spgi])
    await task.process_single_datapoint(spgi, [regular, spgi])

    assert llm.requests[0]["prompt"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "parent prompt"},
                regular["messages"][1]["content"][1],
            ],
        }
    ]
    assert llm.requests[1]["prompt"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "SPGI cleanup prompt"},
                spgi["messages"][1]["content"][1],
            ],
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("row_change", "message"),
    [
        (lambda row: row.pop("decode"), "missing inference overrides field 'decode'"),
        (lambda row: row.__setitem__("decode", []), "must be a mapping"),
    ],
)
async def test_request_inference_mapping_fails_closed(monkeypatch, tmp_path, row_change, message):
    task, llm = _task(monkeypatch, tmp_path, inference_overrides_key="decode")
    row = _row()
    row_change(row)
    error_type = ValueError if "missing" in message else TypeError
    with pytest.raises(error_type, match=message):
        await task.process_single_datapoint(row, [row])
    assert llm.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "override",
    [
        {"endpoint_type": "chat"},
        {"extra_body": {"_speechlm_router": {"read_timeout_s": 1}}},
        {"response_format": {"type": "json_object"}},
        {"stream": True},
        {"timeout": 1},
        {"tools": []},
        {"_speechlm_router": {"read_timeout_s": 1}},
        {"_speechlm_router.read_timeout_s": 1},
    ],
)
async def test_request_inference_rejects_router_and_control_fields(monkeypatch, tmp_path, override):
    task, llm = _task(monkeypatch, tmp_path, inference_overrides_key="decode")
    row = _row()
    row["decode"] = override
    with pytest.raises(ValueError, match="router/transport/control field"):
        await task.process_single_datapoint(row, [row])
    assert llm.requests == []


@pytest.mark.asyncio
async def test_request_inference_rejects_non_allowlisted_field(monkeypatch, tmp_path):
    task, llm = _task(monkeypatch, tmp_path, inference_overrides_key="decode")
    row = _row()
    row["decode"] = {"reasoning_effort": "high"}
    with pytest.raises(ValueError, match="Unsupported per-request inference override"):
        await task.process_single_datapoint(row, [row])
    assert llm.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("tokens_to_generate", 0, "positive"),
        ("tokens_to_generate", True, "integer"),
        ("temperature", float("nan"), "finite"),
        ("temperature", 2.1, "in \\[0, 2\\]"),
        ("top_p", 0, "in \\(0, 1\\]"),
        ("min_p", -0.1, "in \\[0, 1\\]"),
        ("repetition_penalty", 0, "positive"),
        ("top_k", 0, "-1 or positive"),
        ("top_logprobs", 21, "in \\[0, 20\\]"),
        ("random_seed", -1, "non-negative"),
    ],
)
async def test_request_inference_override_values_fail_closed(monkeypatch, tmp_path, name, value, message):
    task, llm = _task(monkeypatch, tmp_path, inference_overrides_key="decode")
    row = _row()
    row["decode"] = {name: value}
    with pytest.raises(ValueError, match=message):
        await task.process_single_datapoint(row, [row])
    assert llm.requests == []


def test_request_override_config_and_template_errors_are_strict(monkeypatch, tmp_path):
    base = {
        "input_file": str(tmp_path / "input.jsonl"),
        "output_file": str(tmp_path / "output.jsonl"),
        "prompt_format": "openai",
        "prompt_config": None,
        "server": {"server_type": "vllm_multimodal"},
    }
    with pytest.raises(ValueError, match="Only one of user_message and user_message_template"):
        GenerationTaskConfig(**base, user_message="static", user_message_template="{language}")
    with pytest.raises(ValueError, match="only simple"):
        GenerationTaskConfig(**base, user_message_template="{extra_fields[language_name]}")

    task, llm = _task(monkeypatch, tmp_path, user_message_template="Transcribe in {missing_language}")
    with pytest.raises(ValueError, match="missing field 'missing_language'"):
        task.fill_prompt(_row(), [])
    assert llm.requests == []

    task, _llm = _task(monkeypatch, tmp_path, user_message_template="Transcribe {language_name}")
    row = _row()
    row["language_name"] = "Deutsch"
    with pytest.raises(ValueError, match="ambiguous"):
        task.fill_prompt(row, [row])
