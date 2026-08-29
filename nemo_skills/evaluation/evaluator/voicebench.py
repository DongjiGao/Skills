# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Pinned VoiceBench evaluation, with random fallback isolated as parity only."""

import asyncio
import hashlib
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from nemo_skills.evaluation.evaluator.base import BaseEvaluator, BaseEvaluatorConfig
from nemo_skills.evaluation.evaluator.voicebench_bbh_official import BBHEvaluator
from nemo_skills.evaluation.evaluator.voicebench_harm_official import HarmEvaluator
from nemo_skills.evaluation.evaluator.voicebench_mcq_official import MCQEvaluator
from nemo_skills.utils import nested_dataclass

OFFICIAL_CODE_REVISION = "3c3b0d3a7a956f745305eb348f5e03ce7ec73dad"
PEDANT_SOURCE_REVISION = "26623c6ca3313e7a58c48fd6ce2b8579eb0c742e"
PEDANT_ARTIFACT_SHA256 = {
    "lr_classifier.pkl": "9c3dd5998745ece1cfb6cb16b953358f794e8716d63a08f3b212d8bc32a59b8b",
    "tf-idf_vectorizer.pkl": "e886c9ba3ddb15a4b693b075dfe19cfad36dc74e7f8f195b8d9e98e989f2ba4f",
}
JUDGE_MODEL = "gpt-4o-mini"
JUDGE_SYSTEM_PROMPT = "You are a helpful assistant who tries to help answer the user's question."
JUDGE_REQUEST_KWARGS = {
    "model": JUDGE_MODEL,
    "max_tokens": 1024,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "stop": None,
    "temperature": 0.5,
    "top_p": 0.95,
    "n": 3,
}

OPEN_JUDGE_TEMPLATE = """
I need your help to evaluate the performance of several models in the speech interaction scenario. The models will receive a speech input from the user, which they need to understand and respond to with a speech output.
Your task is to rate the model’s responses based on the provided user input transcription [Instruction] and the model’s output transcription [Response].

Please evaluate the response on a scale of 1 to 5:
1 point: The response is largely irrelevant, incorrect, or fails to address the user’s query. It may be off-topic or provide incorrect information.
2 points: The response is somewhat relevant but lacks accuracy or completeness. It may only partially answer the user’s question or include extraneous information.
3 points: The response is relevant and mostly accurate, but it may lack conciseness or include unnecessary details that don’t contribute to the main point.
4 points: The response is relevant, accurate, and concise, providing a clear answer to the user’s question without unnecessary elaboration.
5 points: The response is exceptionally relevant, accurate, and to the point. It directly addresses the user’s query in a highly effective and efficient manner, providing exactly the information needed.

Below are the transcription of user’s instruction and models’ response:
### [Instruction]: {prompt}
### [Response]: {response}

After evaluating, please output the score only without anything else.
You don’t need to provide any explanations.
"""

QA_JUDGE_TEMPLATE = """
### Question
{prompt}

### Reference answer
{reference}

### Candidate answer
{response}

Is the candidate answer correct based on the question and reference answer?\x20
Please only output a single "Yes" or "No". Do not output anything else.
""".strip()

OPEN_TASKS = frozenset(
    {"alpacaeval", "alpacaeval_full", "alpacaeval_speaker", "commoneval", "wildvoice"}
)
MCQ_TASKS = frozenset({"openbookqa", "mmsu"})
BBH_GROUND_TRUTH = {"yes": 1, "no": 0, "(a)": 0, "(b)": 1}


def extract_open_rating(output: str) -> float:
    """Official float-first parser, including its ``[[number]]`` fallback."""
    try:
        return float(output)
    except Exception:  # noqa: BLE001 - exact official float-or-regex behavior
        match = re.search(r"\[\[(\d+)\]\]", output)
        if not match:
            raise ValueError(f"VoiceBench open judge output is not a rating: {output!r}")
        return float(match.group(1))


def official_majority_vote(outputs: list[str]) -> bool:
    """Exact official SD-QA GPT majority semantics."""
    lowered = [item.lower() for item in outputs]
    final_answer = max(set(lowered), key=lowered.count)
    return final_answer == "yes"


def judge_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pinned_pedant(pedant_module=None):
    """Load PEDANT without its unconditional mutable-master network refresh."""
    if pedant_module is None:
        try:
            import qa_metrics.pedant as pedant_module
        except ImportError as exc:
            raise RuntimeError("Official SD-QA PEDANT scoring requires qa_metrics==0.2.17") from exc

    artifact_dir = Path(pedant_module.__file__).resolve().parent / "classifier"
    for filename, expected_hash in PEDANT_ARTIFACT_SHA256.items():
        artifact = artifact_dir / filename
        if not artifact.is_file():
            raise RuntimeError(
                f"Pinned PEDANT artifact is missing: {artifact}. Use the validated NeMo Skills eval image."
            )
        actual_hash = _sha256(artifact)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Pinned PEDANT artifact hash mismatch for {artifact}: expected {expected_hash}, got {actual_hash}"
            )

    original_download = pedant_module.download_link
    pedant_module.download_link = lambda *_args, **_kwargs: None
    try:
        return pedant_module.PEDANT()
    finally:
        pedant_module.download_link = original_download


@nested_dataclass(kw_only=True)
class VoiceBenchEvaluatorConfig(BaseEvaluatorConfig):
    openai_api_key_env: str = "OPENAI_API_KEY"
    judge_max_retries: int = 25
    judge_error_sleep_seconds: float = 30.0
    judge_success_sleep_seconds: float = 0.5
    enable_pedant: bool = True


class VoiceBenchEvaluator(BaseEvaluator):
    def __init__(self, config: dict[str, Any]):
        self.config = VoiceBenchEvaluatorConfig(**config)
        super().__init__(self.config, num_parallel_requests=4)
        self._client = None
        self._pedant = None
        self._mcq = MCQEvaluator()
        self._bbh = BBHEvaluator()
        self._harm = HarmEvaluator()

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            api_key = os.environ.get(self.config.openai_api_key_env)
            if not api_key:
                raise RuntimeError(f"VoiceBench judge requires {self.config.openai_api_key_env}")
            self._client = OpenAI(api_key=api_key)
        return self._client

    def _judge_sync(self, prompt: str) -> list[str]:
        request = dict(JUDGE_REQUEST_KWARGS)
        request["messages"] = judge_messages(prompt)
        error = None
        for _ in range(self.config.judge_max_retries):
            try:
                completion = self._get_client().chat.completions.create(**request)
                outputs = [choice.message.content.strip() for choice in completion.choices]
                if self.config.judge_success_sleep_seconds:
                    time.sleep(self.config.judge_success_sleep_seconds)
                return outputs
            except Exception as exc:  # noqa: BLE001 - official API helper retries every provider error
                error = exc
                time.sleep(self.config.judge_error_sleep_seconds)
        raise RuntimeError("VoiceBench GPT judge exhausted retries") from error

    def _pedant_sync(self, data_point: dict, response: str) -> bool:
        if self._pedant is None:
            self._pedant = load_pinned_pedant()
        return bool(
            self._pedant.evaluate(
                [str(data_point["reference"]).lower()], response.lower(), str(data_point["prompt"]).lower()
            )
        )

    async def eval_single(self, data_point: dict[str, Any]) -> dict[str, Any]:
        task = data_point["voicebench_task"]
        generation = data_point.get("generation")
        response = generation if isinstance(generation, str) else ""
        result: dict[str, Any] = {"response": response, "official_code_revision": OFFICIAL_CODE_REVISION}

        if task in OPEN_TASKS:
            prompt = OPEN_JUDGE_TEMPLATE.replace("{prompt}", data_point["prompt"]).replace("{response}", response)
            outputs = await asyncio.to_thread(self._judge_sync, prompt)
            result.update(score=outputs, voicebench_open_ratings=[extract_open_rating(value) for value in outputs])
        elif task == "sd-qa":
            prompt = (
                QA_JUDGE_TEMPLATE.replace("{prompt}", data_point["prompt"])
                .replace("{reference}", data_point["reference"])
                .replace("{response}", response)
            )
            outputs = await asyncio.to_thread(self._judge_sync, prompt)
            result.update(score=outputs, voicebench_qa_gpt_correct=official_majority_vote(outputs))
            if self.config.enable_pedant:
                result["voicebench_qa_pedant_correct"] = await asyncio.to_thread(
                    self._pedant_sync, data_point, response
                )
        elif task in MCQ_TASKS:
            parsed = self._mcq.extract_answer(response)
            parity_answer = parsed if parsed is not None else random.choice(["A", "B", "C", "D"])
            result.update(
                voicebench_parsed_answer=parsed,
                voicebench_parse_success=parsed is not None,
                voicebench_accuracy_unparsed_wrong=parsed == data_point["reference"],
                voicebench_random_fallback_parity_answer=parity_answer,
                voicebench_accuracy_random_fallback_parity=parity_answer == data_point["reference"],
            )
        elif task == "advbench":
            result["voicebench_refusal"] = self._harm.evaluate_single(response)
        elif task == "bbh":
            parsed = self._bbh.extract_answer(response, data_point["id"])
            reference = BBH_GROUND_TRUTH[data_point["reference"].lower()]
            parity_answer = parsed if parsed is not None else random.choice([0, 1])
            result.update(
                voicebench_parsed_answer=parsed,
                voicebench_parse_success=parsed is not None,
                voicebench_accuracy_unparsed_wrong=parsed == reference,
                voicebench_random_fallback_parity_answer=parity_answer,
                voicebench_accuracy_random_fallback_parity=parity_answer == reference,
            )
        elif task == "mtbench":
            result.update(voicebench_protocol_supported=False, voicebench_protocol_gap=data_point["protocol_gap"])
        else:
            raise ValueError(f"Unsupported VoiceBench task: {task}")
        return result
