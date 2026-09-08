# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Pinned VoiceBench evaluation, with random fallback isolated as parity only."""

import asyncio
import hashlib
import json
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
PEDANT_SOURCE_ARTIFACT_SHA256 = {
    "lr_classifier.pkl": "9c3dd5998745ece1cfb6cb16b953358f794e8716d63a08f3b212d8bc32a59b8b",
    "tf-idf_vectorizer.pkl": "e886c9ba3ddb15a4b693b075dfe19cfad36dc74e7f8f195b8d9e98e989f2ba4f",
}
PEDANT_PORTABLE_DIR = Path(__file__).resolve().parent / "resources" / "voicebench_pedant"
PEDANT_PORTABLE_ARTIFACT_SHA256 = {
    "classes.npy": "f8c1c077bc172a77ad7909da673a28b507c0b1fb3f273380e16fbdf424c76150",
    "coef.npy": "9a1be843750eba422217b43e5ba511e771e55a50f5cf202533e74e0993e7ad03",
    "idf.npy": "97f57677f9a4c5c4ca20fc77f6a5ac70217ca6196c7c0c4a836fa7160124b121",
    "intercept.npy": "5e278f912b28b4df6a086f50dca84eb012e8cf95a364b9061eff14bf5d237251",
    "model.json": "e188e36b04431b42061a06c8a8eb7d3365424c5c0feafd84d45126443306f048",
    "parity.json": "a1b1b8b79c4522d59d16a615f8b91a741975ab6897b4b91a9ecb4d3e4539e306",
    "vectorizer.json": "f46bf836d373bebb2afd0f1de19efa78f2263828792f4da8479116de1edd4387",
    "vocabulary.json": "b929c6943a4ff2fe27efbcf10f192d710ba56726de48a37561d16004407b58b7",
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


def _verify_artifacts(directory: Path, expected: dict[str, str]) -> None:
    for filename, expected_hash in expected.items():
        artifact = directory / filename
        if not artifact.is_file():
            raise RuntimeError(f"Pinned PEDANT artifact is missing: {artifact}")
        actual_hash = _sha256(artifact)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Pinned PEDANT artifact hash mismatch for {artifact}: expected {expected_hash}, got {actual_hash}"
            )


class _PortableLinearClassifier:
    """Stable inference-only port of PEDANT's scikit-learn 1.3 SGDClassifier."""

    def __init__(self, coef, intercept, classes):
        self.coef = coef
        self.intercept = intercept
        self.classes = classes

    def decision_function(self, features):
        import numpy as np

        scores = np.asarray(features @ self.coef.T) + self.intercept
        return scores[:, 0] if self.coef.shape[0] == 1 else scores

    def predict(self, features):
        import numpy as np

        scores = self.decision_function(features)
        if self.coef.shape[0] == 1:
            indices = (scores > 0.0).astype(int)
        else:
            indices = np.argmax(scores, axis=1)
        return self.classes[indices]


def load_pinned_pedant(pedant_module=None, portable_dir: Path | None = None):
    """Load an exact portable PEDANT without mutable downloads/old sklearn pickles."""
    if pedant_module is None:
        try:
            import qa_metrics.pedant as pedant_module
        except ImportError as exc:
            raise RuntimeError("Official SD-QA PEDANT scoring requires qa_metrics==0.2.17") from exc

    import numpy as np
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer

    portable_dir = portable_dir or PEDANT_PORTABLE_DIR
    _verify_artifacts(portable_dir, PEDANT_PORTABLE_ARTIFACT_SHA256)
    model_meta = json.loads((portable_dir / "model.json").read_text(encoding="utf-8"))
    vocabulary = json.loads((portable_dir / "vocabulary.json").read_text(encoding="utf-8"))
    vectorizer_config = json.loads((portable_dir / "vectorizer.json").read_text(encoding="utf-8"))
    vectorizer_config["ngram_range"] = tuple(vectorizer_config["ngram_range"])

    vectorizer = TfidfVectorizer(**vectorizer_config, vocabulary=vocabulary)
    vectorizer.fit(["portable pedant initialization"])
    vectorizer.idf_ = np.load(portable_dir / "idf.npy", allow_pickle=False)
    model = _PortableLinearClassifier(
        coef=np.load(portable_dir / "coef.npy", allow_pickle=False),
        intercept=np.load(portable_dir / "intercept.npy", allow_pickle=False),
        classes=np.load(portable_dir / "classes.npy", allow_pickle=False),
    )
    if list(model.coef.shape) != model_meta["coef_shape"]:
        raise RuntimeError("Portable PEDANT coefficient shape does not match authenticated metadata")
    if model.classes.tolist() != model_meta["classes"]:
        raise RuntimeError("Portable PEDANT classes do not match authenticated metadata")

    parity = json.loads((portable_dir / "parity.json").read_text(encoding="utf-8"))
    parity_features = hstack(
        [vectorizer.transform(parity["texts"]), np.asarray(parity["extras"], dtype=np.float64)]
    )
    decision = np.asarray(model.decision_function(parity_features))
    if not np.allclose(decision, np.asarray(parity["decision"]), rtol=0.0, atol=1e-12):
        raise RuntimeError("Portable PEDANT decision parity check failed")
    if model.predict(parity_features).tolist() != parity["prediction"]:
        raise RuntimeError("Portable PEDANT prediction parity check failed")

    pedant = pedant_module.PEDANT.__new__(pedant_module.PEDANT)
    pedant.model = model
    pedant.tokenizer = vectorizer
    return pedant


@nested_dataclass(kw_only=True)
class VoiceBenchEvaluatorConfig(BaseEvaluatorConfig):
    openai_api_key_env: str = "OPENAI_API_KEY"
    openai_base_url: str | None = None
    judge_model: str = JUDGE_MODEL
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
            client_kwargs = {"api_key": api_key}
            if self.config.openai_base_url is not None:
                client_kwargs["base_url"] = self.config.openai_base_url
            self._client = OpenAI(**client_kwargs)
        return self._client

    def _judge_sync(self, prompt: str) -> list[str]:
        request = dict(JUDGE_REQUEST_KWARGS)
        target_outputs = int(request.get("n", 1))
        request["model"] = self.config.judge_model
        request["messages"] = judge_messages(prompt)
        error = None
        outputs = []
        for _ in range(self.config.judge_max_retries):
            request["n"] = target_outputs - len(outputs)
            try:
                completion = self._get_client().chat.completions.create(**request)
                batch = [choice.message.content.strip() for choice in completion.choices]
                if not batch:
                    raise RuntimeError("VoiceBench GPT judge returned no choices")
                outputs.extend(batch[: target_outputs - len(outputs)])
                if self.config.judge_success_sleep_seconds:
                    time.sleep(self.config.judge_success_sleep_seconds)
                if len(outputs) == target_outputs:
                    return outputs
            except Exception as exc:  # noqa: BLE001 - official API helper retries every provider error
                error = exc
                if self.config.judge_error_sleep_seconds:
                    time.sleep(self.config.judge_error_sleep_seconds)
        raise RuntimeError(
            f"VoiceBench GPT judge exhausted retries with {len(outputs)}/{target_outputs} choices"
        ) from error

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
