# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

"""Deterministic evaluators for the public Marco-LongSpeech task subset."""

import re
import unicodedata
from collections import Counter
from typing import Any

from nemo_skills.evaluation.evaluator.base import BaseEvaluator, BaseEvaluatorConfig
from nemo_skills.utils import nested_dataclass

SUPPORTED_TASKS = {
    "asr",
    "translation",
    "summary",
    "content-separation",
    "speaker-count",
    "language-detection",
}
CJK_LANGUAGE_CODES = {"zh", "zh-cn", "zh-tw", "ja", "ko"}

_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

LANGUAGE_LABEL_TO_CODE = {
    "Czech": "cs",
    "Dutch": "nl",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Hungarian": "hu",
    "Italian": "it",
    "Polish": "pl",
    "Romanian": "ro",
    "Spanish": "es",
}


def _normalize_words(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text)).casefold()
    return " ".join(
        "".join(character if unicodedata.category(character)[0] in {"L", "N"} else " " for character in text).split()
    )


def _normalize_characters(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text)).casefold()
    return "".join(character for character in text if unicodedata.category(character)[0] in {"L", "M", "N"})


def _edit_counts(reference: str, hypothesis: str, *, characters: bool) -> tuple[int, int]:
    import jiwer

    if characters:
        measures = jiwer.process_characters(reference, hypothesis)
    else:
        measures = jiwer.process_words(reference, hypothesis)
    errors = measures.substitutions + measures.deletions + measures.insertions
    reference_units = measures.substitutions + measures.deletions + measures.hits
    return errors, reference_units


def evaluate_asr_sample(reference: str, hypothesis: str, language: str | None) -> dict[str, Any]:
    normalized_reference = _normalize_words(reference)
    normalized_hypothesis = _normalize_words(hypothesis)
    character_reference = _normalize_characters(reference)
    character_hypothesis = _normalize_characters(hypothesis)
    if not normalized_reference or not character_reference:
        raise ValueError("LongSpeech ASR reference becomes empty after deterministic normalization")

    cer_errors, cer_ref_chars = _edit_counts(
        character_reference,
        character_hypothesis,
        characters=True,
    )
    language_code = str(language or "").casefold()
    is_cjk = language_code in CJK_LANGUAGE_CODES
    result = {
        "asr_cer_errors": cer_errors,
        "asr_cer_ref_chars": cer_ref_chars,
        "asr_is_cjk": is_cjk,
        "asr_exact": character_reference == character_hypothesis,
    }
    if not is_cjk:
        wer_errors, wer_ref_words = _edit_counts(
            normalized_reference,
            normalized_hypothesis,
            characters=False,
        )
        result.update(
            asr_wer_errors=wer_errors,
            asr_wer_ref_words=wer_ref_words,
        )
    return result


def _tokens(text: str) -> list[str]:
    return _normalize_words(text).split()


def _f1(overlap: int, predicted: int, reference: int) -> float:
    if predicted == 0 or reference == 0 or overlap == 0:
        return 0.0
    precision = overlap / predicted
    recall = overlap / reference
    return 2.0 * precision * recall / (precision + recall)


def _ngram_counts(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))


def _rouge_n_f1(reference_tokens: list[str], hypothesis_tokens: list[str], n: int) -> float:
    reference_counts = _ngram_counts(reference_tokens, n)
    hypothesis_counts = _ngram_counts(hypothesis_tokens, n)
    overlap = sum((reference_counts & hypothesis_counts).values())
    return _f1(overlap, sum(hypothesis_counts.values()), sum(reference_counts.values()))


def _lcs_length(left: list[str], right: list[str]) -> int:
    if len(right) > len(left):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def evaluate_summary_sample(reference: str, hypothesis: str) -> dict[str, Any]:
    reference_tokens = _tokens(reference)
    hypothesis_tokens = _tokens(hypothesis)
    lcs = _lcs_length(reference_tokens, hypothesis_tokens)
    return {
        "rouge1_f1": _rouge_n_f1(reference_tokens, hypothesis_tokens, 1),
        "rouge2_f1": _rouge_n_f1(reference_tokens, hypothesis_tokens, 2),
        "rougeL_f1": _f1(lcs, len(hypothesis_tokens), len(reference_tokens)),
    }


def parse_numeric_answer(generation: str) -> int | None:
    """Parse one unambiguous integer from a fixed-answer response."""
    normalized = str(generation).casefold()
    values = [int(value) for value in re.findall(r"(?<!\w)[+-]?\d+(?!\w)", normalized)]
    values.extend(_NUMBER_WORDS[token] for token in re.findall(r"[a-z]+", normalized) if token in _NUMBER_WORDS)
    unique = set(values)
    return unique.pop() if len(unique) == 1 else None


def parse_language_answer(generation: str) -> str | None:
    """Parse one canonical pinned-test language name from a response."""
    normalized = str(generation).casefold()
    found = set()
    for label, code in LANGUAGE_LABEL_TO_CODE.items():
        if re.search(rf"\b{re.escape(label.casefold())}\b", normalized):
            found.add(label)
        if normalized.strip() == code:
            found.add(label)
    return found.pop() if len(found) == 1 else None


def evaluate_longspeech_sample(data_point: dict[str, Any]) -> dict[str, Any]:
    task = data_point.get("longspeech_task")
    if task not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unsupported LongSpeech task {task!r}; temporal QA and emotion are intentionally unavailable"
        )

    generation = str(data_point.get("generation") or "").strip()
    reference = data_point["expected_answer"]
    common = {
        "generation": generation,
        "longspeech_task": task,
        "longspeech_no_answer": not bool(generation),
    }

    if task == "asr":
        scores = evaluate_asr_sample(str(reference), generation, data_point.get("language"))
        return {**common, **scores, "is_correct": scores["asr_exact"]}

    if task == "translation":
        return {
            **common,
            "translation_reference": str(reference),
            "translation_hypothesis": generation,
            "is_correct": generation.casefold() == str(reference).strip().casefold(),
        }

    if task == "summary":
        scores = evaluate_summary_sample(str(reference), generation)
        return {**common, **scores, "is_correct": scores["rougeL_f1"] == 1.0}

    if task in {"content-separation", "speaker-count"}:
        expected_numeric = int(reference)
        parsed_numeric = parse_numeric_answer(generation)
        return {
            **common,
            "numeric_expected": expected_numeric,
            "numeric_prediction": parsed_numeric,
            "numeric_parsed": parsed_numeric is not None,
            "numeric_correct": parsed_numeric == expected_numeric,
            "is_correct": parsed_numeric == expected_numeric,
        }

    expected_language = str(reference)
    if expected_language not in LANGUAGE_LABEL_TO_CODE:
        raise ValueError(f"Unknown pinned LongSpeech language label: {expected_language!r}")
    predicted_language = parse_language_answer(generation)
    return {
        **common,
        "language_expected": expected_language,
        "language_prediction": predicted_language,
        "language_parsed": predicted_language is not None,
        "language_correct": predicted_language == expected_language,
        "is_correct": predicted_language == expected_language,
    }


@nested_dataclass(kw_only=True)
class LongSpeechEvaluatorConfig(BaseEvaluatorConfig):
    pass


class LongSpeechEvaluator(BaseEvaluator):
    async def eval_single(self, data_point: dict[str, Any]) -> dict[str, Any]:
        return evaluate_longspeech_sample(data_point)
