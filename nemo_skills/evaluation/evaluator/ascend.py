# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Protocol-faithful ASCEND MER/CER evaluator."""

from __future__ import annotations

import re
from typing import Any

import jieba

from nemo_skills.evaluation.evaluator.base import BaseEvaluator, BaseEvaluatorConfig
from nemo_skills.utils import nested_dataclass

OFFICIAL_SCORER_REVISION = "8c50c2e2c65f8555eb1d7655310dd585dc1f6710"
JIEBA_VERSION = "0.42.1"

if getattr(jieba, "__version__", None) != JIEBA_VERSION:
    raise RuntimeError(
        f"ASCEND scoring requires jieba=={JIEBA_VERSION}; found {getattr(jieba, '__version__', None)!r}"
    )


# Copied from the official ASCEND utils.py scorer. Upstream uses a character
# class, so multi-character-looking entries act as individual ignored chars.
CHARS_TO_IGNORE = [
    ",",
    "?",
    "¿",
    ".",
    "!",
    "¡",
    ";",
    "；",
    ":",
    '""',
    "%",
    '"',
    "�",
    "ʿ",
    "·",
    "჻",
    "~",
    "՞",
    "؟",
    "،",
    "।",
    "॥",
    "«",
    "»",
    "„",
    "“",
    "”",
    "「",
    "」",
    "‘",
    "’",
    "《",
    "》",
    "(",
    ")",
    "{",
    "}",
    "=",
    "\x60",
    "_",
    "+",
    "<",
    ">",
    "…",
    "–",
    "°",
    "´",
    "ʾ",
    "‹",
    "›",
    "©",
    "®",
    "—",
    "→",
    "。",
    "、",
    "﹂",
    "﹁",
    "‧",
    "～",
    "﹏",
    "，",
    "｛",
    "｝",
    "（",
    "）",
    "［",
    "］",
    "【",
    "】",
    "‥",
    "〽",
    "『",
    "』",
    "〝",
    "〟",
    "⟨",
    "⟩",
    "〜",
    "：",
    "！",
    "？",
    "♪",
    "؛",
    "/",
    "\\",
    "º",
    "−",
    "^",
    "ʻ",
    "ˆ",
]
_IGNORE_RE = re.compile("[" + re.escape("".join(CHARS_TO_IGNORE)) + "]")


def normalize_ascend(text: str) -> str:
    """Apply ASCEND's official lowercasing and punctuation removal."""
    return _IGNORE_RE.sub("", text).lower() + " "


def mer_tokens(text: str) -> list[str]:
    """Jieba-segment, preserving ASCII spans and splitting non-ASCII spans."""
    output = []
    for token in jieba.lcut(text):
        if not token.strip():
            continue
        output.extend([token] if token.isascii() else list(token))
    return output


def cer_tokens(text: str) -> list[str]:
    return [character for character in text if character.strip()]


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    """Levenshtein distance with O(min(n, m)) memory."""
    if len(reference) < len(hypothesis):
        hypothesis, reference = reference, hypothesis
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_token in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_token in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + (ref_token != hyp_token),
                )
            )
        previous = current
    return previous[-1]


def evaluate_ascend_sample(data_point: dict[str, Any]) -> dict[str, Any]:
    reference = normalize_ascend(str(data_point["expected_answer"]))
    generation = (data_point.get("generation") or "").strip()
    hypothesis = normalize_ascend(generation)
    reference_mer = mer_tokens(reference)
    hypothesis_mer = mer_tokens(hypothesis)
    reference_cer = cer_tokens(reference)
    hypothesis_cer = cer_tokens(hypothesis)
    mer_errors = edit_distance(reference_mer, hypothesis_mer)
    cer_errors = edit_distance(reference_cer, hypothesis_cer)
    return {
        "generation": generation,
        "text": reference,
        "pred_text": hypothesis,
        "mer": mer_errors / len(reference_mer) if reference_mer else None,
        "mer_errors": mer_errors,
        "mer_ref_units": len(reference_mer),
        "cer": cer_errors / len(reference_cer) if reference_cer else None,
        "cer_errors": cer_errors,
        "cer_ref_chars": len(reference_cer),
        "is_correct": mer_errors == 0,
    }


@nested_dataclass(kw_only=True)
class ASCENDEvaluatorConfig(BaseEvaluatorConfig):
    pass


class ASCENDEvaluator(BaseEvaluator):
    async def eval_single(self, data_point: dict[str, Any]) -> dict[str, Any]:
        return evaluate_ascend_sample(data_point)
