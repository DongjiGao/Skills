# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline, annotation-only entity verification for VoiceCodeBench.

This verifier is deliberately conservative: it marks an entity present only
when an exact canonical or expected-acoustic rendering is recoverable after a
versioned set of deterministic formatting normalizations. It does not use the
reference transcript, entity hints during generation, or an external judge.
Its CTEM/TSR values are therefore reproducible, but are not interchangeable
with the benchmark release's ``openai_gpt_5_5_v1`` verifier values.
"""

import hashlib
import json
import re
import unicodedata
from typing import Any

DETERMINISTIC_VERIFIER_ID = "deterministic_annotations_v1"
DETERMINISTIC_VERIFIER_MODEL = "offline_exact_annotation_normalizer"
_RULE_SPEC = {
    "id": DETERMINISTIC_VERIFIER_ID,
    "unicode": "NFKC+casefold",
    "match": "contiguous exact canonical-or-acoustic semantic tokens",
    "numbers": "english-cardinal-and-ordinal-to-digits",
    "spelled_sequence": "explicit_single_letter_sequence_only",
}
DETERMINISTIC_VERIFIER_DIGEST = hashlib.sha256(
    json.dumps(_RULE_SPEC, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()

_PUNCTUATION = {
    ".": " dot ",
    "-": " dash ",
    "_": " underscore ",
    "/": " slash ",
    "\\": " backslash ",
    "@": " at ",
    "+": " plus ",
    ":": " colon ",
    "*": " star ",
    "%": " percent ",
    "$": " dollar ",
    "=": " equals ",
    "#": " hash ",
    "|": " pipe ",
    "~": " tilde ",
    "&": " and ",
}
_UNITS = {
    "zero": 0,
    "oh": 0,
    "o": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}
_TEENS = {
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
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
    "thirtieth": 30,
}
_SCALES = {"hundred", "thousand", "million"}
_NUMBER_WORDS = set(_UNITS) | set(_TEENS) | set(_TENS) | set(_ORDINALS) | _SCALES | {"and"}
_ALIASES = {
    "dollars": "dollar",
    "cents": "cent",
    "pounds": "lb",
    "kilograms": "kg",
    "inches": "inch",
    "extension": "ext",
    "x": "ext",
    "eastern": "et",
    "central": "ct",
    "pacific": "pt",
    "mountain": "mt",
    "point": "dot",
}
_TYPE_IGNORED = {
    "phone_number": {"area", "code", "dash"},
    "phone_extension": {"ext", "number"},
    "currency_amount": {"dollar", "cent", "and", "dot"},
    "time": {"between", "from", "to", "and", "dash", "colon", "dot"},
}
_SPELLING_SEPARATORS = {"dash", "dot"}


def _lex(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = text.casefold().replace("’", "'")
    for punctuation, name in _PUNCTUATION.items():
        text = text.replace(punctuation, name)
    return re.findall(r"[a-z0-9]+", text)


def _parse_number_run(words: list[str]) -> list[str]:
    words = [word for word in words if word != "and"]
    if not words:
        return []
    if any(word in _SCALES for word in words):
        total = 0
        current = 0
        for word in words:
            if word in _UNITS:
                current += _UNITS[word]
            elif word in _TEENS:
                current += _TEENS[word]
            elif word in _TENS:
                current += _TENS[word]
            elif word in _ORDINALS:
                current += _ORDINALS[word]
            elif word == "hundred":
                current = max(1, current) * 100
            elif word == "thousand":
                total += max(1, current) * 1000
                current = 0
            elif word == "million":
                total += max(1, current) * 1_000_000
                current = 0
        return list(str(total + current))

    digits: list[str] = []
    index = 0
    while index < len(words):
        word = words[index]
        if word in _TENS and index + 1 < len(words):
            following = words[index + 1]
            following_value = _UNITS.get(following, _ORDINALS.get(following))
            if following_value is not None and following_value < 10:
                digits.extend(str(_TENS[word] + following_value))
                index += 2
                continue
        value = _UNITS.get(word, _TEENS.get(word, _TENS.get(word, _ORDINALS.get(word))))
        if value is None:
            raise ValueError(f"Unhandled number word: {word}")
        digits.extend(str(value))
        index += 1
    return digits


def _join_letter_runs(tokens: list[str]) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(tokens):
        if len(tokens[index]) == 1 and tokens[index].isalpha():
            end = index + 1
            while end < len(tokens) and len(tokens[end]) == 1 and tokens[end].isalpha():
                end += 1
            result.append("".join(tokens[index:end]))
            index = end
        else:
            result.append(tokens[index])
            index += 1
    return result


def _semantic_tokens(text: str) -> list[str]:
    source = _lex(text)
    result: list[str] = []
    index = 0
    while index < len(source):
        word = source[index]
        if word.isdigit():
            result.extend(word)
            index += 1
            continue
        if word in _NUMBER_WORDS and word != "and":
            end = index + 1
            while end < len(source) and source[end] in _NUMBER_WORDS:
                end += 1
            result.extend(_parse_number_run(source[index:end]))
            index = end
            continue
        result.append(_ALIASES.get(word, word))
        index += 1
    return _join_letter_runs(result)


def _contains(needle: list[str], haystack: list[str]) -> bool:
    width = len(needle)
    return bool(width) and any(
        haystack[index : index + width] == needle
        for index in range(len(haystack) - width + 1)
    )


def _significant(tokens: list[str], entity_type: str) -> list[str]:
    ignored = _TYPE_IGNORED.get(entity_type, set())
    return [token for token in tokens if token not in ignored]


def _contains_explicit_spelling(canonical: str, transcript: str) -> bool:
    target = [token for token in _lex(canonical) if len(token) == 1 and token.isalpha()]
    if len(target) < 2:
        return False
    source = _lex(transcript)
    for start, token in enumerate(source):
        if token != target[0]:
            continue
        cursor = start + 1
        matched = 1
        while matched < len(target):
            while cursor < len(source) and source[cursor] in _SPELLING_SEPARATORS:
                cursor += 1
            if cursor >= len(source) or source[cursor] != target[matched]:
                break
            cursor += 1
            matched += 1
        if matched == len(target):
            return True
    return False


def _entity_present(entity: dict[str, Any], transcript: str) -> tuple[bool, str]:
    entity_type = str(entity["type"])
    if entity_type == "spelled_sequence":
        present = _contains_explicit_spelling(str(entity["canonical"]), transcript)
        return present, "explicit letter sequence" if present else "no exact explicit letter sequence"

    haystack = _significant(_semantic_tokens(transcript), entity_type)
    for field in ("canonical", "acoustic"):
        needle = _significant(_semantic_tokens(str(entity[field])), entity_type)
        if _contains(needle, haystack):
            return True, f"exact normalized {field} rendering"
    return False, "no exact canonical or expected-acoustic rendering"


def verify_deterministic_entity_matches(
    datapoint: dict[str, Any], transcript: str
) -> list[dict[str, Any]]:
    """Return official-schema entity decisions using only frozen annotations."""
    entities = datapoint.get("entities")
    if not isinstance(entities, list) or not entities:
        raise ValueError("VoiceCodeBench datapoint has no entity annotations")
    matches = []
    for target_index, entity in enumerate(entities):
        for field in ("type", "canonical", "acoustic"):
            if not isinstance(entity.get(field), str) or not entity[field]:
                raise ValueError(f"VoiceCodeBench entity {target_index} has invalid {field}")
        present, reason = _entity_present(entity, transcript)
        matches.append(
            {
                "target_index": target_index,
                "type": entity["type"],
                "canonical": entity["canonical"],
                "present": present,
                "evidence": reason if present else "",
                "reason": f"{DETERMINISTIC_VERIFIER_ID}: {reason}",
            }
        )
    return matches
