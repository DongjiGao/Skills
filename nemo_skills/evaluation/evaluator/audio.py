# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Audio evaluation framework supporting ASR, ASR-PC, Translation, CER, and more."""

import asyncio
import itertools
import logging
import re
import unicodedata
from typing import Any

import numpy as np

from nemo_skills.evaluation.evaluator.base import BaseEvaluator, BaseEvaluatorConfig
from nemo_skills.utils import get_logger_name, nested_dataclass

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class AudioEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for audio evaluation."""

    prompt_config: str = "eval/speechlm/audio"
    normalize_asr_pc_standard_wer: bool = True
    strip_helpful_prefixes: bool = False
    apply_normalization: bool = True
    # See VALID_NORMALIZATION_MODES for accepted values.
    normalization_mode: str = "standard"

    # Optional list of reference fields to calculate WER against (e.g., ["text_tn", "text_itn"])
    # For each field, WER will be computed and stored with corresponding metric name
    reference_fields: list[str] | None = None


# non-ASCII letters that are not separated by "NFKD" normalization
ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}


def remove_symbols_and_diacritics(s: str, keep: str = ""):
    """
    Replace any other markers, symbols, and punctuations with a space, and drop any diacritics (category 'Mn' and some
    manual mappings)
    """

    def replace_character(char):
        if char in keep:
            return char
        elif char in ADDITIONAL_DIACRITICS:
            return ADDITIONAL_DIACRITICS[char]

        elif unicodedata.category(char) == "Mn":
            return ""

        elif unicodedata.category(char)[0] in "MSP":
            return " "

        return char

    return "".join(replace_character(c) for c in unicodedata.normalize("NFKD", s))


def remove_symbols(s: str):
    """
    Replace any other markers, symbols, punctuations with a space, keeping diacritics
    """
    return "".join(" " if unicodedata.category(c)[0] in "MSP" else c for c in unicodedata.normalize("NFKC", s))


def remove_symbols_keep_marks(s: str):
    """
    Replace symbols and punctuation (categories S*, P*) with a space, but KEEP combining
    marks (M*) and letters/numbers. For abugida scripts (Thai/Lao/Myanmar/Khmer/Indic),
    the vowel and tone marks are meaning-bearing, unlike Latin diacritics, so they must
    not be stripped. Uses NFC to keep precomposed forms intact.
    """
    return "".join(" " if unicodedata.category(c)[0] in "SP" else c for c in unicodedata.normalize("NFC", s))


def normalize_compound_pairs(ref_text: str, pred_text: str) -> tuple[str, str]:
    """Normalize compound word boundaries between ref/pred pairs.

    When a mismatch region has identical characters ignoring whitespace,
    normalize both sides to the joined form.
    """
    from difflib import SequenceMatcher

    ref_words = ref_text.split()
    pred_words = pred_text.split()

    sm = SequenceMatcher(None, ref_words, pred_words)
    new_rw, new_pw = [], []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            new_rw.extend(ref_words[i1:i2])
            new_pw.extend(pred_words[j1:j2])
        else:
            rc = "".join(ref_words[i1:i2])
            pc = "".join(pred_words[j1:j2])
            if rc == pc:
                new_rw.append(rc)
                new_pw.append(pc)
            else:
                new_rw.extend(ref_words[i1:i2])
                new_pw.extend(pred_words[j1:j2])
    return " ".join(new_rw), " ".join(new_pw)


class MultilingualTextNormalizer:
    """Multilingual text normalizer with optional number normalization.

    Call with just text for standard normalization.
    Pass lang= to also convert digits to words via num2words.
    """

    def __init__(self, remove_diacritics: bool = True, preserve_marks: bool = False):
        # preserve_marks keeps combining marks (M*) — essential vowels/tones in abugida
        # scripts — while still removing symbols/punctuation. Overrides remove_diacritics.
        self.preserve_marks = preserve_marks
        if preserve_marks:
            self.clean = remove_symbols_keep_marks
        else:
            self.clean = remove_symbols_and_diacritics if remove_diacritics else remove_symbols

    def _normalize_numbers(self, text, lang):
        import num2words

        # Join space-separated thousand groups (e.g. "10 000" -> "10000")
        text = re.sub(r"(\d)\s+(\d{3})\b", r"\1\2", text)

        # Convert remaining digit sequences to words
        def _replace(m):
            try:
                return num2words.num2words(int(m.group()), lang=lang)
            except Exception:
                return m.group()

        return re.sub(r"\d+", _replace, text)

    def __call__(self, s: str, lang=None):
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = self.clean(s).lower()

        # Remove punctuations and extra spaces. Skip the [^\w\s] strip when preserving
        # marks (it would delete combining marks, which \w does not match); self.clean
        # has already removed symbols/punctuation in that case.
        if not self.preserve_marks:
            s = re.sub(r"[^\w\s]", "", s)
        s = normalize_whitespace(s)

        if lang is not None:
            s = self._normalize_numbers(s, lang)
        return s


# Known model failure responses that should be treated as empty transcriptions
_FAILURE_RESPONSES = [
    r"the speech is in audio format and needs to be transcribed",
    r"i do not have access to audio",
    r"i cannot access audio",
    r"i'm sorry.*i do not have access",
    r"as an ai language model.*i do not have access",
]


def extract_asr_text(generation: str) -> str:
    """Extract ASR text from generation."""

    def parse_qwen_asr_output(generation: str) -> str:
        _ASR_TEXT_TAG = "<asr_text>"
        s = str(generation).strip()
        has_tag = _ASR_TEXT_TAG in s
        if has_tag:
            s = s.split(_ASR_TEXT_TAG, 1)[1]
        return s.strip()

    result = parse_qwen_asr_output(generation)
    return result.strip()


def strip_helpful_prefixes(text: str) -> str:
    """Strip ASR response prefixes like 'The audio says: ...' for accurate WER.

    Also removes SRT subtitle timestamps that can appear in vLLM chunked audio generation.
    """
    result = text.strip()

    # Check for model failure responses
    for failure_pattern in _FAILURE_RESPONSES:
        if re.search(failure_pattern, result, flags=re.IGNORECASE):
            return ""

    # Remove SRT subtitle timestamps (vLLM chunked audio artifact)
    result = re.sub(r"\d+\s+\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}\s+", "", result)
    result = re.sub(r"\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}\s*", "", result)
    result = re.sub(r"\\n\d+\s+(?=\d{2}:\d{2})", " ", result)
    result = re.sub(r"\n\d+\s+(?=\d{2}:\d{2})", " ", result)

    # Extract from double quotes
    match = re.search(r'"((?:\\.|[^"\\])*)"', result)
    if match:
        result = match.group(1)

    # Handle colon-quote patterns (e.g., "The audio says: 'hello'")
    if ": '" in result:
        result = result.split(": '", 1)[1]
        # Remove trailing quote (with punctuation either inside or outside)
        result = re.sub(r"[.!?]?'[.!?]?\s*$", "", result)
    elif ":'" in result:
        result = result.split(":'", 1)[1]
        result = re.sub(r"[.!?]?'[.!?]?\s*$", "", result)
    else:
        # Single quote extraction - only match quotes at end of string (not contractions like o'clock)
        match = re.search(r"'(.+?)'\s*\.?\s*$", result)
        if match:
            result = match.group(1)

    return result.strip()


def normalize_whitespace(text: str) -> str:
    """Normalize multiple spaces to single space."""
    return re.sub(r"\s+", " ", text).strip()


def split_tokens(text: str) -> list[str]:
    """Split text into words and punctuation as separate tokens."""
    return re.findall(r"\w+|[^\w\s]", text)


# Scriptio-continua scripts (no inter-word spaces) whose units are tokenized
# individually for Mixed Error Rate, while space-delimited scripts (e.g. Latin)
# keep their whitespace-delimited words. Covers every CER-style script that
# occurs in CS-FLEURS plus the obvious siblings (Lao/Khmer).
_CONTINUA_RANGES = (
    "㐀-䶿"  # CJK Unified Ideographs Extension A
    "一-鿿"  # CJK Unified Ideographs
    "豈-﫿"  # CJK Compatibility Ideographs
    "぀-ゟ"  # Hiragana
    "゠-ヿ"  # Katakana
    "ㇰ-ㇿ"  # Katakana Phonetic Extensions
    "ᄀ-ᇿ"  # Hangul Jamo
    "㄰-㆏"  # Hangul Compatibility Jamo
    "가-힣"  # Hangul Syllables
    "฀-๿"  # Thai
    "຀-໿"  # Lao
    "က-႟"  # Myanmar
    "ꩠ-ꩿ"  # Myanmar Extended-A
    "ក-៿"  # Khmer
)
_CONTINUA_RE = re.compile(f"[{_CONTINUA_RANGES}]")


def _grapheme_clusters(text: str) -> list[str]:
    """Group each base character with its trailing combining marks.

    A lightweight, dependency-free grapheme segmentation: combining marks
    (Unicode categories Mn/Mc/Me) attach to the preceding base. This keeps a
    Thai/Myanmar consonant together with its vowel signs and tone marks, which
    naive per-codepoint splitting would tear apart.
    """
    clusters: list[str] = []
    for ch in text:
        if clusters and unicodedata.category(ch) in ("Mn", "Mc", "Me"):
            clusters[-1] += ch
        else:
            clusters.append(ch)
    return clusters


def mixed_segment(text: str, grapheme: bool = False) -> list[str]:
    """Tokenize code-switched text for Mixed Error Rate (MER).

    Each scriptio-continua unit (Han/kana/Hangul/Thai/Lao/Myanmar/Khmer) becomes
    its own token; runs of space-delimited script (e.g. Latin) stay as their
    whitespace-delimited words. The result, joined with spaces, can be scored
    with the standard word-level edit distance so Mandarin is counted by
    character and English by word in the same utterance.

    Args:
        text: Normalized transcription (normalize first, then segment).
        grapheme: When True, scriptio-continua units are grapheme clusters
            (base + combining marks) rather than single codepoints. Needed for
            Thai/Myanmar to avoid splitting tone/vowel marks off their base;
            unnecessary for Han/kana/Hangul (no combining marks). Default False
            keeps per-codepoint counting, consistent with the evaluator's CER.
    """
    units = _grapheme_clusters(text) if grapheme else list(text)
    tokens: list[str] = []
    buffer: list[str] = []

    def _flush():
        if buffer:
            tokens.append("".join(buffer))
            buffer.clear()

    for unit in units:
        if unit[0].isspace():
            _flush()
        elif _CONTINUA_RE.match(unit[0]):
            _flush()
            tokens.append(unit)
        else:
            buffer.append(unit)
    _flush()
    return tokens


def extract_punctuation(text: str) -> list[str]:
    """Extract only punctuation characters from text."""
    return [c for c in text if not c.isalnum() and not c.isspace()]


def calculate_per(reference: str, hypothesis: str) -> float:
    """Calculate Punctuation Error Rate (PER): (I+D+S) / (I+D+S+C)"""
    ref_punct = extract_punctuation(reference)
    hyp_punct = extract_punctuation(hypothesis)

    len_r, len_h = len(ref_punct), len(hyp_punct)

    if len_r == 0 and len_h == 0:
        return 0.0

    dp = np.zeros((len_r + 1, len_h + 1, 4), dtype=int)

    for i in range(1, len_r + 1):
        dp[i, 0][2] = i
    for j in range(1, len_h + 1):
        dp[0, j][3] = j

    for i in range(1, len_r + 1):
        for j in range(1, len_h + 1):
            if ref_punct[i - 1] == hyp_punct[j - 1]:
                dp[i, j] = dp[i - 1, j - 1].copy()
                dp[i, j][0] += 1
            else:
                sub = dp[i - 1, j - 1].copy()
                sub[1] += 1
                delete = dp[i - 1, j].copy()
                delete[2] += 1
                insert = dp[i, j - 1].copy()
                insert[3] += 1
                dp[i, j] = min([sub, delete, insert], key=lambda x: x[1] + x[2] + x[3])

    correct, substitution, deletion, insertion = dp[len_r, len_h]
    total = correct + substitution + deletion + insertion
    per = (substitution + deletion + insertion) / total if total > 0 else 0.0
    return per


def evaluate_asr_pc(
    reference: str, hypothesis: str, normalize_standard_wer: bool = True, normalization_mode: str = "standard"
) -> dict[str, Any]:
    """Evaluate ASR-PC: computes WER, WER_C, WER_PC, PER.

    Args:
        reference: Ground truth transcription.
        hypothesis: Model output transcription.
        normalize_standard_wer: Whether to apply normalization to standard WER.
        normalization_mode: Normalization mode for standard WER ("standard", "audiobench", "hf_leaderboard", "none", "no_tn_itn").
    """
    import jiwer

    ref_pc = normalize_whitespace(reference)
    hyp_pc = normalize_whitespace(hypothesis)

    ref_tokens = split_tokens(ref_pc)
    hyp_tokens = split_tokens(hyp_pc)
    wer_pc = jiwer.wer(" ".join(ref_tokens), " ".join(hyp_tokens))

    ref_c = normalize_whitespace(re.sub(r"[^\w\s]", "", reference))
    hyp_c = normalize_whitespace(re.sub(r"[^\w\s]", "", hypothesis))
    wer_c = jiwer.wer(ref_c, hyp_c)

    if normalize_standard_wer:
        ref_std = preprocess_asr_text(reference, mode=normalization_mode)
        hyp_std = preprocess_asr_text(hypothesis, mode=normalization_mode)
    else:
        ref_std = normalize_whitespace(re.sub(r"[^\w\s]", "", reference.lower()))
        hyp_std = normalize_whitespace(re.sub(r"[^\w\s]", "", hypothesis.lower()))

    result = _wer_with_counts(ref_std, hyp_std)
    per = calculate_per(reference, hypothesis)
    result["wer_c"] = wer_c
    result["wer_pc"] = wer_pc
    result["per"] = per
    result["is_correct"] = wer_pc < 0.5
    result["text"] = ref_std
    result["pred_text"] = hyp_std
    return result


def _normalize_digits_to_words(text: str) -> str:
    """Convert standalone digits to words (e.g., '1' -> 'one')."""
    digits_to_words = {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
        "10": "ten",
        "11": "eleven",
        "12": "twelve",
        "13": "thirteen",
        "14": "fourteen",
        "15": "fifteen",
        "16": "sixteen",
        "17": "seventeen",
        "18": "eighteen",
        "19": "nineteen",
        "20": "twenty",
        "30": "thirty",
        "40": "forty",
        "50": "fifty",
        "60": "sixty",
        "70": "seventy",
        "80": "eighty",
        "90": "ninety",
    }
    for digit, word in digits_to_words.items():
        text = re.sub(r"\b" + digit + r"\b", word, text)
    return text


def _expand_contractions(text: str) -> str:
    """Expand common English contractions (e.g., "I'm" -> "I am")."""
    contractions = {
        "i'm": "i am",
        "you're": "you are",
        "he's": "he is",
        "she's": "she is",
        "it's": "it is",
        "we're": "we are",
        "they're": "they are",
        "i've": "i have",
        "you've": "you have",
        "we've": "we have",
        "they've": "they have",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "hasn't": "has not",
        "haven't": "have not",
        "hadn't": "had not",
        "doesn't": "does not",
        "don't": "do not",
        "didn't": "did not",
        "that's": "that is",
    }
    for contraction, expanded in contractions.items():
        text = re.sub(r"\b" + contraction + r"\b", expanded, text)
    return text


def _remove_non_speech_elements(text: str) -> str:
    """Remove filler words (uh, um, er, ah)."""
    non_speech_patterns = r"\b(uh|umm|um|er|ah)\b"
    return re.sub(non_speech_patterns, "", text)


VALID_NORMALIZATION_MODES = (
    "standard",
    "audiobench",
    "hf_leaderboard",
    "none",
    "no_tn_itn",
    "multilingual",
    "apptek_callcenter",
    "lower_nopunct",
)
TASKS_NEED_PARSING = {"ASR", "ASR-PC", "ASR_LEADERBOARD", "Multilingual-ASR", "CER", "Hallucination", "PC-Rate"}


def resolve_asr_normalization_mode(config: AudioEvaluatorConfig) -> str:
    """Resolve effective normalization mode for ASR-family tasks.

    - no_tn_itn is explicit and does not use whisper normalization.
    - Other modes respect apply_normalization toggle.
    """
    if config.normalization_mode == "no_tn_itn":
        return "no_tn_itn"
    return config.normalization_mode if config.apply_normalization else "none"


def preprocess_asr_text(text: str, mode: str = "standard", **kwargs) -> str:
    """Normalize ASR text for WER calculation.

    Args:
        text: Raw text.
        mode: Normalization mode:
            - "standard": Whisper normalization (default) - converts number words to digits
            - "audiobench": Full AudioBench normalization (whisper + digits to words + more)
            - "hf_leaderboard": HuggingFace leaderboard style (whisper normalization)
            - "none": No normalization (whitespace only)
            - "no_tn_itn": Lowercase + remove punctuation, no number word conversion (for TN/ITN eval)
            - "multilingual": Multilingual normalization
            - "apptek_callcenter": AppTek Call-Center Dialogues official normalization
            - "lower_nopunct": Lowercase + remove punctuation while preserving combining marks
        **kwargs: Additional keyword arguments.
    """
    if mode not in VALID_NORMALIZATION_MODES:
        raise ValueError(
            f"Invalid normalization_mode '{mode}'. Available options: {', '.join(VALID_NORMALIZATION_MODES)}"
        )

    if mode == "none":
        return normalize_whitespace(text)

    if mode == "no_tn_itn":
        # Lowercase + remove punctuation + whitespace normalization
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return normalize_whitespace(text)

    if mode == "apptek_callcenter":
        from nemo_skills.evaluation.evaluator.apptek_callcenter import normalize_apptek_callcenter_text

        return normalize_apptek_callcenter_text(text, is_prediction=kwargs.get("is_prediction", False))

    if mode == "lower_nopunct":
        # Lowercase + drop punctuation only, PRESERVING combining marks (Mn/Mc).
        # Unlike no_tn_itn (whose [^\w\s] also strips abugida vowel/tone marks,
        # since \w does not match them), this keeps Thai/Myanmar/Indic vowels and
        # tones. Matches the CS-FLEURS paper's "case insensitive and unpunctuated"
        # character error rate, so per-language CER is comparable to the paper.
        text = text.lower()
        text = "".join("" if unicodedata.category(ch).startswith("P") else ch for ch in text)
        return normalize_whitespace(text)

    from whisper_normalizer.english import EnglishTextNormalizer

    if mode in ["standard", "hf_leaderboard"]:
        text = text.lower()
        text = EnglishTextNormalizer()(text)
        return normalize_whitespace(text)

    # "audiobench" uses additional (audiobench-specific) normalization
    if mode == "audiobench":
        import jiwer

        text = text.lower()
        text = EnglishTextNormalizer()(text)
        text = _normalize_digits_to_words(text)
        text = _expand_contractions(text)
        text = re.sub(r"(\[|\(|\{|\<)[^\(\)\\n\[\]]*(\]|\)|\}|\>)", "", text)
        jiwer_process = jiwer.Compose(
            [
                jiwer.RemoveMultipleSpaces(),
                jiwer.ExpandCommonEnglishContractions(),
                jiwer.RemoveKaldiNonWords(),
                jiwer.RemovePunctuation(),
            ]
        )
        text = jiwer_process(text)
        text = _remove_non_speech_elements(text)
        return normalize_whitespace(text)

    # "multilingual" uses multilingual normalization for non-English languages
    # and whisper normalization for English
    if mode == "multilingual":
        text = text.lower()
        lang = kwargs["lang"]
        if lang in [None, "en"]:
            text = EnglishTextNormalizer()(text)
        else:
            text = MultilingualTextNormalizer(
                remove_diacritics=kwargs.get("remove_diacritics", False),
                preserve_marks=kwargs.get("preserve_marks", False),
            )(text, lang=lang)
        return normalize_whitespace(text)


def _wer_with_counts(ref: str, hyp: str) -> dict[str, Any]:
    """Compute WER and return both the score and raw error/reference counts for corpus-level aggregation."""
    import jiwer

    wer_score = jiwer.wer(ref, hyp)
    measures = jiwer.process_words(ref, hyp)
    wer_errors = measures.substitutions + measures.deletions + measures.insertions
    wer_ref_words = measures.substitutions + measures.deletions + measures.hits

    return {
        "wer": wer_score,
        "wer_errors": wer_errors,
        "wer_ref_words": wer_ref_words,
        "wer_substitutions": measures.substitutions,
        "wer_insertions": measures.insertions,
        "wer_deletions": measures.deletions,
    }


def _cer_with_counts(ref: str, hyp: str, key_prefix: str = "cer") -> dict[str, Any]:
    """Compute CER and return both the score and raw error/reference counts for corpus-level aggregation."""
    import jiwer

    assert key_prefix in ["cer", "wer"], "key_prefix must be 'cer' or 'wer'"

    cer_score = jiwer.cer(ref, hyp)
    measures = jiwer.process_characters(ref, hyp)
    cer_errors = measures.substitutions + measures.deletions + measures.insertions
    cer_ref_chars = measures.substitutions + measures.deletions + measures.hits

    return {
        f"{key_prefix}": cer_score,
        f"{key_prefix}_errors": cer_errors,
        f"{key_prefix}_ref_words": cer_ref_chars,
        f"{key_prefix}_substitutions": measures.substitutions,
        f"{key_prefix}_insertions": measures.insertions,
        f"{key_prefix}_deletions": measures.deletions,
    }


def evaluate_asr(
    reference: str, hypothesis: str, normalization_mode: str = "standard", normalize_compound: bool = False, **kwargs
) -> dict[str, Any]:
    """Evaluate ASR: computes WER with normalization.

    Args:
        reference: Ground truth transcription.
        hypothesis: Model output transcription.
        normalization_mode: "standard", "audiobench", "hf_leaderboard", "none", "no_tn_itn", or
            "apptek_callcenter".
        normalize_compound: Whether to normalize compound pairs.
        **kwargs: Additional keyword arguments.
    """
    if normalization_mode == "apptek_callcenter":
        from nemo_skills.evaluation.evaluator.apptek_callcenter import normalize_apptek_callcenter_text

        ref = normalize_apptek_callcenter_text(reference, is_prediction=False)
        hyp = normalize_apptek_callcenter_text(hypothesis, is_prediction=True)

        if not ref:
            result = {"wer": None, "is_correct": None, "text": "", "pred_text": hyp or ""}
            return result

        result = _wer_with_counts(ref, hyp)
        result["is_correct"] = result["wer"] < 0.5
        result["text"] = ref
        result["pred_text"] = hyp
        return result

    ref = preprocess_asr_text(reference, mode=normalization_mode, **kwargs)
    hyp = preprocess_asr_text(hypothesis, mode=normalization_mode, **kwargs)

    if normalize_compound:
        ref, hyp = normalize_compound_pairs(ref, hyp)

    # Match the HF Open ASR Leaderboard: drop samples whose normalized
    # reference is empty rather than scoring them against a placeholder.
    if not ref:
        result = {"wer": None, "is_correct": None, "text": "", "pred_text": hyp or ""}
        return result

    if not hyp:
        hyp = "empty"

    result = _wer_with_counts(ref, hyp)
    result["is_correct"] = result["wer"] < 0.5
    result["text"] = ref
    result["pred_text"] = hyp
    return result


def _split_librispeechmix_lines(text: str) -> list[str]:
    """Split newline-delimited LibriSpeechMix generations into non-empty utterances."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def _extend_unique(items: list[str], new_items: list[str]) -> None:
    for item in new_items:
        if item not in items:
            items.append(item)


def _join_transcript_parts(parts: list[str]) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip()).strip()


_PROMPT_ARTIFACT_PATTERNS = [
    re.compile(r"these\s+audio\s+clips\s+are\s+enrollment\s+samples[^.?!]*(?:[.?!]|$)", re.IGNORECASE),
    re.compile(r"the\s+following\s+audio\s+clip\s+contains\s+the\s+speech\s+to\s+transcribe[^.?!]*(?:[.?!]|$)", re.IGNORECASE),
    re.compile(r"transcribe\s+this\s+audio\s+with\s+punctuation\s+and\s+capitalization[^.?!]*(?:[.?!]|$)", re.IGNORECASE),
]


def _strip_prompt_artifacts(text: str) -> tuple[str, list[str]]:
    """Remove known prompt echoes only as an explicit recovery path."""
    warnings: list[str] = []
    cleaned = text
    for pattern in _PROMPT_ARTIFACT_PATTERNS:
        cleaned, replacements = pattern.subn(" ", cleaned)
        if replacements:
            warnings.append("stripped_prompt_artifact")
    if warnings:
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned, warnings


def _counts_for_normalized_pair(reference: str, hypothesis: str) -> dict[str, int]:
    """Compute WER counts for a normalized reference/hypothesis pair, handling empty refs safely."""
    ref = reference.strip()
    hyp = hypothesis.strip()

    if not ref:
        insertions = len(hyp.split()) if hyp else 0
        return {
            "wer_errors": insertions,
            "wer_ref_words": 0,
            "wer_substitutions": 0,
            "wer_insertions": insertions,
            "wer_deletions": 0,
        }

    if not hyp:
        deletions = len(ref.split())
        return {
            "wer_errors": deletions,
            "wer_ref_words": deletions,
            "wer_substitutions": 0,
            "wer_insertions": 0,
            "wer_deletions": deletions,
        }

    return _wer_with_counts(ref, hyp)


def _pick_best_librispeechmix_alignment(references: list[str], hypotheses: list[str]) -> dict[str, Any]:
    """Choose the assignment with the fewest ASR errors.

    The number of references is tiny for LibriSpeechMix, but recovered model
    outputs can contain many speaker-tag fragments. This dynamic program is
    linear in the number of hypothesis fragments and exponential only in the
    number of reference streams.
    """
    references = list(references)
    hypotheses = list(hypotheses)
    ref_count = len(references)

    def empty_totals() -> dict[str, Any]:
        return {
            "wer_errors": 0,
            "wer_ref_words": 0,
            "wer_substitutions": 0,
            "wer_insertions": 0,
            "wer_deletions": 0,
        }

    def add_counts(total: dict[str, Any], counts: dict[str, int]) -> dict[str, Any]:
        out = dict(total)
        out["wer_errors"] += counts["wer_errors"]
        out["wer_ref_words"] += counts["wer_ref_words"]
        out["wer_substitutions"] += counts["wer_substitutions"]
        out["wer_insertions"] += counts["wer_insertions"]
        out["wer_deletions"] += counts["wer_deletions"]
        return out

    def score_key(state: dict[str, Any]) -> tuple[int, int, int, int]:
        return (
            state["wer_errors"],
            state["wer_insertions"],
            state["wer_deletions"],
            state["wer_substitutions"],
        )

    def maybe_keep(states: dict[int, dict[str, Any]], mask: int, candidate: dict[str, Any]) -> None:
        current = states.get(mask)
        if current is None or score_key(candidate) < score_key(current):
            states[mask] = candidate

    if ref_count == 0:
        state = empty_totals()
        for hypothesis in hypotheses:
            state = add_counts(state, _counts_for_normalized_pair("", hypothesis))
        state["aligned_references"] = [""] * len(hypotheses)
        state["aligned_hypotheses"] = list(hypotheses)
        state["wer"] = 0.0 if state["wer_errors"] == 0 else float("inf")
        state["is_correct"] = state["wer_errors"] == 0
        return state

    dp: dict[int, dict[str, Any]] = {0: {**empty_totals(), "assignments": [None] * ref_count, "extras": []}}
    for hyp_index, hypothesis in enumerate(hypotheses):
        next_dp: dict[int, dict[str, Any]] = {}
        extra_counts = _counts_for_normalized_pair("", hypothesis)
        for mask, state in dp.items():
            extra_state = add_counts(state, extra_counts)
            extra_state["assignments"] = list(state["assignments"])
            extra_state["extras"] = [*state["extras"], hyp_index]
            maybe_keep(next_dp, mask, extra_state)

            for ref_index, reference in enumerate(references):
                bit = 1 << ref_index
                if mask & bit:
                    continue
                pair_state = add_counts(state, _counts_for_normalized_pair(reference, hypothesis))
                assignments = list(state["assignments"])
                assignments[ref_index] = hyp_index
                pair_state["assignments"] = assignments
                pair_state["extras"] = list(state["extras"])
                maybe_keep(next_dp, mask | bit, pair_state)
        dp = next_dp

    best_result = None
    best_key = None
    for mask, state in dp.items():
        candidate = dict(state)
        assignments = list(candidate.pop("assignments"))
        extras = list(candidate.pop("extras"))
        for ref_index, reference in enumerate(references):
            if assignments[ref_index] is None:
                candidate = add_counts(candidate, _counts_for_normalized_pair(reference, ""))
        key = score_key(candidate)
        if best_key is None or key < best_key:
            best_key = key
            aligned_references = list(references)
            aligned_hypotheses = [hypotheses[index] if index is not None else "" for index in assignments]
            for hyp_index in extras:
                aligned_references.append("")
                aligned_hypotheses.append(hypotheses[hyp_index])
            candidate["aligned_references"] = aligned_references
            candidate["aligned_hypotheses"] = aligned_hypotheses
            best_result = candidate

    total_ref_words = best_result["wer_ref_words"]
    best_result["wer"] = best_result["wer_errors"] / total_ref_words if total_ref_words > 0 else 0.0
    best_result["is_correct"] = best_result["wer"] < 0.5 if total_ref_words > 0 else best_result["wer_errors"] == 0
    return best_result


def _prefix_librispeechmix_wer_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    fields = ["wer", "wer_errors", "wer_ref_words", "wer_substitutions", "wer_insertions", "wer_deletions"]
    return {f"{prefix}_{field}": metrics[field] for field in fields if field in metrics}


_ASR_RECOVERY_TAG_PATTERN = re.compile(
    r"(?:^|\s)(?:speaker[_\s-]*(\d+)\s*[:\-]|<spk:(\d+)>|\[(s\d+)\])",
    re.IGNORECASE,
)


def _parse_librispeechmix_asr_canonical(hypothesis: str) -> dict[str, Any]:
    """Parse base LibriSpeechMix ASR into canonical unlabeled streams."""
    raw_lines = _split_librispeechmix_lines(hypothesis)
    warnings: list[str] = []
    streams: list[str] = []
    recovered = False

    for line in raw_lines:
        cleaned_line, artifact_warnings = _strip_prompt_artifacts(line)
        if artifact_warnings:
            recovered = True
            _extend_unique(warnings, artifact_warnings)
        if not cleaned_line:
            continue

        matches = list(_ASR_RECOVERY_TAG_PATTERN.finditer(cleaned_line))
        if not matches:
            streams.append(cleaned_line)
            continue

        recovered = True
        prefix = cleaned_line[: matches[0].start()].strip()
        if prefix:
            streams.append(prefix)
            warnings.append("untagged_prefix_preserved")
        for idx, match in enumerate(matches):
            next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned_line)
            transcript = cleaned_line[match.end() : next_start].strip()
            if transcript:
                streams.append(transcript)

    if not streams and hypothesis.strip():
        warnings.append("empty_after_format_parse")
        status = "failed"
    elif not streams:
        status = "failed"
    else:
        status = "recovered" if recovered else "exact"

    return {
        "streams": streams,
        "format_parse_status": status,
        "format_warnings": warnings,
        "canonical_prediction": {"streams": streams},
        "speaker_mapping": {},
    }


def evaluate_librispeechmix_asr(
    references: list[str], hypothesis: str, normalization_mode: str = "standard"
) -> dict[str, Any]:
    """Evaluate LibriSpeechMix overlapped ASR with permutation-invariant WER."""
    normalized_refs = [preprocess_asr_text(reference, mode=normalization_mode) for reference in references]

    strict_hyp_streams = _split_librispeechmix_lines(hypothesis)
    if not strict_hyp_streams and hypothesis.strip():
        strict_hyp_streams = [hypothesis.strip()]
    normalized_strict_hyps = [preprocess_asr_text(line, mode=normalization_mode) for line in strict_hyp_streams]
    strict_result = _pick_best_librispeechmix_alignment(normalized_refs, normalized_strict_hyps)

    parsed = _parse_librispeechmix_asr_canonical(hypothesis)
    normalized_hyps = [preprocess_asr_text(line, mode=normalization_mode) for line in parsed["streams"]]
    canonical_result = _pick_best_librispeechmix_alignment(normalized_refs, normalized_hyps)

    result = dict(canonical_result)
    result.update(_prefix_librispeechmix_wer_metrics(strict_result, "strict"))
    result.update(_prefix_librispeechmix_wer_metrics(canonical_result, "canonical"))
    result.update(_prefix_librispeechmix_wer_metrics(canonical_result, "content"))
    result["reference_streams"] = normalized_refs
    result["predicted_streams"] = result.pop("aligned_hypotheses")
    result["aligned_reference_streams"] = result.pop("aligned_references")
    result["strict_predicted_streams"] = strict_result.get("aligned_hypotheses", [])
    result["strict_aligned_reference_streams"] = strict_result.get("aligned_references", [])
    result["format_parse_status"] = parsed["format_parse_status"]
    result["format_warnings"] = parsed["format_warnings"]
    result["canonical_prediction"] = parsed["canonical_prediction"]
    result["speaker_mapping"] = parsed["speaker_mapping"]
    return result


_SA_ASR_EXACT_LINE_PATTERN = re.compile(r"^\s*speaker_(\d+)\s*:\s*(.*)$", re.IGNORECASE)
_SA_ASR_SPEAKER_LINE_PATTERN = re.compile(r"^\s*speaker\s+(\d+)\s*[-:]\s*(.*)$", re.IGNORECASE)
_SA_ASR_TAG_PATTERN = re.compile(r"<spk:(\d+)>|\[(s\d+)\]", re.IGNORECASE)


def _sa_asr_speaker_label(tag_index: int, speaker_profile_index: list[int] | None) -> str:
    if speaker_profile_index is not None and 0 <= tag_index < len(speaker_profile_index):
        return f"speaker_{int(speaker_profile_index[tag_index])}"
    return f"speaker_{tag_index}"


def _append_sa_asr_prediction(parsed_predictions: dict[str, str], speaker_label: str, transcript: str) -> None:
    transcript = transcript.strip()
    if not transcript:
        return
    if speaker_label in parsed_predictions and parsed_predictions[speaker_label]:
        parsed_predictions[speaker_label] = f"{parsed_predictions[speaker_label]} {transcript}"
    else:
        parsed_predictions[speaker_label] = transcript


def _parse_sa_asr_strict_hypothesis(hypothesis: str) -> tuple[dict[str, str], list[str]]:
    """Parse only the original ``speaker_N: text`` contract, preserving duplicate penalties."""
    parsed_predictions: dict[str, str] = {}
    extras: list[str] = []

    for line in _split_librispeechmix_lines(hypothesis):
        match = _SA_ASR_EXACT_LINE_PATTERN.match(line)
        if match is not None:
            speaker_label = f"speaker_{int(match.group(1))}"
            transcript = match.group(2).strip()
            if speaker_label in parsed_predictions:
                extras.append(transcript)
            else:
                parsed_predictions[speaker_label] = transcript
        else:
            extras.append(line)

    return parsed_predictions, extras


def _parse_sa_asr_tagged_line(
    line: str, speaker_profile_index: list[int] | None
) -> tuple[dict[str, str], list[str], dict[str, str]]:
    parsed_predictions: dict[str, str] = {}
    extras: list[str] = []
    speaker_mapping: dict[str, str] = {}
    matches = list(_SA_ASR_TAG_PATTERN.finditer(line))
    if not matches:
        return parsed_predictions, [line], speaker_mapping

    prefix = line[: matches[0].start()].strip()
    if prefix:
        extras.append(prefix)

    for idx, match in enumerate(matches):
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(line)
        transcript = line[match.end() : next_start].strip()
        spk_tag, bracket_tag = match.groups()
        if spk_tag is not None:
            tag_index = int(spk_tag)
            raw_label = f"<spk:{tag_index}>"
        else:
            tag_index = int(bracket_tag[1:])
            raw_label = f"[{bracket_tag.lower()}]"
        speaker_label = _sa_asr_speaker_label(tag_index, speaker_profile_index)
        speaker_mapping[raw_label] = speaker_label
        _append_sa_asr_prediction(parsed_predictions, speaker_label, transcript)

    return parsed_predictions, extras, speaker_mapping


def _parse_sa_asr_canonical_hypothesis(
    hypothesis: str, speaker_profile_index: list[int] | None = None
) -> dict[str, Any]:
    """Parse SA-ASR output into canonical speaker-profile keyed streams."""
    parsed_predictions: dict[str, str] = {}
    extras: list[str] = []
    warnings: list[str] = []
    speaker_mapping: dict[str, str] = {}
    recovered = False
    saw_exact = False

    for line in _split_librispeechmix_lines(hypothesis):
        cleaned_line, artifact_warnings = _strip_prompt_artifacts(line)
        if artifact_warnings:
            recovered = True
            _extend_unique(warnings, artifact_warnings)
        if not cleaned_line:
            continue

        exact_match = _SA_ASR_EXACT_LINE_PATTERN.match(cleaned_line)
        if exact_match is not None:
            saw_exact = True
            speaker_label = f"speaker_{int(exact_match.group(1))}"
            _append_sa_asr_prediction(parsed_predictions, speaker_label, exact_match.group(2).strip())
            continue

        speaker_line_match = _SA_ASR_SPEAKER_LINE_PATTERN.match(cleaned_line)
        if speaker_line_match is not None:
            recovered = True
            speaker_label = f"speaker_{int(speaker_line_match.group(1))}"
            _append_sa_asr_prediction(parsed_predictions, speaker_label, speaker_line_match.group(2).strip())
            continue

        if _SA_ASR_TAG_PATTERN.search(cleaned_line):
            recovered = True
            tagged_predictions, tagged_extras, tagged_mapping = _parse_sa_asr_tagged_line(
                cleaned_line, speaker_profile_index
            )
            for speaker_label, transcript in tagged_predictions.items():
                _append_sa_asr_prediction(parsed_predictions, speaker_label, transcript)
            extras.extend(tagged_extras)
            speaker_mapping.update(tagged_mapping)
        else:
            extras.append(cleaned_line)

    if extras:
        recovered = True
        warnings.append("unlabeled_text_scored_as_insertion")

    if not parsed_predictions and hypothesis.strip():
        status = "failed"
    elif not parsed_predictions:
        status = "failed"
    elif recovered:
        status = "recovered"
    else:
        status = "exact" if saw_exact else "recovered"

    return {
        "streams": parsed_predictions,
        "extras": extras,
        "format_parse_status": status,
        "format_warnings": warnings,
        "canonical_prediction": {"streams": dict(sorted(parsed_predictions.items()))},
        "speaker_mapping": speaker_mapping,
    }


def _score_librispeechmix_sa_asr_map(
    reference_map: dict[str, str], normalized_predictions: dict[str, str], normalized_extras: list[str]
) -> dict[str, Any]:
    counts = {
        "wer_errors": 0,
        "wer_ref_words": 0,
        "wer_substitutions": 0,
        "wer_insertions": 0,
        "wer_deletions": 0,
    }
    aligned_predictions = {}
    remaining_predictions = dict(normalized_predictions)

    for speaker_label, reference in reference_map.items():
        hypothesis_text = remaining_predictions.pop(speaker_label, "")
        aligned_predictions[speaker_label] = hypothesis_text
        pair_counts = _counts_for_normalized_pair(reference, hypothesis_text)
        counts["wer_errors"] += pair_counts["wer_errors"]
        counts["wer_ref_words"] += pair_counts["wer_ref_words"]
        counts["wer_substitutions"] += pair_counts["wer_substitutions"]
        counts["wer_insertions"] += pair_counts["wer_insertions"]
        counts["wer_deletions"] += pair_counts["wer_deletions"]

    for hypothesis_text in list(remaining_predictions.values()) + normalized_extras:
        pair_counts = _counts_for_normalized_pair("", hypothesis_text)
        counts["wer_errors"] += pair_counts["wer_errors"]
        counts["wer_ref_words"] += pair_counts["wer_ref_words"]
        counts["wer_substitutions"] += pair_counts["wer_substitutions"]
        counts["wer_insertions"] += pair_counts["wer_insertions"]
        counts["wer_deletions"] += pair_counts["wer_deletions"]

    total_ref_words = counts["wer_ref_words"]
    counts["wer"] = counts["wer_errors"] / total_ref_words if total_ref_words > 0 else 0.0
    counts["is_correct"] = counts["wer"] < 0.5 if total_ref_words > 0 else counts["wer_errors"] == 0
    counts["aligned_predictions"] = aligned_predictions
    counts["remaining_predictions"] = remaining_predictions
    counts["extra_predictions"] = normalized_extras
    return counts


def evaluate_librispeechmix_sa_asr(
    references: list[str], speaker_profile_index: list[int], hypothesis: str, normalization_mode: str = "standard"
) -> dict[str, Any]:
    """Evaluate LibriSpeechMix speaker-attributed ASR using speaker-profile indices."""
    reference_map = {
        f"speaker_{speaker_idx}": preprocess_asr_text(reference, mode=normalization_mode)
        for speaker_idx, reference in zip(speaker_profile_index, references)
    }

    strict_predictions, strict_extras = _parse_sa_asr_strict_hypothesis(hypothesis)
    normalized_strict_predictions = {
        speaker_label: preprocess_asr_text(text, mode=normalization_mode)
        for speaker_label, text in strict_predictions.items()
    }
    normalized_strict_extras = [preprocess_asr_text(text, mode=normalization_mode) for text in strict_extras]
    strict_result = _score_librispeechmix_sa_asr_map(
        reference_map, normalized_strict_predictions, normalized_strict_extras
    )

    parsed = _parse_sa_asr_canonical_hypothesis(hypothesis, speaker_profile_index)
    normalized_predictions = {
        speaker_label: preprocess_asr_text(text, mode=normalization_mode)
        for speaker_label, text in parsed["streams"].items()
    }
    normalized_extras = [preprocess_asr_text(text, mode=normalization_mode) for text in parsed["extras"]]
    canonical_result = _score_librispeechmix_sa_asr_map(reference_map, normalized_predictions, normalized_extras)

    content_prediction_streams = list(normalized_predictions.values()) + normalized_extras
    content_result = _pick_best_librispeechmix_alignment(list(reference_map.values()), content_prediction_streams)

    result = dict(canonical_result)
    result.pop("remaining_predictions", None)
    result.pop("extra_predictions", None)
    result.update(_prefix_librispeechmix_wer_metrics(strict_result, "strict"))
    result.update(_prefix_librispeechmix_wer_metrics(canonical_result, "canonical"))
    result.update(_prefix_librispeechmix_wer_metrics(content_result, "content"))
    result["reference_streams"] = [reference_map[key] for key in sorted(reference_map)]
    result["predicted_streams"] = [canonical_result["aligned_predictions"].get(key, "") for key in sorted(reference_map)]
    result["strict_predicted_streams"] = [strict_result["aligned_predictions"].get(key, "") for key in sorted(reference_map)]
    result["content_predicted_streams"] = content_result.get("aligned_hypotheses", [])
    result["content_aligned_reference_streams"] = content_result.get("aligned_references", [])
    result["format_parse_status"] = parsed["format_parse_status"]
    result["format_warnings"] = parsed["format_warnings"]
    result["canonical_prediction"] = parsed["canonical_prediction"]
    result["speaker_mapping"] = parsed["speaker_mapping"]
    result["is_correct"] = canonical_result["is_correct"]
    return result


_BLEU_TOKENIZE_BY_LANG = {
    "ja": "ja-mecab",
    "zh": "zh",
    "cmn": "zh",
    "yue": "zh",
    "ko": "ko-mecab",
}


def resolve_bleu_tokenize(tgt_lang: str | None) -> str:
    """Resolve sacrebleu tokenize from a target language code."""
    if not isinstance(tgt_lang, str):
        return "13a"
    lang_code = tgt_lang.split("_")[0]
    return _BLEU_TOKENIZE_BY_LANG.get(lang_code, "13a")


def evaluate_translation(
    reference: str,
    hypothesis: str,
    tgt_lang: str | None = None,
) -> dict[str, Any]:
    """Evaluate translation: computes sentence-level BLEU score."""
    tokenize = resolve_bleu_tokenize(tgt_lang)
    try:
        import sacrebleu

        text = reference.strip()
        pred_text = hypothesis.strip()
        bleu = sacrebleu.sentence_bleu(pred_text, [text], tokenize=tokenize)
        bleu_score = bleu.score / 100.0

        return {
            "bleu": bleu_score,
            "is_correct": bleu_score > 0.3,
            "text": text,
            "pred_text": pred_text,
            "bleu_tokenize": tokenize,
        }
    except Exception as e:
        return {
            "bleu": 0.0,
            "is_correct": False,
            "error": str(e),
            "text": reference.strip(),
            "pred_text": hypothesis.strip(),
            "bleu_tokenize": tokenize,
        }


def evaluate_cer(
    reference: str,
    hypothesis: str,
    normalization_mode: str = "none",
    key_prefix: str = "cer",
    normalize_compound: bool = False,
    **kwargs,
) -> dict[str, Any]:
    """Evaluate CER: character-level edit distance."""

    ref = preprocess_asr_text(reference, mode=normalization_mode, **kwargs)
    hyp = preprocess_asr_text(hypothesis, mode=normalization_mode, **kwargs)

    if normalize_compound:
        ref, hyp = normalize_compound_pairs(ref, hyp)

    # Mirror evaluate_asr: drop samples whose normalized reference is empty rather
    # than dividing by zero in the character edit distance.
    if not ref:
        return {key_prefix: None, "is_correct": None, "text": "", "pred_text": hyp or ""}

    result = _cer_with_counts(ref, hyp, key_prefix=key_prefix)
    result["is_correct"] = result[key_prefix] < 0.5
    result["text"] = ref
    result["pred_text"] = hyp
    return result


def evaluate_mer(
    reference: str,
    hypothesis: str,
    normalization_mode: str = "none",
    normalize_compound: bool = False,
    grapheme: bool = False,
    **kwargs,
) -> dict[str, Any]:
    """Evaluate Mixed Error Rate for code-switched ASR.

    Scriptio-continua scripts are counted by character and space-delimited
    scripts by word within the same utterance (see ``mixed_segment``), then
    scored with the standard word-level edit distance. Results are reported
    under the ``wer*`` keys so the mixed figure occupies the headline WER column
    (matching how ``Multilingual-ASR`` folds CER into ``wer*``).
    """
    ref = preprocess_asr_text(reference, mode=normalization_mode, **kwargs)
    hyp = preprocess_asr_text(hypothesis, mode=normalization_mode, **kwargs)

    if normalize_compound:
        ref, hyp = normalize_compound_pairs(ref, hyp)

    # Mirror evaluate_asr/evaluate_cer: drop samples whose normalized reference
    # is empty rather than scoring against a placeholder.
    if not ref:
        return {"wer": None, "is_correct": None, "text": "", "pred_text": hyp or ""}

    if not hyp:
        hyp = "empty"

    ref_seg = " ".join(mixed_segment(ref, grapheme=grapheme))
    hyp_seg = " ".join(mixed_segment(hyp, grapheme=grapheme))

    result = _wer_with_counts(ref_seg, hyp_seg)
    result["is_correct"] = result["wer"] < 0.5
    # Store the human-readable (unsegmented) normalized text, not the spaced tokens.
    result["text"] = ref
    result["pred_text"] = hyp
    return result


def evaluate_hallucination(reference: str, hypothesis: str, audio_context: dict = None) -> dict[str, Any]:
    """Detect potential hallucinations via speaking rate anomaly.

    Normal speech: ~600-900 chars/minute. Higher rates suggest repetition/hallucination.
    Requires audio_duration in audio_context.
    """
    audio_duration = audio_context.get("audio_duration") if audio_context else None

    if not audio_duration or audio_duration <= 0:
        return {
            "hallucination_rate": 0.0,
            "char_rate": 0.0,
            "is_correct": True,
            "error": "missing_audio_duration",
            "text": reference,
            "pred_text": hypothesis,
        }

    char_count = len(hypothesis)
    # Convert to chars/minute
    char_rate = (char_count / audio_duration) * 60.0

    # Hallucination threshold: >1500 chars/min (25 chars/second * 60)
    is_hallucinating = char_rate > 1500.0

    return {
        "hallucination_rate": 1.0 if is_hallucinating else 0.0,
        "char_rate": round(char_rate, 2),
        "is_correct": not is_hallucinating,
        "text": reference,
        "pred_text": hypothesis,
    }


def evaluate_pc_rate(reference: str, hypothesis: str) -> dict[str, Any]:
    """Evaluate detailed Punctuation and Capitalization metrics."""
    # Extract punctuation with positions
    ref_puncts = [(m.group(), m.start()) for m in re.finditer(r"[.,!?;:\-]", reference)]
    hyp_puncts = [(m.group(), m.start()) for m in re.finditer(r"[.,!?;:\-]", hypothesis)]

    # Punctuation matching (within 2 char tolerance)
    matched = 0
    for ref_p, ref_pos in ref_puncts:
        for hyp_p, hyp_pos in hyp_puncts:
            if ref_p == hyp_p and abs(ref_pos - hyp_pos) <= 2:
                matched += 1
                break

    punct_precision = matched / len(hyp_puncts) if hyp_puncts else 0.0
    punct_recall = matched / len(ref_puncts) if ref_puncts else 0.0
    punct_f1 = (
        2 * punct_precision * punct_recall / (punct_precision + punct_recall)
        if (punct_precision + punct_recall) > 0
        else 0.0
    )

    # Capitalization: check sentence starts and word capitals
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if len(ref_words) != len(hyp_words):
        cap_accuracy = 0.0
    else:
        cap_matches = sum(
            1 for r, h in zip(ref_words, hyp_words, strict=True) if r and h and r[0].isupper() == h[0].isupper()
        )
        cap_accuracy = cap_matches / len(ref_words) if ref_words else 0.0

    # Overall PC rate (average of punct F1 and cap accuracy)
    pc_rate = (punct_f1 + cap_accuracy) / 2.0

    return {
        "pc_rate": round(pc_rate, 3),
        "punct_precision": round(punct_precision, 3),
        "punct_recall": round(punct_recall, 3),
        "punct_f1": round(punct_f1, 3),
        "cap_accuracy": round(cap_accuracy, 3),
        "is_correct": pc_rate > 0.5,
        "text": reference,
        "pred_text": hypothesis,
    }


class AudioEvaluator(BaseEvaluator):
    """Audio evaluator supporting ASR, ASR-PC, Translation, CER, etc."""

    def __init__(self, config: dict, num_parallel_requests=10):
        super().__init__(config, num_parallel_requests)
        self.eval_config = AudioEvaluatorConfig(**self.config)

    async def eval_single(self, data_point: dict[str, any]) -> dict[str, any]:
        """Evaluate single audio sample - can be called during generation.

        Returns dict of updates to be merged into data_point by BaseEvaluator.
        """
        return evaluate_sample(data_point, self.eval_config)


def eval_audio(cfg):
    """Function wrapper for backward compatibility."""
    evaluator = AudioEvaluator(cfg)
    asyncio.run(evaluator.eval_full())


def coerce_text_field(value: Any) -> str:
    """Convert loose AudioBench text/reference fields to plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("text", "expected_answer", "answer", "transcript", "reference"):
            if key in value and value[key] is not None:
                return coerce_text_field(value[key])
        return " ".join(coerce_text_field(item) for item in value.values() if item is not None).strip()
    if isinstance(value, (list, tuple)):
        return " ".join(coerce_text_field(item) for item in value if item is not None).strip()
    return str(value).strip()


def evaluate_sample(sample: dict[str, Any], config: AudioEvaluatorConfig) -> dict[str, Any]:
    """Evaluate single sample based on task_type. Returns dict of updates to merge."""
    updates = {}
    task_type = sample.get("task_type", "unknown")
    generation = coerce_text_field(sample.get("generation", ""))
    expected_answer = coerce_text_field(sample.get("expected_answer", ""))

    # Extract ASR text from generation
    # E.g Qwen ASR uses <asr_text> tags to indicate the ASR text
    if task_type in TASKS_NEED_PARSING:
        generation = extract_asr_text(generation)

    # Strip helpful prefixes for ASR tasks (e.g., "The audio says: ...")
    if config.strip_helpful_prefixes:
        generation = strip_helpful_prefixes(generation)

    if task_type == "ASR-PC":
        mode = resolve_asr_normalization_mode(config)
        metrics = evaluate_asr_pc(
            expected_answer,
            generation,
            normalize_standard_wer=config.normalize_asr_pc_standard_wer,
            normalization_mode=mode,
        )
        updates.update(metrics)

    elif task_type == "ASR":
        mode = resolve_asr_normalization_mode(config)
        metrics = evaluate_asr(expected_answer, generation, normalization_mode=mode)
        updates.update(metrics)
        updates["predicted_answer"] = generation

    elif task_type == "LIBRISPEECHMIX_ASR":
        mode = resolve_asr_normalization_mode(config)
        metrics = evaluate_librispeechmix_asr(sample["reference_streams"], generation, normalization_mode=mode)
        updates.update(metrics)
        updates["predicted_answer"] = generation

    elif task_type == "LIBRISPEECHMIX_SA_ASR":
        mode = resolve_asr_normalization_mode(config)
        metrics = evaluate_librispeechmix_sa_asr(
            sample["reference_streams"],
            sample["speaker_profile_index"],
            generation,
            normalization_mode=mode,
        )
        updates.update(metrics)
        updates["predicted_answer"] = generation

    elif task_type == "ASR_LEADERBOARD":
        mode = resolve_asr_normalization_mode(config)
        metrics = evaluate_asr(expected_answer, generation, normalization_mode=mode)
        updates.update(metrics)

        # Additional WER calculation for specified reference fields
        if config.reference_fields:
            for ref_field in config.reference_fields:
                ref_value = sample[ref_field]  # fail if field is missing - user-specified fields must exist
                # Compute WER against this reference field
                ref_metrics = evaluate_asr(ref_value, generation, normalization_mode=mode)
                # Derive metric name from field name (e.g., "text_tn" -> "wer_tn")
                metric_suffix = ref_field.replace("text_", "") if ref_field.startswith("text_") else ref_field
                updates[f"wer_{metric_suffix}"] = ref_metrics["wer"]
                updates[f"is_correct_{metric_suffix}"] = ref_metrics["is_correct"]

    elif task_type in ["AST", "Translation", "Multilingual-AST"]:
        extra_fields = sample.get("extra_fields", {})
        tgt_lang = extra_fields.get("tgt_lang", None)
        metrics = evaluate_translation(expected_answer, generation, tgt_lang)
        updates.update(metrics)

    elif task_type == "Multilingual-ASR":
        mode = resolve_asr_normalization_mode(config)
        extra_fields = sample.get("extra_fields", {})
        use_cer = extra_fields.get("use_cer", False)
        src_lang = extra_fields.get("src_lang", None)
        if src_lang is not None and "_" in src_lang:
            src_lang = src_lang.split("_")[0]
        preprocess_kwargs = {
            "lang": src_lang,
            "remove_diacritics": True,
        }
        # For code-switched data (use_mer, i.e. cs-fleurs), preserve combining marks in
        # the multilingual normalization so MER counts Thai/Myanmar vowels/tones (and so
        # grapheme-cluster segmentation is meaningful). num2words and other steps still
        # apply. Monolingual benchmarks keep the existing mark-stripping behavior.
        if extra_fields.get("use_mer", False):
            preprocess_kwargs["preserve_marks"] = True
        # Only normalize compound pairs for non-English languages
        normalize_compound = src_lang not in [None, "en"]

        # Mixed WER column: CJK-style languages (use_cer) fold their CER into the WER
        # column (stored under wer* keys), all other languages contribute standard WER.
        # Code-switched samples (use_mer) instead get a true Mixed Error Rate, counting
        # scriptio-continua scripts by character and space-delimited scripts by word in
        # the same utterance; also stored under the wer* column. Monolingual benchmarks
        # (e.g. fleurs) never set use_mer, so their use_cer/WER behavior is unchanged.
        if extra_fields.get("use_mer", False):
            metrics = evaluate_mer(
                expected_answer,
                generation,
                normalization_mode=mode,
                normalize_compound=normalize_compound,
                grapheme=extra_fields.get("mer_grapheme", False),
                **preprocess_kwargs,
            )
        elif use_cer:
            metrics = evaluate_cer(
                expected_answer,
                generation,
                normalization_mode=mode,
                key_prefix="wer",  # fold CER into the mixed WER column
                normalize_compound=normalize_compound,
                **preprocess_kwargs,
            )
        else:
            metrics = evaluate_asr(
                expected_answer,
                generation,
                normalization_mode=mode,
                normalize_compound=normalize_compound,
                **preprocess_kwargs,
            )
        updates.update(metrics)

        # Parallel only-CER column: character error rate computed for every sample,
        # reported separately (cer* keys) alongside the mixed WER column.
        # Code-switched benchmarks (use_mer, i.e. cs-fleurs) score this CER with
        # mark-preserving normalization (lowercase + unpunctuated, combining marks
        # kept) so it matches the CS-FLEURS paper's CER; monolingual benchmarks keep
        # the standard multilingual normalization unchanged.
        cer_mode = "lower_nopunct" if extra_fields.get("use_mer", False) else mode
        cer_metrics = evaluate_cer(
            expected_answer,
            generation,
            normalization_mode=cer_mode,
            key_prefix="cer",
            normalize_compound=normalize_compound,
            **preprocess_kwargs,
        )
        # Merge only cer* keys so the mixed metric's is_correct/text/pred_text are preserved.
        for cer_key in (
            "cer",
            "cer_errors",
            "cer_ref_words",
            "cer_substitutions",
            "cer_insertions",
            "cer_deletions",
        ):
            if cer_key in cer_metrics:
                updates[cer_key] = cer_metrics[cer_key]

    elif task_type == "CER":
        metrics = evaluate_cer(expected_answer, generation, normalization_mode="none", key_prefix="cer")
        updates.update(metrics)

    elif task_type == "Hallucination":
        audio_context = {"audio_duration": sample.get("audio_duration")}
        metrics = evaluate_hallucination(expected_answer, generation, audio_context)
        updates.update(metrics)

    elif task_type == "PC-Rate":
        metrics = evaluate_pc_rate(expected_answer, generation)
        updates.update(metrics)

    else:
        if "requires_judge" not in sample:
            updates["requires_judge"] = True
        if "is_correct" not in sample:
            updates["is_correct"] = False

    audio_duration = sample.get("audio_duration", None)
    if audio_duration and audio_duration > 0 and expected_answer and generation:
        # chars/minute (chars/second * 60)
        updates["ref_char_rate"] = (len(expected_answer) / audio_duration) * 60.0
        updates["hyp_char_rate"] = (len(generation) / audio_duration) * 60.0
        updates["char_rate_diff"] = abs(updates["hyp_char_rate"] - updates["ref_char_rate"])

    return updates
