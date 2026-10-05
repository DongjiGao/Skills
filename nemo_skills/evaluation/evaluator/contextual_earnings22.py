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

"""Contextual Earnings-22 evaluator: WER + keyword Precision / Recall / F1.

Scoring follows Argmax OpenBench (``earnings22-keywords``), which publishes the
reference results for this benchmark:

- WER: OpenBench's English text normalizer (vendored in
  ``openbench_normalizer``; expands contractions, removes fillers, converts
  spoken numbers to digits, standardizes spelling) applied to the original
  cased reference and the hypothesis, scored with ``jiwer``.
- Keywords: OpenBench's keyword algorithm. Reference, hypothesis and keywords
  are normalized with Whisper's ``BasicTextNormalizer``; a keyword n-gram in the
  reference is a TP when the aligned hypothesis words match it, and a keyword
  n-gram in the hypothesis that is not aligned to the same reference words is
  an FP.

The legacy scoring below (``normalize_text`` + ``calculate_wer_with_alignment``)
lowercases and strips punctuation only, and scores against the dataset's
``norm_text``, which drops digits from the reference; it is kept for reference
and reported under ``legacy_*`` keys.

Legacy metric definition, after the Contextual Earnings-22 paper (Argmax,
2025, Section 3):

- WER: standard word error rate computed with edit-distance DP.
- Keyword TP/FP/FN: a keyword is a True Positive iff it matches the
  reference text AND its alignment position (computed via minimum edit
  distance) maps to identical hypothesis tokens. Otherwise it is an FN
  (in reference but not hypothesis at the aligned position) or an FP
  (in hypothesis but not reference at that position).

Per-sample raw counts are accumulated to corpus-level micro
precision / recall / F1 in ``ContextualEarnings22Metrics``.
"""

import re

from nemo_skills.evaluation.evaluator.base import BaseEvaluator, BaseEvaluatorConfig
from nemo_skills.utils import nested_dataclass

_OP_MATCH = "M"
_OP_SUB = "S"
_OP_INS = "I"
_OP_DEL = "D"


def normalize_text(text: str) -> str:
    """Mirror the ``norm_text`` field's normalization.

    Lowercases, replaces any non-letter/non-digit/non-whitespace character
    (and underscores) with a space, then collapses whitespace. Preserves
    Unicode letters via ``re.UNICODE``.

    Examples::

        "Q&A session"     -> "q a session"
        "Hyoung-il"       -> "hyoung il"
        "we'll"           -> "we ll"
        "100% of"         -> "100 of"
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def calculate_wer_with_alignment(hyp_tokens: list[str], ref_tokens: list[str]):
    """Word-level edit distance + traceback.

    Returns ``(wer, errors, ref_to_hyp)`` where:

    - ``wer``: insertions + deletions + substitutions, divided by ``len(ref_tokens)``
      (or by ``len(hyp_tokens)`` when the reference is empty, matching the paper's
      degenerate case behavior).
    - ``errors``: total raw error count.
    - ``ref_to_hyp``: list of length ``len(ref_tokens)``. Each entry is either
      ``(op, hyp_index)`` where ``op`` is ``"M"`` (match) or ``"S"``
      (substitution) and ``hyp_index`` is the 0-based hypothesis index aligned
      to that reference token; or ``("D", None)`` for a deletion (the reference
      token has no hypothesis counterpart).
    """
    n_hyp = len(hyp_tokens)
    n_ref = len(ref_tokens)

    if n_ref == 0:
        wer = 0.0 if n_hyp == 0 else float(n_hyp)
        return wer, n_hyp, []

    dp = [[0] * (n_ref + 1) for _ in range(n_hyp + 1)]
    bt = [[None] * (n_ref + 1) for _ in range(n_hyp + 1)]

    for i in range(1, n_hyp + 1):
        dp[i][0] = i
        bt[i][0] = _OP_INS
    for j in range(1, n_ref + 1):
        dp[0][j] = j
        bt[0][j] = _OP_DEL

    for i in range(1, n_hyp + 1):
        for j in range(1, n_ref + 1):
            if hyp_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                bt[i][j] = _OP_MATCH
                continue
            sub = dp[i - 1][j - 1] + 1
            ins = dp[i - 1][j] + 1
            dele = dp[i][j - 1] + 1
            best = min(sub, ins, dele)
            dp[i][j] = best
            if best == sub:
                bt[i][j] = _OP_SUB
            elif best == dele:
                bt[i][j] = _OP_DEL
            else:
                bt[i][j] = _OP_INS

    ref_to_hyp: list[tuple[str, int | None]] = [None] * n_ref  # type: ignore[list-item]
    i, j = n_hyp, n_ref
    while i > 0 or j > 0:
        op = bt[i][j]
        if op == _OP_MATCH:
            ref_to_hyp[j - 1] = (_OP_MATCH, i - 1)
            i -= 1
            j -= 1
        elif op == _OP_SUB:
            ref_to_hyp[j - 1] = (_OP_SUB, i - 1)
            i -= 1
            j -= 1
        elif op == _OP_DEL:
            ref_to_hyp[j - 1] = (_OP_DEL, None)
            j -= 1
        elif op == _OP_INS:
            i -= 1
        else:
            break

    errors = dp[n_hyp][n_ref]
    wer = errors / n_ref
    return wer, errors, ref_to_hyp


def _find_subseq_occurrences(tokens: list[str], pattern: list[str]) -> list[int]:
    """Return all start indices of non-overlapping occurrences of ``pattern`` in ``tokens``.

    Greedy left-to-right scan; once a match is consumed, the next search
    resumes after its end. Returns an empty list if ``pattern`` is empty
    or longer than ``tokens``.
    """
    n, m = len(tokens), len(pattern)
    if m == 0 or m > n:
        return []
    out = []
    i = 0
    while i <= n - m:
        if tokens[i : i + m] == pattern:
            out.append(i)
            i += m
        else:
            i += 1
    return out


def _legacy_scores(data_point: dict) -> dict:
    """Evaluate a single Contextual Earnings-22 sample.

    Returns per-sample metrics with raw counts for corpus-level aggregation:
    ``wer``, ``wer_errors``, ``wer_ref_words``, ``keyword_tp``, ``keyword_fp``,
    ``keyword_fn``, ``keyword_total_ref``, ``is_correct`` (= ``wer < 0.5``),
    plus the normalized ``text`` / ``pred_text`` for inspection.
    """
    reference = data_point["expected_answer"]
    generation = data_point["generation"].strip()
    keyword_list = data_point["keyword_list"]

    norm_ref = normalize_text(reference)
    norm_hyp = normalize_text(generation)

    ref_tokens = norm_ref.split()
    hyp_tokens = norm_hyp.split()

    wer, wer_errors, ref_to_hyp = calculate_wer_with_alignment(hyp_tokens, ref_tokens)
    wer_ref_words = len(ref_tokens)

    norm_keywords = []
    for kw in keyword_list:
        nk = normalize_text(kw)
        if nk:
            norm_keywords.append(nk.split())

    tp = 0
    fp = 0
    fn = 0
    total_ref_keyword_instances = 0

    matched_hyp_spans: set[tuple[int, int]] = set()

    for kw_tokens in norm_keywords:
        kw_len = len(kw_tokens)
        ref_starts = _find_subseq_occurrences(ref_tokens, kw_tokens)
        total_ref_keyword_instances += len(ref_starts)

        for ref_start in ref_starts:
            aligned_hyp = []
            exact = True
            for k in range(kw_len):
                op_info = ref_to_hyp[ref_start + k]
                if op_info is None:
                    exact = False
                    break
                op, hyp_idx = op_info
                if op != _OP_MATCH or hyp_idx is None:
                    exact = False
                    break
                aligned_hyp.append(hyp_idx)

            if exact and len(aligned_hyp) == kw_len:
                hyp_start = aligned_hyp[0]
                contiguous = all(
                    aligned_hyp[k] == hyp_start + k and hyp_tokens[hyp_start + k] == kw_tokens[k]
                    for k in range(kw_len)
                )
                if contiguous:
                    tp += 1
                    matched_hyp_spans.add((hyp_start, hyp_start + kw_len))
                    continue
            fn += 1

        hyp_starts = _find_subseq_occurrences(hyp_tokens, kw_tokens)
        hyp_unmatched = sum(1 for hs in hyp_starts if (hs, hs + kw_len) not in matched_hyp_spans)
        fp += hyp_unmatched

    return {
        "wer": wer,
        "wer_errors": wer_errors,
        "wer_ref_words": wer_ref_words,
        "keyword_tp": tp,
        "keyword_fp": fp,
        "keyword_fn": fn,
        "keyword_total_ref": total_ref_keyword_instances,
        "is_correct": wer < 0.5,
        "text": norm_ref,
        "pred_text": norm_hyp,
    }


_EPS = "<eps>"
_OPENBENCH_NORMALIZER = None
_BASIC_NORMALIZER = None


def _openbench_normalize(text: str) -> list[str]:
    global _OPENBENCH_NORMALIZER
    if _OPENBENCH_NORMALIZER is None:
        from nemo_skills.evaluation.evaluator.openbench_normalizer import EnglishTextNormalizer

        _OPENBENCH_NORMALIZER = EnglishTextNormalizer(None)
    return _OPENBENCH_NORMALIZER.process_transcript(text).split()


def _basic_normalize(text: str) -> str:
    global _BASIC_NORMALIZER
    if _BASIC_NORMALIZER is None:
        from transformers.models.whisper.english_normalizer import BasicTextNormalizer

        _BASIC_NORMALIZER = BasicTextNormalizer()
    return _BASIC_NORMALIZER(text)


def align_words(ref: list[str], hyp: list[str]) -> list[tuple[str, str]]:
    """Word alignment of two sequences as ``(ref, hyp)`` pairs, eps-padded.

    Uses ``texterrors.align_texts`` (as OpenBench does) when available, which
    makes the keyword metric identical to OpenBench. Otherwise falls back to a
    plain Levenshtein alignment, whose tie-breaking can differ on a few
    utterances.
    """
    try:
        import texterrors
    except ImportError:
        texterrors = None
    if texterrors is not None:
        aligned_ref, aligned_hyp = texterrors.align_texts(ref, hyp)[:2]
        return list(zip(aligned_ref, aligned_hyp))
    return _levenshtein_align(ref, hyp)


def _levenshtein_align(ref: list[str], hyp: list[str]) -> list[tuple[str, str]]:
    """Fallback alignment; ties prefer matches/substitutions, then deletions."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1)
            dp[i][j] = min(diag, dp[i - 1][j] + 1, dp[i][j - 1] + 1)
    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            pairs.append((ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            pairs.append((ref[i - 1], _EPS))
            i -= 1
        else:
            pairs.append((_EPS, hyp[j - 1]))
            j -= 1
    return pairs[::-1]


def _ngrams_with_positions(ali: list[tuple[str, str]], side: int, order: int):
    """Yield sliding n-grams over the non-eps words of one side of an alignment.

    Mirrors OpenBench: windows advance by one word, and each item carries the
    alignment index of its words so the other side can be read off.
    """
    idx = 0
    items: list[tuple[str, int]] = []
    while idx < len(ali):
        if items:
            items = [items[1]] if order > 1 else []
            idx = items[0][1] + 1 if items else idx
        while len(items) != order and idx < len(ali):
            word = ali[idx][side]
            idx += 1
            if word != _EPS:
                items.append((word, idx - 1))
        if len(items) == order:
            yield items


def openbench_keyword_stats(reference: str, hypothesis: str, keywords: list[str]) -> tuple[int, int, int]:
    """Return ``(tp, ground_truth, fp)`` following OpenBench's keyword metric."""
    ref_words = _basic_normalize(reference).split()
    hyp_words = _basic_normalize(hypothesis).split()
    norm_keywords = [_basic_normalize(kw) for kw in keywords]
    norm_keywords = [kw for kw in norm_keywords if kw.strip()]
    if not norm_keywords:
        return 0, 0, 0
    stats = {kw: [0, 0, 0] for kw in norm_keywords}  # [tp, gt, fp]
    ali = align_words(ref_words, hyp_words)

    for word_ref, word_hyp in ali:
        if word_ref in stats:
            stats[word_ref][1] += 1
            if word_ref == word_hyp:
                stats[word_ref][0] += 1
        elif word_hyp in stats:
            stats[word_hyp][2] += 1

    max_order = max(len(kw.split()) for kw in norm_keywords)
    for order in range(2, max_order + 1):
        for items in _ngrams_with_positions(ali, 0, order):
            phrase_ref = " ".join(w for w, _ in items)
            phrase_hyp = " ".join(ali[k][1] for _, k in items)
            if phrase_ref in stats:
                stats[phrase_ref][1] += 1
                if phrase_ref == phrase_hyp:
                    stats[phrase_ref][0] += 1
        for items in _ngrams_with_positions(ali, 1, order):
            phrase_hyp = " ".join(w for w, _ in items)
            phrase_ref = " ".join(ali[k][0] for _, k in items)
            if phrase_hyp in stats and phrase_hyp != phrase_ref:
                stats[phrase_hyp][2] += 1

    tp = sum(v[0] for v in stats.values())
    gt = sum(v[1] for v in stats.values())
    fp = sum(v[2] for v in stats.values())
    return tp, gt, fp


def evaluate_contextual_earnings22_sample(data_point: dict) -> dict:
    """Evaluate a single Contextual Earnings-22 sample with OpenBench scoring.

    Returns raw counts for corpus-level aggregation (``wer_errors``,
    ``wer_ref_words``, ``keyword_tp``, ``keyword_fp``, ``keyword_fn``,
    ``keyword_total_ref``), the per-sample ``wer``, ``is_correct``
    (= ``wer < 0.5``), the normalized ``text`` / ``pred_text``, and the legacy
    scores under ``legacy_*`` keys.
    """
    import jiwer

    # OpenBench scores against the word-level ``transcript`` field. Older
    # prepared data only carries ``text`` (cut at character boundaries) and the
    # digit-stripped ``norm_text``.
    reference = (
        data_point.get("reference_transcript")
        or data_point.get("expected_answer_cased")
        or data_point["expected_answer"]
    )
    generation = (data_point.get("generation") or "").strip()
    keywords = data_point["keyword_list"]

    ref_tokens = _openbench_normalize(reference)
    hyp_tokens = _openbench_normalize(generation)
    if ref_tokens:
        out = jiwer.process_words(" ".join(ref_tokens), " ".join(hyp_tokens) if hyp_tokens else "")
        wer_errors = out.substitutions + out.deletions + out.insertions
    else:
        wer_errors = len(hyp_tokens)
    wer_ref_words = len(ref_tokens)
    wer = wer_errors / wer_ref_words if wer_ref_words else float(wer_errors)

    tp, gt, fp = openbench_keyword_stats(reference, generation, keywords)

    legacy = _legacy_scores(data_point)
    result = {
        "wer": wer,
        "wer_errors": wer_errors,
        "wer_ref_words": wer_ref_words,
        "keyword_tp": tp,
        "keyword_fp": fp,
        "keyword_fn": gt - tp,
        "keyword_total_ref": gt,
        "is_correct": wer < 0.5,
        "text": " ".join(ref_tokens),
        "pred_text": " ".join(hyp_tokens),
        "scoring": "openbench-v1",
    }
    for key in ("wer", "wer_errors", "wer_ref_words", "keyword_tp", "keyword_fp", "keyword_fn", "keyword_total_ref"):
        result[f"legacy_{key}"] = legacy[key]
    return result


@nested_dataclass(kw_only=True)
class ContextualEarnings22EvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for Contextual Earnings-22 evaluation."""

    pass


class ContextualEarnings22Evaluator(BaseEvaluator):
    """Evaluator for Contextual Earnings-22: WER + keyword P/R/F1."""

    def __init__(self, config: dict, num_parallel_requests: int = 10):
        """Initialize with evaluator config and parallelism settings."""
        super().__init__(config, num_parallel_requests)

    async def eval_single(self, data_point: dict) -> dict:
        """Evaluate a single sample, returning WER + keyword TP/FP/FN counts."""
        return evaluate_contextual_earnings22_sample(data_point)
