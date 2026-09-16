# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for the Mixed Error Rate (MER) tokenizer used by code-switched ASR.

``mixed_segment`` is pure ``re``/``unicodedata`` and runs without optional deps.
The ``evaluate_mer`` tests need ``jiwer`` (and the normalization stack), so they
``importorskip`` and run in the GPU/CI image rather than the bare sandbox.
"""

import pytest

from nemo_skills.evaluation.evaluator.audio import (
    mixed_segment,
    normalize_cer_whitespace,
    resolve_cer_whitespace_mode,
    resolve_sample_cer_whitespace_mode,
)


class TestMixedSegmentCoverage:
    """Each scriptio-continua script must split into individual units, while
    Latin runs stay as whitespace-delimited words."""

    def test_latin_only_unchanged(self):
        assert mixed_segment("hello world") == ["hello", "world"]

    def test_empty(self):
        assert mixed_segment("") == []

    def test_mandarin_chars_split_english_words_kept(self):
        # The core MER property for cmn-eng.
        assert mixed_segment("我 like 猫") == ["我", "like", "猫"]

    def test_mandarin_without_spaces_still_splits(self):
        assert mixed_segment("报道称") == ["报", "道", "称"]

    def test_han_extensions_split(self):
        assert mixed_segment("𨋢𠮶") == ["𨋢", "𠮶"]

    def test_japanese_kana_split(self):
        # Naive CJK-only regexes miss kana; these must each be their own token.
        assert mixed_segment("サイエンス") == ["サ", "イ", "エ", "ン", "ス"]
        assert mixed_segment("もの") == ["も", "の"]

    def test_korean_hangul_split(self):
        assert mixed_segment("알려진") == ["알", "려", "진"]

    def test_thai_codepoint_mode_splits_every_codepoint(self):
        # Thai is not CJK and has no spaces; codepoint mode yields one token per
        # codepoint (base + each combining mark), consistent with jiwer CER.
        assert mixed_segment("ที่") == ["ท", "ี", "่"]

    def test_myanmar_codepoint_mode_splits_every_codepoint(self):
        assert mixed_segment("ကျော်") == ["က", "ျ", "ေ", "ာ", "်"]

    def test_mixed_scripts_in_one_utterance(self):
        assert mixed_segment("AI は computer の科目") == ["AI", "は", "computer", "の", "科", "目"]


class TestMixedSegmentGrapheme:
    """Grapheme mode keeps combining marks attached to their base; it matters
    only for the combining-mark scripts (Thai/Myanmar)."""

    def test_thai_grapheme_keeps_marks_with_base(self):
        # base ท + vowel ี + tone ่ -> a single perceived character.
        assert mixed_segment("ที่", grapheme=True) == ["ที่"]

    def test_myanmar_grapheme_keeps_stack_together(self):
        assert mixed_segment("ကျော်", grapheme=True) == ["ကျော်"]

    def test_grapheme_is_noop_for_han_kana_hangul(self):
        # No combining marks -> grapheme and codepoint modes agree.
        for text in ("报道称", "サイエンス", "알려진"):
            assert mixed_segment(text, grapheme=True) == mixed_segment(text, grapheme=False)

    def test_grapheme_keeps_latin_accents_on_their_word(self):
        # Decomposed e + U+0301 (combining acute) must stay attached to the
        # Latin word, not split it. Escapes pin the normalization form so the
        # test does not depend on the source file's encoding of accented chars.
        text = "cafe\u0301 \u6211"  # 'cafe' + combining acute, then Han '\u6211'
        assert mixed_segment(text, grapheme=True) == ["cafe\u0301", "\u6211"]


class TestEvaluateMer:
    """End-to-end MER scoring. Needs jiwer + the normalization stack."""

    def _evaluate_mer(self):
        import pytest

        pytest.importorskip("jiwer")
        pytest.importorskip("num2words")
        from nemo_skills.evaluation.evaluator.audio import evaluate_mer

        return evaluate_mer

    def test_perfect_match_is_zero(self):
        evaluate_mer = self._evaluate_mer()
        r = evaluate_mer("我 like 猫", "我 like 猫", normalization_mode="none")
        assert r["wer"] == 0.0
        assert r["wer_ref_words"] == 3  # 我 + like + 猫

    def test_one_english_word_substitution(self):
        evaluate_mer = self._evaluate_mer()
        # 3 mixed tokens, one English word wrong -> 1/3.
        r = evaluate_mer("我 like 猫", "我 hate 猫", normalization_mode="none")
        assert r["wer_ref_words"] == 3
        assert r["wer_substitutions"] == 1
        assert abs(r["wer"] - 1 / 3) < 1e-9

    def test_english_boundary_error_survives_mandarin_spacing_difference(self):
        evaluate_mer = self._evaluate_mer()
        r = evaluate_mer("今 天 use AI", "今天 useAI", normalization_mode="none", normalize_compound=False)
        assert r["wer_ref_words"] == 4  # 今 + 天 + use + AI
        assert r["wer"] == 0.5

    def test_empty_reference_is_dropped(self):
        evaluate_mer = self._evaluate_mer()
        r = evaluate_mer("", "anything", normalization_mode="none")
        assert r["wer"] is None


class TestCerStripWhitespace:
    """strip_whitespace makes CER count only non-space characters (paper-style)."""

    def _evaluate_cer(self):
        import pytest

        pytest.importorskip("jiwer")
        from nemo_skills.evaluation.evaluator.audio import evaluate_cer

        return evaluate_cer

    def test_spaces_not_counted_when_stripped(self):
        evaluate_cer = self._evaluate_cer()
        # ref/hyp differ only by a space -> 0 CER when whitespace is stripped.
        r = evaluate_cer("ab cd", "abcd", normalization_mode="none", strip_whitespace=True)
        assert r["cer"] == 0.0

    def test_spaces_counted_by_default(self):
        evaluate_cer = self._evaluate_cer()
        r = evaluate_cer("ab cd", "abcd", normalization_mode="none", strip_whitespace=False)
        assert r["cer"] > 0

    def test_legacy_strip_true_accepts_compound_normalization(self):
        evaluate_cer = self._evaluate_cer()
        r = evaluate_cer(
            "ab cd",
            "abcd",
            normalization_mode="none",
            normalize_compound=True,
            strip_whitespace=True,
        )
        assert r["cer"] == 0.0

    def test_all_mode_accepts_compound_normalization(self):
        evaluate_cer = self._evaluate_cer()
        r = evaluate_cer(
            "ab cd",
            "abcd",
            normalization_mode="none",
            normalize_compound=True,
            whitespace_mode="all",
        )
        assert r["cer"] == 0.0

    def test_han_adjacent_preserves_english_word_spaces(self):
        assert normalize_cer_whitespace("今 天 use AI", mode="han_adjacent") == "今天use AI"
        assert normalize_cer_whitespace("我喜欢 ice cream", mode="han_adjacent") == "我喜欢ice cream"
        assert normalize_cer_whitespace("今天 NVIDIA RTX 5090 很好", mode="han_adjacent") == "今天NVIDIA RTX 5090很好"

    def test_han_adjacent_covers_ideographic_zero_and_extensions(self):
        assert normalize_cer_whitespace("〇 〇", mode="han_adjacent") == "〇〇"
        assert normalize_cer_whitespace("𨋢 𠮶", mode="han_adjacent") == "𨋢𠮶"

    def test_han_adjacent_preserves_embedded_korean_word_space(self):
        assert normalize_cer_whitespace("我喜欢 한국 음식", mode="han_adjacent") == "我喜欢한국 음식"

    def test_han_adjacent_scores_english_boundary_error(self):
        evaluate_cer = self._evaluate_cer()
        r = evaluate_cer(
            "今 天 use AI",
            "今天 useAI",
            normalization_mode="none",
            whitespace_mode="han_adjacent",
        )
        assert r["text"] == "今天use AI"
        assert r["pred_text"] == "今天useAI"
        assert r["cer_deletions"] == 1
        assert r["cer_ref_words"] == 8
        assert r["cer"] == 0.125

    def test_han_adjacent_ignores_only_mandarin_formatting_spaces(self):
        evaluate_cer = self._evaluate_cer()
        r = evaluate_cer(
            "我 喜 欢 New York",
            "我喜欢New York",
            normalization_mode="none",
            whitespace_mode="han_adjacent",
        )
        assert r["cer"] == 0.0
        assert r["text"] == "我喜欢New York"

    def test_invalid_mode_fails_fast(self):
        with pytest.raises(ValueError, match="Invalid CER whitespace mode"):
            normalize_cer_whitespace("text", mode="invalid")

    def test_compound_normalization_conflicts_with_han_adjacent_whitespace(self):
        from nemo_skills.evaluation.evaluator.audio import evaluate_cer

        with pytest.raises(ValueError, match="normalize_compound cannot be combined"):
            evaluate_cer(
                "今 天 use AI",
                "今天 useAI",
                normalization_mode="none",
                normalize_compound=True,
                whitespace_mode="han_adjacent",
            )

    def test_invalid_mode_is_not_masked_by_legacy_argument(self):
        from nemo_skills.evaluation.evaluator.audio import evaluate_cer

        with pytest.raises(ValueError, match="Invalid CER whitespace mode"):
            evaluate_cer("text", "text", strip_whitespace=False, whitespace_mode="invalid")


class TestCerWhitespacePolicy:
    """CER whitespace behavior is explicit and scoped to audited datasets."""

    def test_policy_is_dataset_opt_in(self):
        assert resolve_cer_whitespace_mode() == "keep"

    def test_dataset_override_wins(self):
        assert resolve_cer_whitespace_mode(False) == "keep"
        assert resolve_cer_whitespace_mode(True) == "all"
        assert resolve_cer_whitespace_mode("han_adjacent") == "han_adjacent"

    def test_string_boolean_override_is_rejected(self):
        with pytest.raises(ValueError, match="Invalid CER whitespace override"):
            resolve_cer_whitespace_mode("false")

    def test_conflicting_override_keys_are_rejected(self):
        with pytest.raises(ValueError, match="Set only one"):
            resolve_sample_cer_whitespace_mode(
                {"src_lang": "cmn", "cer_whitespace_mode": "han_adjacent", "cer_strip_whitespace": False}
            )

    @pytest.mark.parametrize(
        ("extra_fields", "message"),
        [
            ({"cer_whitespace_mode": None}, "cer_whitespace_mode must be"),
            ({"cer_whitespace_mode": True}, "cer_whitespace_mode must be"),
            ({"cer_strip_whitespace": None}, "cer_strip_whitespace must be"),
            ({"cer_strip_whitespace": "false"}, "cer_strip_whitespace must be"),
        ],
    )
    def test_invalid_metadata_types_are_rejected(self, extra_fields, message):
        with pytest.raises(ValueError, match=message):
            resolve_sample_cer_whitespace_mode(extra_fields)

    def test_use_mer_retains_all_whitespace_fallback_for_existing_records(self):
        assert resolve_sample_cer_whitespace_mode({"use_mer": True}) == "all"

    def test_multilingual_mandarin_routes_both_cer_columns_through_policy(self, monkeypatch):
        from nemo_skills.evaluation.evaluator import audio

        modes = []

        def fake_evaluate_cer(reference, hypothesis, *, key_prefix="cer", whitespace_mode="keep", **kwargs):
            modes.append((key_prefix, whitespace_mode))
            return {key_prefix: 0.0, "is_correct": True, "text": reference, "pred_text": hypothesis}

        monkeypatch.setattr(audio, "evaluate_cer", fake_evaluate_cer)
        sample = {
            "task_type": "Multilingual-ASR",
            "expected_answer": "今 天 天 气 很 好",
            "generation": "今天天气很好",
            "extra_fields": {
                "src_lang": "cmn_hans_cn",
                "use_cer": True,
                "cer_whitespace_mode": "han_adjacent",
            },
        }
        audio.evaluate_sample(sample, audio.AudioEvaluatorConfig(apply_normalization=False))

        assert modes == [("wer", "han_adjacent"), ("cer", "han_adjacent")]

    def test_unconfigured_chinese_dataset_keeps_whitespace(self, monkeypatch):
        from nemo_skills.evaluation.evaluator import audio

        modes = []

        def fake_evaluate_cer(reference, hypothesis, *, key_prefix="cer", whitespace_mode="keep", **kwargs):
            modes.append((key_prefix, whitespace_mode))
            return {key_prefix: 0.0, "is_correct": True, "text": reference, "pred_text": hypothesis}

        monkeypatch.setattr(audio, "evaluate_cer", fake_evaluate_cer)
        sample = {
            "task_type": "Multilingual-ASR",
            "expected_answer": "今 天",
            "generation": "今天",
            "extra_fields": {"src_lang": "zh-CN", "use_cer": True},
        }
        audio.evaluate_sample(sample, audio.AudioEvaluatorConfig(apply_normalization=False))

        assert modes == [("wer", "keep"), ("cer", "keep")]

    def test_use_mer_preserves_headline_boundaries_and_honors_false_cer_override(self, monkeypatch):
        from nemo_skills.evaluation.evaluator import audio

        calls = []

        def fake_evaluate_mer(reference, hypothesis, *, normalize_compound=False, **kwargs):
            calls.append(("mer", normalize_compound))
            return {"wer": 0.5, "is_correct": False, "text": reference, "pred_text": hypothesis}

        def fake_evaluate_cer(reference, hypothesis, *, whitespace_mode="keep", **kwargs):
            calls.append(("cer", whitespace_mode))
            return {"cer": 0.0, "is_correct": True, "text": reference, "pred_text": hypothesis}

        monkeypatch.setattr(audio, "evaluate_mer", fake_evaluate_mer)
        monkeypatch.setattr(audio, "evaluate_cer", fake_evaluate_cer)
        sample = {
            "task_type": "Multilingual-ASR",
            "expected_answer": "今 天 use AI",
            "generation": "今天 useAI",
            "extra_fields": {
                "src_lang": "cmn_hans_cn",
                "use_cer": True,
                "use_mer": True,
                "cer_strip_whitespace": False,
            },
        }
        audio.evaluate_sample(sample, audio.AudioEvaluatorConfig(apply_normalization=False))

        assert calls == [("mer", False), ("cer", "keep")]

    @pytest.mark.parametrize(
        "whitespace_metadata",
        [{}, {"cer_whitespace_mode": "keep"}, {"cer_strip_whitespace": False}],
    )
    def test_keep_modes_preserve_compound_normalization(self, monkeypatch, whitespace_metadata):
        from nemo_skills.evaluation.evaluator import audio

        calls = []

        def fake_evaluate_cer(
            reference,
            hypothesis,
            *,
            key_prefix="cer",
            normalize_compound=False,
            whitespace_mode="keep",
            **kwargs,
        ):
            calls.append((key_prefix, normalize_compound, whitespace_mode))
            return {key_prefix: 0.0, "is_correct": True, "text": reference, "pred_text": hypothesis}

        monkeypatch.setattr(audio, "evaluate_cer", fake_evaluate_cer)
        sample = {
            "task_type": "Multilingual-ASR",
            "expected_answer": "私は ice cream が好き",
            "generation": "私は icecream が好き",
            "extra_fields": {
                "src_lang": "jpn_jp",
                "use_cer": True,
                **whitespace_metadata,
            },
        }
        audio.evaluate_sample(sample, audio.AudioEvaluatorConfig(apply_normalization=False))

        assert calls == [("wer", True, "keep"), ("cer", True, "keep")]

    def test_legacy_true_routes_both_cer_columns_to_all(self, monkeypatch):
        from nemo_skills.evaluation.evaluator import audio

        calls = []

        def fake_evaluate_cer(
            reference,
            hypothesis,
            *,
            key_prefix="cer",
            normalize_compound=False,
            whitespace_mode="keep",
            **kwargs,
        ):
            calls.append((key_prefix, normalize_compound, whitespace_mode))
            return {key_prefix: 0.0, "is_correct": True, "text": reference, "pred_text": hypothesis}

        monkeypatch.setattr(audio, "evaluate_cer", fake_evaluate_cer)
        sample = {
            "task_type": "Multilingual-ASR",
            "expected_answer": "今 天",
            "generation": "今天",
            "extra_fields": {"src_lang": "cmn_hans_cn", "use_cer": True, "cer_strip_whitespace": True},
        }
        audio.evaluate_sample(sample, audio.AudioEvaluatorConfig(apply_normalization=False))

        assert calls == [("wer", False, "all"), ("cer", False, "all")]

    def test_direct_cer_task_honors_whitespace_mode(self, monkeypatch):
        from nemo_skills.evaluation.evaluator import audio

        modes = []

        def fake_evaluate_cer(reference, hypothesis, *, whitespace_mode="keep", **kwargs):
            modes.append(whitespace_mode)
            return {"cer": 0.0, "is_correct": True, "text": reference, "pred_text": hypothesis}

        monkeypatch.setattr(audio, "evaluate_cer", fake_evaluate_cer)
        sample = {
            "task_type": "CER",
            "expected_answer": "今 天",
            "generation": "今天",
            "extra_fields": {"src_lang": "cmn_hans_cn", "cer_whitespace_mode": "han_adjacent"},
        }
        audio.evaluate_sample(sample, audio.AudioEvaluatorConfig(apply_normalization=False))

        assert modes == ["han_adjacent"]


class TestLowerNopunctNormalization:
    """Mark-preserving normalization (lowercase + unpunctuated) for paper-comparable
    CER. Dependency-free: this mode returns before any optional imports."""

    def _norm(self):
        from nemo_skills.evaluation.evaluator.audio import preprocess_asr_text

        return preprocess_asr_text

    def test_lowercases_and_strips_punctuation(self):
        assert self._norm()("Hello, World!", mode="lower_nopunct") == "hello world"

    def test_preserves_thai_combining_marks(self):
        # base ท + vowel ี + tone ่ all kept; trailing punctuation dropped.
        assert self._norm()("ที่!", mode="lower_nopunct") == "ที่"

    def test_preserves_myanmar_marks(self):
        assert self._norm()("ကျော่.", mode="lower_nopunct") == "ကျော่"

    def test_differs_from_no_tn_itn_which_strips_marks(self):
        # no_tn_itn drops the Thai marks (\w doesn't match them); lower_nopunct keeps them.
        norm = self._norm()
        assert norm("ที่", mode="lower_nopunct") != norm("ที่", mode="no_tn_itn")


class TestMultilingualPreserveMarks:
    """`preserve_marks` opt-in on multilingual normalization keeps abugida vowel/tone
    marks (so MER counts them) while retaining the rest of the multilingual pipeline.
    Dependency-free here: non-English lang + no digits avoids whisper/num2words imports."""

    def _norm(self):
        import pytest

        # multilingual mode imports whisper_normalizer unconditionally (for the
        # English-normalizer branch), so skip if it's unavailable.
        pytest.importorskip("whisper_normalizer")
        from nemo_skills.evaluation.evaluator.audio import preprocess_asr_text

        return preprocess_asr_text

    def test_preserve_marks_keeps_thai_vowel_and_tone(self):
        out = self._norm()("ที่!", mode="multilingual", lang="th", preserve_marks=True)
        assert out == "ที่"  # base + vowel + tone kept, punctuation dropped

    def test_default_multilingual_strips_marks(self):
        out = self._norm()("ที่", mode="multilingual", lang="th", preserve_marks=False)
        assert "ี" not in out and "่" not in out  # vowel ี and tone ่ removed
