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

import pytest

pytest.importorskip("kaldialign")

from nemo_skills.evaluation.evaluator.audio import evaluate_msasr, normalize_cpwer_stream, resolve_speaker_tag_style
from nemo_skills.evaluation.metrics.audio_metrics import AudioMetrics
from nemo_skills.evaluation.metrics.chime8_normalizer import normalize as normalize_chime8
from nemo_skills.evaluation.metrics.cpwer_utils import (
    calculate_session_cpWER,
    calculate_session_cpWER_bruteforce,
    session_cpwer_counts,
)
from nemo_skills.evaluation.metrics.librispeechmix_sot_utils import parse_sot_speaker_streams


PLAN_EXAMPLE_REF = (
    "<spk:0> so how far along are you <spk:1> i have no idea i'll have to try to figure it out "
    "<spk:0> have you seen a doctor yet <spk:1> no <spk:0> alison i had the best birth <spk:1> aw i'm so glad"
)
PLAN_EXAMPLE_HYP = PLAN_EXAMPLE_REF


def test_resolve_speaker_tag_style_defaults_to_spk():
    assert resolve_speaker_tag_style("<spk:*>") == "spk"
    assert resolve_speaker_tag_style("spk") == "spk"


def test_parse_sot_speaker_streams_default_spk_tags():
    streams = parse_sot_speaker_streams(
        "<spk:0> hello there <spk:1> good morning",
        mode="strict",
        strict_tag_style="spk",
    )
    assert streams["s0"] == "hello there"
    assert streams["s1"] == "good morning"


def test_chime8_normalizer_is_idempotent():
    text = "Hello, Dr. Smith said 10 dollars."
    once = normalize_chime8(text)
    twice = normalize_chime8(once)
    assert once == twice
    assert once == normalize_cpwer_stream(text, mode="chime8")


def test_session_cpwer_counts_permutation_invariant():
    ref = ["hello world", "good morning"]
    hyp = ["good morning", "hello world"]
    counts = session_cpwer_counts(ref, hyp)
    assert counts["errors"] == 0
    assert counts["cpwer"] == 0.0
    assert counts["ref_words"] == 4


def test_nemo_cpwer_matches_bruteforce():
    ref = ["hi how are you well that's nice", "i'm good yeah how is your sister"]
    hyp = ["hey how are you we that's nice", "i'm good yes hi is your sister"]
    hungarian = calculate_session_cpWER(hyp, ref)[0]
    bruteforce = calculate_session_cpWER_bruteforce(hyp, ref)[0]
    assert hungarian == pytest.approx(bruteforce)


def test_evaluate_msasr_perfect_match():
    result = evaluate_msasr(PLAN_EXAMPLE_REF, PLAN_EXAMPLE_HYP)
    assert result["cpwer"] == 0.0
    assert result["cpwer_errors"] == 0
    assert result["is_correct"] is True
    assert "so how far along are you" in result["text"]


def test_evaluate_msasr_speaker_permutation():
    hyp = (
        "<spk:1> i have no idea i'll have to try to figure it out no aw i'm so glad "
        "<spk:0> so how far along are you have you seen a doctor yet alison i had the best birth"
    )
    result = evaluate_msasr(PLAN_EXAMPLE_REF, hyp)
    assert result["cpwer"] == 0.0
    assert result["cpwer_errors"] == 0


def test_audio_metrics_accumulates_cpwer():
    metrics = AudioMetrics()
    sample = {
        "cpwer": 0.1,
        "cpwer_errors": 1,
        "cpwer_substitutions": 1,
        "cpwer_insertions": 0,
        "cpwer_deletions": 0,
        "cpwer_ref_words": 10,
        "is_correct": False,
    }
    metrics.update([sample])
    metrics.update(
        [
            {
                **sample,
                "cpwer": 0.2,
                "cpwer_errors": 2,
                "cpwer_substitutions": 0,
                "cpwer_insertions": 2,
            }
        ]
    )
    output = metrics.get_metrics()["pass@1"]
    assert output["cpwer"] == 15.0
    assert output["cpwer_macro"] == 15.0
    assert output["cpwer_errors"] == 3
    assert output["cpwer_ref_words"] == 20


@pytest.mark.parametrize(
    "reference,hypothesis",
    [
        (
            "<spk:0> hello world <spk:1> good morning",
            "<spk:0> hello there <spk:1> good morning",
        ),
        (
            "<spk:0> ten seconds <spk:1> twenty minutes",
            "<spk:0> 10 seconds <spk:1> 20 minutes",
        ),
    ],
)
def test_cpwer_parity_with_meeteval_chime8(reference, hypothesis):
    meeteval = pytest.importorskip("meeteval")
    pytest.importorskip("chime_utils")

    ref_streams = parse_sot_speaker_streams(reference, mode="strict", strict_tag_style="spk")
    hyp_streams = parse_sot_speaker_streams(hypothesis, mode="canonical", strict_tag_style="spk")
    normalized_ref = {speaker: normalize_chime8(text) for speaker, text in ref_streams.items()}
    normalized_hyp = {speaker: normalize_chime8(text) for speaker, text in hyp_streams.items()}
    skills_counts = session_cpwer_counts(normalized_ref, normalized_hyp)

    ref_seglst = meeteval.io.SegLST([
        meeteval.io.SegLSTSegment(
            session_id="session",
            speaker=speaker,
            words=text,
            start_time=float(idx),
            end_time=float(idx + 1),
        )
        for idx, (speaker, text) in enumerate(normalized_ref.items())
    ])
    hyp_seglst = meeteval.io.SegLST([
        meeteval.io.SegLSTSegment(
            session_id="session",
            speaker=speaker,
            words=text,
            start_time=float(idx),
            end_time=float(idx + 1),
        )
        for idx, (speaker, text) in enumerate(normalized_hyp.items())
    ])

    from chime_utils.text_norm import get_txt_norm

    normalizer = get_txt_norm("chime8")
    meeteval_error_rate = meeteval.wer.cp_word_error_rate(
        ref_seglst,
        hyp_seglst,
        normalizer=normalizer,
    )
    assert skills_counts["cpwer"] == pytest.approx(float(meeteval_error_rate), rel=1e-6, abs=1e-6)
