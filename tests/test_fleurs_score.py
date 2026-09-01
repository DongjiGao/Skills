# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from nemo_skills.dataset.fleurs.audio_score import compute_score
from nemo_skills.evaluation.metrics.audio_metrics import AudioMetrics


def _prediction(
    subset: str,
    *,
    errors: int,
    ref_units: int,
    cer_errors: int,
    cer_ref_chars: int,
) -> dict:
    return {
        "generation": "answer",
        "is_correct": False,
        "subset_for_metrics": subset,
        "wer": errors / ref_units,
        "wer_errors": errors,
        "wer_ref_words": ref_units,
        "wer_substitutions": errors,
        "wer_insertions": 0,
        "wer_deletions": 0,
        "cer": cer_errors / cer_ref_chars,
        "cer_errors": cer_errors,
        "cer_ref_words": cer_ref_chars,
        "cer_substitutions": cer_errors,
        "cer_insertions": 0,
        "cer_deletions": 0,
    }


def test_audio_metrics_reports_corpus_error_rates_per_locale() -> None:
    metrics = AudioMetrics(max_k=1)
    metrics.update([_prediction("large", errors=10, ref_units=100, cer_errors=5, cer_ref_chars=100)])
    metrics.update([_prediction("small", errors=9, ref_units=10, cer_errors=4, cer_ref_chars=10)])

    output = metrics.get_metrics()["pass@1"]

    assert output["wer"] == 17.27
    assert output["wer::large"] == 10.0
    assert output["wer::small"] == 90.0
    assert output["wer::large_ref_words"] == 100
    assert output["wer::small_num_entries"] == 1
    assert output["cer::large"] == 5.0
    assert output["cer::small"] == 40.0
    assert output["cer::small_ref_chars"] == 10


def test_audio_metrics_reports_corpus_bleu_per_language_pair() -> None:
    metrics = AudioMetrics(max_k=1)
    for subset, reference, hypothesis in (
        ("en->de", "the cat is here", "the cat is here"),
        ("en->de", "the dog is here", "the dog is here"),
        ("fr->en", "a bird is here", "wrong words are here"),
    ):
        metrics.update(
            [
                {
                    "generation": hypothesis,
                    "is_correct": False,
                    "subset_for_metrics": subset,
                    "bleu": 0.0,
                    "text": reference,
                    "pred_text": hypothesis,
                    "bleu_tokenize": "13a",
                }
            ]
        )

    output = metrics.get_metrics()["pass@1"]

    assert output["bleu::en->de"] == 100.0
    assert output["bleu::en->de_num_entries"] == 2
    assert output["bleu::fr->en"] < 100.0
    assert output["bleu::fr->en_num_entries"] == 1


def test_fleurs_group_reports_unweighted_language_and_pair_macros() -> None:
    combined = {
        "fleurs.asr": {
            "pass@1": {
                "num_entries": 110,
                "avg_tokens": 10,
                "gen_seconds": 5,
                "success_rate": 50.0,
                "no_answer": 0.0,
                "wer": 17.27,
                "wer_macro": 50.0,
                "cer": 8.18,
                "wer::large": 10.0,
                "wer::small": 90.0,
                "cer::large": 5.0,
                "cer::small": 40.0,
            }
        },
        "fleurs.st": {
            "pass@1": {
                "num_entries": 30,
                "avg_tokens": 12,
                "gen_seconds": 7,
                "success_rate": 25.0,
                "no_answer": 0.0,
                "bleu": 30.0,
                "bleu::en_us->de_de": 50.0,
                "bleu::de_de->en_us": 10.0,
                "bleu::en_us->fr_fr": 20.0,
            }
        },
    }

    output = compute_score(combined)["pass@1"]

    # Existing row-weighted fields are retained and explicitly differ from
    # the new equal-language/equal-pair comparable aggregates.
    assert output["wer"] == 17.27
    assert output["bleu"] == 30.0
    assert output["cer"] == 8.18
    assert output["wer_language_macro"] == 50.0
    assert output["wer_language_count"] == 2
    assert output["cer_language_macro"] == 22.5
    assert output["cer_language_count"] == 2
    assert output["bleu_pair_macro"] == 26.67
    assert output["bleu_pair_count"] == 3
    assert output["bleu_en_to_x_macro"] == 35.0
    assert output["bleu_en_to_x_count"] == 2
    assert output["bleu_x_to_en_macro"] == 10.0
    assert output["bleu_x_to_en_count"] == 1
    assert output["wer::small"] == 90.0
    assert output["cer::large"] == 5.0
    assert output["bleu::en_us->de_de"] == 50.0


def test_group_does_not_invent_macros_without_subset_metrics() -> None:
    output = compute_score(
        {
            "covost2.asr": {"pass@1": {"num_entries": 2, "wer": 7.0}},
            "covost2.st": {"pass@1": {"num_entries": 2, "bleu": 18.0}},
        }
    )["pass@1"]

    assert output["wer"] == 7.0
    assert output["bleu"] == 18.0
    assert "wer_language_macro" not in output
    assert "cer_language_macro" not in output
    assert "bleu_pair_macro" not in output


def test_zero_scores_are_valid_macros() -> None:
    output = compute_score(
        {
            "fleurs.asr": {
                "pass@1": {
                    "num_entries": 1,
                    "wer::en_us": 0.0,
                    "cer::en_us": 0.0,
                }
            },
            "fleurs.st": {
                "pass@1": {
                    "num_entries": 1,
                    "bleu::en_us->de_de": 0.0,
                }
            },
        }
    )["pass@1"]

    assert output["wer_language_macro"] == 0.0
    assert output["cer_language_macro"] == 0.0
    assert output["bleu_pair_macro"] == 0.0
    assert output["bleu_en_to_x_macro"] == 0.0
