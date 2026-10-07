# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib
import json
import wave
from pathlib import Path

import pytest

from nemo_skills.dataset.utils import get_dataset_module

prepare = importlib.import_module("nemo_skills.dataset.ctc-timestamp-eval.prepare")
scorer = importlib.import_module("nemo_skills.dataset.ctc-timestamp-eval.scorer")


@pytest.mark.parametrize(
    ("alias", "count"),
    (
        ("ch109", 10),
        ("ami-ihm-test", 4),
        ("lsm", 393),
        ("d12a-ar", 409),
        ("d12a-de", 520),
        ("d12a-en", 280),
        ("d12a-es", 273),
        ("d12a-fr", 248),
        ("d12a-hi", 676),
        ("d12a-it", 67),
        ("d12a-ja", 1225),
        ("d12a-ko", 416),
        ("d12a-pt", 948),
        ("d12a-ru", 361),
        ("d12a-th", 738),
        ("d12a-tr", 842),
        ("d12a-vi", 358),
        ("d12a-zh", 590),
    ),
)
def test_aliases_resolve(alias, count):
    module, _ = get_dataset_module(f"ctc-timestamp-eval.{alias}")
    assert module.NUM_SAMPLES == count
    assert module.EVAL_SPLIT == "test"
    assert "CTCTimestampMetrics" in module.METRICS_TYPE


def test_lsm_selection_is_balanced_and_deterministic():
    selected = json.loads((Path(prepare.__file__).parent / "selected_files.json").read_text())["lsm"]
    indices = prepare.lsm_selected_indices(selected)
    assert selected["expected_files"] == 393
    assert selected["source_commit"] == "5143690d1a6ebcd37b7f4d1b0f9fa8c83944aba5"
    assert {mix: len(rows) for mix, rows in indices.items()} == {
        "lsm-test-clean-1mix": 131,
        "lsm-test-clean-2mix": 131,
        "lsm-test-clean-3mix": 131,
    }
    assert indices["lsm-test-clean-1mix"][:5] == [33, 54, 56, 67, 86]
    assert indices["lsm-test-clean-2mix"][:5] == [87, 125, 147, 156, 158]
    assert indices["lsm-test-clean-3mix"][:5] == [22, 23, 25, 81, 91]


def test_reconcile_textgrid_words_uses_exact_lsm_transcript_tokens():
    aligned = [
        {"word": "to-morrow", "start": 0.0, "end": 0.6, "speaker": "speaker0"},
        {"word": "zoof", "start": 0.7, "end": 0.9, "speaker": "speaker0"},
        {"word": "s", "start": 0.9, "end": 1.0, "speaker": "speaker0"},
    ]
    reconciled = prepare.reconcile_textgrid_words(aligned, "to morrow zoof's")
    assert [item["word"] for item in reconciled] == ["to", "morrow", "zoof's"]
    assert reconciled[0]["start"] == 0.0
    assert reconciled[1]["end"] == 0.6
    assert reconciled[2]["start"] == 0.7
    assert reconciled[2]["end"] == 1.0


def test_parse_textgrid_words(tmp_path):
    path = tmp_path / "sample.TextGrid"
    path.write_text(
        """File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0
xmax = 1.25
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        intervals [1]:
            xmin = 0.0
            xmax = 0.2
            text = "<eps>"
        intervals [2]:
            xmin = 0.2
            xmax = 0.6
            text = "Hello"
    item [2]:
        class = "IntervalTier"
        name = "phones"
""",
        encoding="utf-8",
    )
    words, duration = prepare.parse_textgrid_words(path)
    assert duration == 1.25
    assert words == [{"word": "Hello", "start": 0.2, "end": 0.6, "speaker": "speaker0"}]


def test_merge_channel_ctms_and_shift_crop(tmp_path):
    c1 = tmp_path / "c1.ctm"
    c2 = tmp_path / "c2.ctm"
    c1.write_text("rec 1 1.10 0.20 hello 1 lex original_a\n", encoding="utf-8")
    c2.write_text("rec 1 1.20 0.20 yes 1 lex original_b\n", encoding="utf-8")
    words = prepare.crop_and_merge_ctms([c1, c2], offset=1.0, duration=1.0)
    assert words == [
        {"word": "hello", "start": 0.1, "end": 0.3, "speaker": "speaker0"},
        {"word": "yes", "start": 0.2, "end": 0.4, "speaker": "speaker1"},
    ]


def test_d12_word_ends_use_next_start_and_segment_end():
    row = {
        "offset": 3.0,
        "duration": 1.5,
        "alignment": {"word": [["one", 3.1, 0.01, None], ["two", 3.7, 99.0, None]]},
    }
    assert prepare.d12_reference_words(row) == [
        {"word": "one", "start": 0.1, "end": 0.7, "speaker": "speaker0"},
        {"word": "two", "start": 0.7, "end": 1.5, "speaker": "speaker0"},
    ]


def test_d12_event_only_alignment_is_not_scoreable():
    row = {
        "offset": 10.0,
        "duration": 0.08,
        "alignment": {"word": [["[Timestamp alignment GT labels]", 10.0, 0.08, None]]},
    }
    assert prepare.d12_reference_words(row) == []


def test_materialize_d12_language_subset(tmp_path):
    label_root = tmp_path / "labels"
    (label_root / "ar").mkdir(parents=True)
    (label_root / "ar" / "d12a_ar_manifest.json").write_text(
        json.dumps({"audio_filepath": "/source/session-a.opus"}) + "\n", encoding="utf-8"
    )
    aligned = tmp_path / "d12a_test.json"
    aligned.write_text(
        json.dumps(
            {
                "audio_filepath": "/source/session-a/utterance_postprocessed.opus",
                "offset": 2.0,
                "duration": 1.0,
                "alignment": {"word": [["hello", 2.1, None], ["world", 2.6, None]]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    segment_root = tmp_path / "segments"
    segment_root.mkdir()
    with wave.open(str(segment_root / "test.d12a-00000-utterance_postprocessed.wav"), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\0\0" * 16_000)

    data_dir = tmp_path / "data"
    counts = prepare._materialize_d12a(
        languages=["ar"],
        aligned_manifest=aligned,
        label_root=label_root,
        segment_audio_root=segment_root,
        selected={"languages": {"ar": {"expected_files": 1, "sessions": ["session-a"]}}},
        data_dir=data_dir,
        audio_prefix=Path("/data/ctc-timestamp-eval/audio"),
        gt_prefix=Path("/data/ctc-timestamp-eval/ground_truth"),
    )
    assert counts == {"d12a-ar": 1}
    record = json.loads((data_dir / "d12a-ar" / "test.jsonl").read_text())
    assert record["source_lang"] == record["target_lang"] == "ar"
    assert record["reference_words"] == [
        {"word": "hello", "start": 0.1, "end": 0.6, "speaker": "speaker0"},
        {"word": "world", "start": 0.6, "end": 1.0, "speaker": "speaker0"},
    ]
    assert (data_dir / "audio" / "d12a-ar" / "d12a-00000-utterance_postprocessed.wav").is_file()
    assert (data_dir / "ground_truth" / "d12a-ar" / "d12a-00000-utterance_postprocessed.ctm").is_file()


def test_speaker_permutation_and_timestamp_metrics():
    row = {
        "subset_for_metrics": "ch109",
        "reference_words": [
            {"word": "hello", "start": 0.0, "end": 0.4, "speaker": "speaker0"},
            {"word": "yes", "start": 0.1, "end": 0.3, "speaker": "speaker1"},
            {"word": "world", "start": 0.5, "end": 0.9, "speaker": "speaker0"},
            {"word": "no", "start": 0.6, "end": 0.8, "speaker": "speaker1"},
        ],
        "timestamps": {
            "speaker_word_timestamps": {
                "arbitrary_a": [
                    {"word": "yes", "start": 0.2, "end": 0.4},
                    {"word": "no", "start": 0.7, "end": 0.9},
                ],
                "arbitrary_b": [
                    {"word": "Hello,", "start": 0.1, "end": 0.5},
                    {"word": "world", "start": 0.6, "end": 1.0},
                ],
            }
        },
    }
    result = scorer.evaluate_ctc_timestamp_sample(row, tolerance_seconds=0.1)
    assert result["wer_errors"] == 0
    assert result["timestamp_true_positives"] == 8
    assert result["timestamped_word_matches"] == 4
    assert result["start_error_seconds_sum"] == pytest.approx(0.4)
    assert result["end_error_seconds_sum"] == pytest.approx(0.4)


def test_timestamp_f1_uses_single_configurable_tolerance():
    reference_words = [
        {"word": f"word{index}", "start": float(index), "end": float(index) + 0.5, "speaker": "speaker0"}
        for index in range(4)
    ]
    errors = (0.05, 0.15, 0.25, 0.35)
    prediction_words = [
        {"word": item["word"], "start": item["start"] + error, "end": item["end"] + error}
        for item, error in zip(reference_words, errors)
    ]
    result = scorer.evaluate_ctc_timestamp_sample(
        {
            "subset_for_metrics": "test",
            "reference_words": reference_words,
            "timestamps": {"speaker_word_timestamps": {"speaker0": prediction_words}},
        }
    )
    assert scorer.DEFAULT_TOLERANCE_SECONDS == 0.2
    assert result["timestamp_tolerance_seconds"] == 0.2
    assert result["timestamp_true_positives"] == 4

    metrics = scorer.CTCTimestampMetrics(max_k=1)
    metrics.update([result])
    values = metrics.get_metrics()["pass@1"]
    assert values["timestamp_f1"] == 50.0
    assert values["timestamp_reference_events"] == 8
    assert values["timestamp_hypothesis_events"] == 8
    assert values["timestamp_tolerance_ms"] == 200.0
    assert not any(key.startswith("timestamp_f1_") for key in values)

    custom_result = scorer.evaluate_ctc_timestamp_sample(
        {
            "subset_for_metrics": "test",
            "reference_words": reference_words,
            "timestamps": {"speaker_word_timestamps": {"speaker0": prediction_words}},
        },
        tolerance_seconds=0.3,
    )
    assert custom_result["timestamp_true_positives"] == 6


def test_evaluator_built_from_the_config_dict_skills_passes_scores_rows(tmp_path):
    from nemo_skills.evaluation.evaluator import get_evaluator_class

    common = importlib.import_module("nemo_skills.dataset.ctc-timestamp-eval._common")
    eval_type = common.EVAL_ARGS.removeprefix("++eval_type=")
    row = {
        "subset_for_metrics": "test",
        "reference_words": [{"word": "hello", "start": 0.0, "end": 0.5, "speaker": "speaker0"}],
        "timestamps": {"speaker_word_timestamps": {"speaker0": [{"word": "hello", "start": 0.25, "end": 0.75}]}},
    }

    # Generation scores each row as it finishes; both boundaries are 0.25 s off.
    inline = get_evaluator_class(eval_type, {"tolerance_seconds": 0.3})
    assert asyncio.run(inline.eval_single(row))["timestamp_true_positives"] == 2

    # Evaluation after generation rescores the output file in place, at the default tolerance.
    output_file = tmp_path / "output.jsonl"
    output_file.write_text(json.dumps(row) + "\n")
    asyncio.run(get_evaluator_class(eval_type, {"input_file": str(output_file)}).eval_full())
    scored = json.loads(output_file.read_text())
    assert scored["timestamp_tolerance_seconds"] == scorer.DEFAULT_TOLERANCE_SECONDS
    assert scored["timestamp_true_positives"] == 0


def test_timestamp_f1_treats_start_and_end_as_independent_events():
    result = scorer.evaluate_ctc_timestamp_sample(
        {
            "subset_for_metrics": "test",
            "reference_words": [
                {"word": "hello", "start": 0.0, "end": 0.5, "speaker": "speaker0"},
                {"word": "world", "start": 1.0, "end": 1.5, "speaker": "speaker0"},
            ],
            "word_timestamps": [
                {"word": "hello", "start": 0.1, "end": 0.8, "speaker": "speaker0"},
                {"word": "world", "start": 1.3, "end": 1.6, "speaker": "speaker0"},
            ],
        },
        tolerance_seconds=0.2,
    )

    assert result["timestamped_word_matches"] == 2
    assert result["timestamp_true_positives"] == 2

    metrics = scorer.CTCTimestampMetrics(max_k=1)
    metrics.update([result])
    values = metrics.get_metrics()["pass@1"]
    assert values["timestamp_precision"] == 50.0
    assert values["timestamp_recall"] == 50.0
    assert values["timestamp_f1"] == 50.0


def test_scorer_rejects_missing_timestamps():
    with pytest.raises(ValueError, match="no word timestamps"):
        scorer.evaluate_ctc_timestamp_sample(
            {
                "reference_words": [{"word": "hello", "start": 0.0, "end": 0.4, "speaker": "speaker0"}],
                "generation": "hello",
            }
        )


def test_plugin_results_without_word_timestamps_are_counted_as_unavailable():
    reference_words = [{"word": "hello", "start": 0.0, "end": 0.4, "speaker": "speaker0"}]
    served = {
        "words": [{"word": "hello", "start": 0.0, "end": 0.4, "speaker": "0"}],
        "diarization": [{"speaker": 0, "start": 0.0, "end": 0.4}],
        "speaker_tag_to_diarization_speaker": {"0": 0},
        "error": None,
    }
    unavailable = {**served, "words": [], "speaker_tag_to_diarization_speaker": {}, "error": "capture_unavailable"}
    results = [
        scorer.evaluate_ctc_timestamp_sample(
            {"subset_for_metrics": "lsm", "reference_words": reference_words, "ctc_timestamps": payload}
        )
        for payload in (served, unavailable)
    ]

    assert [result["timestamp_error"] for result in results] == [None, "capture_unavailable"]
    # A result without words still counts every reference word as missed.
    assert [result["wer_errors"] for result in results] == [0, 1]
    metrics = scorer.CTCTimestampMetrics(max_k=1)
    for result in results:
        metrics.update([result])
    values = metrics.get_metrics()["pass@1"]
    assert values["timestamp_unavailable_entries"] == 1
    assert values["wer"] == 50.0


def test_timestamp_normalization_preserves_combining_mark_tokens():
    assert scorer._normalized_word("े") == "े"
