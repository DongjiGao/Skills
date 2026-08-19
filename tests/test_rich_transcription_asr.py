# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import importlib
import json
import struct
import wave
from pathlib import Path

import pytest

from nemo_skills.dataset.utils import get_dataset_module
from nemo_skills.evaluation.evaluator import get_evaluator_class
from nemo_skills.evaluation.metrics.map_metrics import get_metrics

metrics_module = importlib.import_module(
    "nemo_skills.dataset.rich_transcription_asr.rich_transcription_asr_metrics"
)
prepare = importlib.import_module("nemo_skills.dataset.rich_transcription_asr.prepare")
eval_module = importlib.import_module(
    "nemo_skills.dataset.rich_transcription_asr.rich_transcription_asr_eval"
)
score_module = importlib.import_module(
    "nemo_skills.dataset.rich_transcription_asr.rich_transcription_asr_score"
)
nemo_asr_backend = importlib.import_module(
    "recipes.multimodal.server.backends.nemo_asr_backend"
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_wav(path: Path, seconds: float, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fout:
        fout.setnchannels(1)
        fout.setsampwidth(2)
        fout.setframerate(sample_rate)
        samples = [
            int(1000 * ((index % 20) - 10))
            for index in range(int(seconds * sample_rate))
        ]
        fout.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


def test_group_and_subbenchmarks_are_registered():
    val_group, _ = get_dataset_module("rich_transcription_asr.val")
    test_group, _ = get_dataset_module("rich_transcription_asr.test")
    d8_test_group, _ = get_dataset_module("rich_transcription_asr.d8_test")
    d12_test_group, _ = get_dataset_module("rich_transcription_asr.d12_test")
    assert set(val_group.BENCHMARKS) == {
        "rich_transcription_asr.val.d12a",
        "rich_transcription_asr.val.d8",
    }
    assert set(test_group.BENCHMARKS) == {
        "rich_transcription_asr.test.d12a",
        "rich_transcription_asr.test.d8",
    }
    assert d8_test_group.IS_BENCHMARK_GROUP
    assert set(d8_test_group.BENCHMARKS) == {
        "rich_transcription_asr.test.d8",
    }
    assert d8_test_group.SCORE_MODULE.endswith(".rich_transcription_asr_score")
    assert d12_test_group.IS_BENCHMARK_GROUP
    assert set(d12_test_group.BENCHMARKS) == {
        "rich_transcription_asr.test.d12a",
    }
    assert d12_test_group.SCORE_MODULE.endswith(".rich_transcription_asr_score")

    subbenchmark, _ = get_dataset_module("rich_transcription_asr.val.d8")
    assert subbenchmark.EVAL_SPLIT == "validation"
    assert subbenchmark.METRICS_TYPE.endswith("::RichTranscriptionASRMetrics")
    assert "::RichTranscriptionASREvaluator" in subbenchmark.EVAL_ARGS
    assert "++eval_config.evaluation_mode=event" in subbenchmark.EVAL_ARGS
    assert "++enable_audio=true" in subbenchmark.GENERATION_ARGS
    assert "++enable_audio_chunking=false" in subbenchmark.GENERATION_ARGS
    eval_type = subbenchmark.EVAL_ARGS.split()[0].removeprefix("++eval_type=")
    evaluator = get_evaluator_class(eval_type, {})
    assert type(evaluator).__name__ == "RichTranscriptionASREvaluator"
    assert evaluator.eval_config.evaluation_mode == "event"
    assert (
        type(get_metrics(subbenchmark.METRICS_TYPE)).__name__
        == "RichTranscriptionASRMetrics"
    )


def test_four_evaluation_modes_and_event_default():
    reference = "wrong words <ev:breath> <sty:stt:positive_affect> yes <sty:end:positive_affect>"
    hypothesis = "different words <ev:laughter> <sty:stt:positive_affect> yes <sty:end:positive_affect>"

    default = eval_module.evaluate_rich_transcription_asr_text(reference, hypothesis)
    event = eval_module.evaluate_rich_transcription_asr_text(reference, hypothesis, "1")
    style = eval_module.evaluate_rich_transcription_asr_text(reference, hypothesis, "2")
    event_style = eval_module.evaluate_rich_transcription_asr_text(
        reference, hypothesis, "3"
    )
    all_tokens = eval_module.evaluate_rich_transcription_asr_text(
        reference, hypothesis, "4"
    )

    assert default == event
    assert event["evaluation_mode"] == "event"
    assert event["wer"] == 1.0
    assert event["wer_ref_words"] == 1
    assert style["evaluation_mode"] == "style"
    assert style["wer"] == 0.0
    assert style["wer_ref_words"] == 2
    assert event_style["evaluation_mode"] == "event_style"
    assert event_style["wer"] == pytest.approx(1 / 3)
    assert all_tokens["evaluation_mode"] == "all"
    assert all_tokens["wer"] > 0


def test_nemo_asr_backend_accepts_explicit_sdpa_fallback():
    config = nemo_asr_backend.NeMoASRConfig.from_dict(
        {"model_path": "model.nemo", "attention_backend": "sdpa"}
    )
    assert config.attention_backend == "sdpa"

    with pytest.raises(ValueError, match="Unsupported NeMo ASR attention_backend"):
        nemo_asr_backend._patch_transformer_encoder_attention("unknown")


def test_special_token_extraction_canonicalizes_spacing_and_excludes_flat():
    text = (
        "< ev : breath > words <sty:stt:positive_affect> more "
        "<sty:end:positive_affect> <sty:stt:flat>"
    )
    assert metrics_module.extract_tokens(text, "all_special") == [
        "<ev:breath>",
        "<sty:stt:positive_affect>",
        "<sty:end:positive_affect>",
    ]


def test_special_alignment_matches_reference_tie_break_and_joint_order():
    event = metrics_module.score_event_tokens(
        ["<ev:breath> <ev:laughter>", "<ev:breath>"],
        ["<ev:breath> <ev:crying>", "<ev:breath> <ev:laughter>"],
        token_type="event",
    )
    assert event.wer == pytest.approx(2 / 3)
    assert (event.substitutions, event.insertions, event.deletions) == (1, 1, 0)

    references = ["<ev:breath> <sty:stt:positive_affect>"]
    hypotheses = ["<sty:stt:positive_affect> <ev:breath>"]
    assert (
        metrics_module.score_event_tokens(
            references, hypotheses, token_type="event"
        ).wer
        == 0.0
    )
    assert (
        metrics_module.score_event_tokens(
            references, hypotheses, token_type="style"
        ).wer
        == 0.0
    )
    assert (
        metrics_module.score_event_tokens(
            references, hypotheses, token_type="all_special"
        ).wer
        == 1.0
    )


def test_prepare_manifest_preserves_exact_rt_fields_and_paths(tmp_path):
    source_audio = tmp_path / "source" / "sample.opus"
    manifest_root = tmp_path / "manifests"
    _write_jsonl(
        manifest_root / "d8_test_noflat.json",
        [
            {
                "audio_filepath": str(source_audio),
                "duration": 12.5,
                "offset": 0,
                "scenario": "A directed performance.",
                "text": "hello <ev:breath> <sty:stt:positive_affect> yes <sty:end:positive_affect>",
            }
        ],
    )

    count = prepare.prepare_manifest_set(
        "test.d8",
        manifest_root=manifest_root,
        output_dir=tmp_path / "prepared",
        data_dir=tmp_path / "data" / "rich_transcription_asr",
        audio_prefix=Path("/data/rich_transcription_asr/audio"),
        no_audio=True,
        overwrite=False,
        max_samples=None,
    )

    assert count == 1
    record = json.loads((tmp_path / "prepared/test/d8/test.jsonl").read_text().strip())
    assert record["source_audio_filepath"] == str(source_audio)
    assert record["offset"] == 0.0
    assert record["duration"] == 12.5
    assert record["expected_answer"] == record["text"]
    assert record["subset_for_metrics"] == "test.d8"
    assert record["task_type"] == "Sound-ASR"
    assert record["extra_fields"]["scenario"] == "A directed performance."
    assert record["messages"][0]["audio"]["path"] == record["audio_filepath"]
    assert record["audio_filepath"].startswith(
        "/data/rich_transcription_asr/audio/test/d8/"
    )


def test_materialize_audio_segment_honors_offset_and_duration(tmp_path):
    source_audio = tmp_path / "source.wav"
    output_audio = tmp_path / "segments" / "slice.wav"
    _write_wav(source_audio, seconds=2.0)

    duration = prepare._materialize_audio_segment(
        source_audio,
        output_audio,
        offset=0.5,
        duration=0.75,
    )

    assert duration == pytest.approx(0.75)
    with wave.open(str(output_audio), "rb") as fin:
        assert fin.getnframes() == 12000
        assert fin.getframerate() == 16000


def test_materialize_audio_segment_pads_ten_millisecond_clip(tmp_path):
    source_audio = tmp_path / "source.wav"
    output_audio = tmp_path / "segments" / "slice.wav"
    _write_wav(source_audio, seconds=0.01)

    duration = prepare._materialize_audio_segment(
        source_audio,
        output_audio,
        offset=0.0,
        duration=0.01,
    )

    assert duration == pytest.approx(0.1)
    with wave.open(str(output_audio), "rb") as fin:
        assert fin.getnframes() == 1600
        assert fin.getframerate() == 16000


def test_materialize_audio_segment_rejects_manifest_overrun(tmp_path):
    source_audio = tmp_path / "source.wav"
    _write_wav(source_audio, seconds=1.0)

    with pytest.raises(
        prepare.InvalidAudioSegmentError, match="extends beyond source audio"
    ):
        prepare._materialize_audio_segment(
            source_audio,
            tmp_path / "segments" / "slice.wav",
            offset=0.75,
            duration=0.5,
        )


def test_metrics_default_to_event_only_corpus_wer():
    calculator = metrics_module.RichTranscriptionASRMetrics(max_k=1)
    calculator.update(
        [
            {
                "expected_answer": "<ev:breath> hello <sty:stt:positive_affect> yes <sty:end:positive_affect>",
                "generation": "<ev:breath> hello <sty:stt:positive_affect> yes <sty:end:positive_affect>",
                "category": "nonjudge",
            }
        ]
    )
    calculator.update(
        [
            {
                "expected_answer": "<ev:laughter> goodbye",
                "generation": "<ev:sneeze> goodbye",
                "category": "nonjudge",
            }
        ]
    )

    result = calculator.get_metrics()["pass@1"]
    assert result["evaluation_mode"] == "event"
    assert result["wer"] == 50.0
    assert result["wer_errors"] == 1
    assert result["ref_words"] == 2


def test_group_score_micro_averages_d12a_and_d8():
    combined = {
        "rich_transcription_asr.test.d12a": {
            "pass@1": {
                "evaluation_mode": "event",
                "num_entries": 2,
                "wer_errors": 2,
                "substitutions": 2,
                "ref_words": 10,
            }
        },
        "rich_transcription_asr.test.d8": {
            "pass@1": {
                "evaluation_mode": "event",
                "num_entries": 3,
                "wer_errors": 3,
                "deletions": 3,
                "ref_words": 10,
            }
        },
        "rich_transcription_asr.test.d8-test.d8": {
            "pass@1": {
                "evaluation_mode": "event",
                "num_entries": 3,
                "wer_errors": 3,
                "deletions": 3,
                "ref_words": 10,
            }
        },
    }

    result = score_module.compute_score(combined)["pass@1"]
    assert result["num_entries"] == 5
    assert result["evaluation_mode"] == "event"
    assert result["wer"] == 25.0


def test_token_stats_tsv_matches_reference_columns(tmp_path):
    results = {
        scope: metrics_module.score_event_tokens(
            ["<ev:breath> <sty:stt:positive_affect>"],
            ["<ev:laughter> <sty:stt:positive_affect>"],
            token_type=scope,
        )
        for scope in metrics_module.SPECIAL_SCOPES
    }
    output = tmp_path / "stats.tsv"
    metrics_module.write_token_stats_tsv(results, output)

    assert output.read_text().splitlines()[0].split("\t") == [
        "metric_scope",
        "token",
        "ref_tokens",
        "hyp_tokens",
        "correct",
        "deletions",
        "insertions",
        "substitutions_as_ref",
        "substitutions_as_hyp",
        "recall_percent",
        "precision_percent",
        "f1_percent",
    ]
