import io
import json
import urllib.request
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from nemo_skills.dataset.mandarin_meeting_common import (
    cpcer_token_string,
    parse_long_textgrid,
    prepare_meeting_tree,
    resumable_http_download,
)
from nemo_skills.evaluation.evaluator.mandarin_meeting import evaluate_mandarin_meeting_sample

ALIMEETING_PREPARE = import_module("nemo_skills.dataset.alimeeting-sot.prepare")
AISHELL4_PREPARE = import_module("nemo_skills.dataset.aishell4-sot.prepare")


@pytest.mark.parametrize("benchmark", ["alimeeting-sot", "aishell4-sot"])
def test_mandarin_meeting_eval_args_are_shell_safe(benchmark):
    eval_args = import_module(f"nemo_skills.dataset.{benchmark}").EVAL_ARGS
    assert "speaker_token=spk" in eval_args
    assert "<spk:*>" not in eval_args

TEXTGRID = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "006-M"
        xmin = 0
        xmax = 1
        intervals: size = 2
        intervals [1]:
            xmin = 0
            xmax = 0.5
            text = "你好，Ａｂ"
        intervals [2]:
            xmin = 0.5
            xmax = 1
            text = ""
    item [2]:
        class = "IntervalTier"
        name = "007-F"
        xmin = 0
        xmax = 1
        intervals: size = 1
        intervals [1]:
            xmin = 0.25
            xmax = 0.75
            text = "世界<sil>"
"""


def test_pinned_mandarin_source_session_contracts():
    assert ALIMEETING_PREPARE.AUDIO_STEM_BY_SESSION == {
        "R8002_M8002": "R8002_M8002_MS802",
        "R8002_M8003": "R8002_M8003_MS803",
        "R8004_M8005": "R8004_M8005_MS803",
        "R8004_M8006": "R8004_M8006_MS805",
        "R8005_M8007": "R8005_M8007_MS806",
        "R8005_M8008": "R8005_M8008_MS806",
        "R8005_M8009": "R8005_M8009_MS802",
        "R8006_M8012": "R8006_M8012_MS803",
        "R8008_M8014": "R8008_M8014_MS807",
        "R8008_M8015": "R8008_M8015_MS808",
        "R8008_M8016": "R8008_M8016_MS808",
        "R8008_M8017": "R8008_M8017_MS808",
        "R8009_M8021": "R8009_M8021_MS810",
        "R8009_M8022": "R8009_M8022_MS810",
        "R8009_M8023": "R8009_M8023_MS811",
        "R8009_M8024": "R8009_M8024_MS811",
        "R8009_M8025": "R8009_M8025_MS811",
        "R8009_M8026": "R8009_M8026_MS812",
        "R8009_M8027": "R8009_M8027_MS812",
        "R8009_M8028": "R8009_M8028_MS812",
    }
    assert AISHELL4_PREPARE.EXPECTED_SESSION_IDS == (
        "L_R003S01C02",
        "L_R003S02C02",
        "L_R003S03C02",
        "L_R003S04C02",
        "L_R004S01C01",
        "L_R004S02C01",
        "L_R004S03C01",
        "L_R004S06C01",
        "M_R003S01C01",
        "M_R003S02C01",
        "M_R003S04C01",
        "M_R003S05C01",
        "S_R003S01C01",
        "S_R003S02C01",
        "S_R003S03C01",
        "S_R003S04C01",
        "S_R004S01C01",
        "S_R004S02C01",
        "S_R004S03C01",
        "S_R004S04C01",
    )


def test_corpus_specific_mandarin_normalization_and_char_tokens():
    assert cpcer_token_string("你 好，Ａｂ<sil>！", "alimeeting-sot") == "你 好 A B"
    assert cpcer_token_string("啊(~) 嗯(SIL)(/OVERLAP)？", "alimeeting-sot") == "啊 嗯"
    assert cpcer_token_string("abc`&，丶。、？ ａ", "alimeeting-sot") == "A B C A"
    assert cpcer_token_string("你 好，ABC123！", "aishell4-sot") == "你 好"
    assert cpcer_token_string("\u4dff一龥\u9fa6𠀀A", "aishell4-sot") == "一 龥"
    with pytest.raises(ValueError, match="Unsupported Mandarin meeting corpus"):
        cpcer_token_string("你好", "unknown")


def test_parse_long_textgrid(tmp_path: Path):
    path = tmp_path / "sample.TextGrid"
    path.write_text(TEXTGRID, encoding="utf-8")
    intervals = parse_long_textgrid(path)
    assert [(item.speaker, item.start, item.end, item.text) for item in intervals] == [
        ("006-M", 0.0, 0.5, "你好，Ａｂ"),
        ("007-F", 0.25, 0.75, "世界<sil>"),
    ]


def test_cpcer_is_speaker_permutation_invariant():
    row = {
        "expected_answer": "<spk:0> 你好 <spk:1> 世界",
        "generation": "<spk:0> 世界 <spk:1> 你好",
        "mandarin_corpus": "alimeeting-sot",
    }
    result = evaluate_mandarin_meeting_sample(row)
    assert result["cpcer"] == 0
    assert result["cpcer_ref_chars"] == 4
    assert result["cpcer_errors"] == 0


def test_cpcer_counts_character_errors_after_official_normalization():
    row = {
        "expected_answer": "<spk:0> 你好，Ａｂ",
        "generation": "<spk:0> 你坏ab",
        "mandarin_corpus": "alimeeting-sot",
    }
    result = evaluate_mandarin_meeting_sample(row)
    assert result["cpcer_ref_chars"] == 4
    assert result["cpcer_errors"] == 1
    assert result["cpcer"] == 0.25


def test_tagless_hypothesis_is_scored_as_one_speaker_not_all_deletions():
    row = {
        "expected_answer": "<spk:0> 你好",
        "generation": "你好",
        "mandarin_corpus": "aishell4-sot",
    }
    result = evaluate_mandarin_meeting_sample(row)
    assert result["cpcer"] == 0
    assert result["hypothesis_streams"] == {"s0": "你 好"}


def test_empty_hypothesis_remains_all_deletions():
    row = {
        "expected_answer": "<spk:0> 你好",
        "generation": "",
        "mandarin_corpus": "aishell4-sot",
    }
    result = evaluate_mandarin_meeting_sample(row)
    assert result["cpcer"] == 1
    assert result["hypothesis_streams"] == {}


def test_evaluator_config_can_pin_normalization_for_legacy_rows():
    row = {"expected_answer": "<spk:0> 你好ABC", "generation": "<spk:0> 你好"}
    result = evaluate_mandarin_meeting_sample(row, normalization="aishell4-sot")
    assert result["cpcer"] == 0


def test_prepare_meeting_tree_materializes_first_channel(tmp_path: Path):
    source_audio = tmp_path / "source" / "wav"
    source_text = tmp_path / "source" / "TextGrid"
    source_audio.mkdir(parents=True)
    source_text.mkdir(parents=True)
    samples = np.stack(
        [np.linspace(-0.5, 0.5, 16000, dtype=np.float32), np.ones(16000, dtype=np.float32) * 0.25],
        axis=1,
    )
    sf.write(source_audio / "session.flac", samples, 16000)
    (source_text / "session.TextGrid").write_text(TEXTGRID, encoding="utf-8")
    output = tmp_path / "prepared"
    manifest = prepare_meeting_tree(
        benchmark_name="fixture",
        source_audio_dir=source_audio,
        source_textgrid_dir=source_text,
        output_dir=output,
        audio_prefix="/data/fixture/audio",
        source_metadata={"source_revision": "fixture"},
    )
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["num_speakers"] == 2
    assert rows[0]["expected_answer"].startswith("<spk:0> 你好，Ａｂ <spk:1> 世界<sil>")
    info = sf.info(output / "audio" / "session.wav")
    assert info.channels == 1
    assert info.frames == 16000


def test_prepare_requires_exact_audio_stem_mapping(tmp_path: Path):
    source_audio = tmp_path / "source" / "wav"
    source_text = tmp_path / "source" / "TextGrid"
    source_audio.mkdir(parents=True)
    source_text.mkdir(parents=True)
    sf.write(source_audio / "session_MS1.wav", np.zeros(1600, dtype=np.float32), 16000)
    (source_text / "session.TextGrid").write_text(TEXTGRID, encoding="utf-8")
    common = {
        "benchmark_name": "fixture",
        "source_audio_dir": source_audio,
        "source_textgrid_dir": source_text,
        "audio_prefix": "/data/fixture/audio",
        "source_metadata": {"source_revision": "fixture"},
        "no_audio": True,
        "expected_session_ids": ("session",),
    }

    with pytest.raises(FileNotFoundError, match="exact source audio stem 'session'"):
        prepare_meeting_tree(output_dir=tmp_path / "unmapped", **common)

    manifest = prepare_meeting_tree(
        output_dir=tmp_path / "mapped",
        audio_stem_by_session={"session": "session_MS1"},
        **common,
    )
    row = json.loads(manifest.read_text().strip())
    assert row["extra_fields"]["source_audio_stem"] == "session_MS1"


def test_prepare_rejects_incomplete_full_session_contract(tmp_path: Path):
    source_audio = tmp_path / "source" / "wav"
    source_text = tmp_path / "source" / "TextGrid"
    source_audio.mkdir(parents=True)
    source_text.mkdir(parents=True)
    sf.write(source_audio / "session.wav", np.zeros(1600, dtype=np.float32), 16000)
    (source_text / "session.TextGrid").write_text(TEXTGRID, encoding="utf-8")
    with pytest.raises(ValueError, match="missing=\\['missing'\\]"):
        prepare_meeting_tree(
            benchmark_name="fixture",
            source_audio_dir=source_audio,
            source_textgrid_dir=source_text,
            output_dir=tmp_path / "prepared",
            audio_prefix="/data/fixture/audio",
            source_metadata={"source_revision": "fixture"},
            no_audio=True,
            expected_session_ids=("session", "missing"),
        )


def test_resumable_http_download_preserves_partial_bytes(tmp_path: Path, monkeypatch):
    destination = tmp_path / "archive.tar.gz"
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.write_bytes(b"abc")
    captured = {}

    class FakeResponse(io.BytesIO):
        def __init__(self, data: bytes):
            super().__init__(data)
            self.status = 206
            self.headers = {"Content-Range": "bytes 3-5/6"}

        def getcode(self):
            return self.status

    def fake_urlopen(request, timeout):
        captured["range"] = request.get_header("Range")
        captured["timeout"] = timeout
        return FakeResponse(b"def")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = resumable_http_download(
        "https://example.invalid/archive.tar.gz",
        destination,
        expected_size=6,
        user_agent="test",
    )

    assert result == destination
    assert destination.read_bytes() == b"abcdef"
    assert not partial.exists()
    assert captured == {"range": "bytes=3-", "timeout": 300}
