import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

from nemo_skills.dataset.mandarin_meeting_common import (
    cpcer_token_string,
    parse_long_textgrid,
    prepare_meeting_tree,
    resumable_http_download,
)
from nemo_skills.evaluation.evaluator.mandarin_meeting import evaluate_mandarin_meeting_sample

TEXTGRID = '''File type = "ooTextFile"
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
'''


def test_official_mandarin_normalization_and_char_tokens():
    assert cpcer_token_string("你 好，Ａｂ<sil>") == "你 好 A B"


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
    }
    result = evaluate_mandarin_meeting_sample(row)
    assert result["cpcer"] == 0
    assert result["cpcer_ref_chars"] == 4
    assert result["cpcer_errors"] == 0


def test_cpcer_counts_character_errors_after_official_normalization():
    row = {
        "expected_answer": "<spk:0> 你好，Ａｂ",
        "generation": "<spk:0> 你坏ab",
    }
    result = evaluate_mandarin_meeting_sample(row)
    assert result["cpcer_ref_chars"] == 4
    assert result["cpcer_errors"] == 1
    assert result["cpcer"] == 0.25


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
