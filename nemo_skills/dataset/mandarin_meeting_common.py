# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared, protocol-frozen preparation helpers for Mandarin meetings.

AliMeeting and AISHELL-4 both publish long-format Praat TextGrid annotations.
The parser below intentionally supports that frozen format without adding a
runtime dependency on the mutable ``textgrid`` package. References are emitted
as serialized speaker turns; the evaluator applies the official corpus text
normalization and character tokenization before minimum speaker permutation.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

SPEAKER_PROMPT = "Transcribe the mixed audio. Put a distinct <spk:N> tag before every speaker turn."
_ITEM_RE = re.compile(r"^\s*item\s*\[\d+\]:\s*$")
_INTERVAL_RE = re.compile(r"^\s*intervals\s*\[\d+\]:\s*$")
_ASSIGN_RE = re.compile(r"^\s*(name|xmin|xmax|text)\s*=\s*(.*?)\s*$")
_MARKERS = ("<sil>", "<%>", "<->", "<$>", "<#>", "<_>", "<space>")
_DELETE_CHARS = "`&,，丶。、？"
_FULLWIDTH_ASCII = str.maketrans({"Ａ": "A", "ａ": "A", "ｂ": "B", "ｃ": "C", "ｋ": "K", "ｔ": "T"})


@dataclass(frozen=True)
class MeetingInterval:
    speaker: str
    start: float
    end: float
    text: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resumable_http_download(
    url: str,
    destination: Path,
    *,
    expected_size: int,
    user_agent: str,
    timeout: int = 300,
) -> Path:
    """Download an immutable large object with validated byte-range resume."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        actual_size = destination.stat().st_size
        if actual_size == expected_size:
            return destination
        raise ValueError(f"Existing final download has {actual_size} bytes; expected {expected_size}: {destination}")

    partial = destination.with_suffix(destination.suffix + ".partial")
    offset = partial.stat().st_size if partial.is_file() else 0
    if offset > expected_size:
        raise ValueError(f"Partial download has {offset} bytes; expected at most {expected_size}: {partial}")
    if offset == expected_size:
        partial.replace(destination)
        return destination

    headers = {"User-Agent": user_agent}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        if error.code == 416 and partial.is_file() and partial.stat().st_size == expected_size:
            partial.replace(destination)
            return destination
        raise

    status = getattr(response, "status", None) or response.getcode()
    mode = "ab"
    if offset:
        if status == 206:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {offset}-"):
                response.close()
                raise ValueError(f"Server resumed at an unexpected range {content_range!r}; requested byte {offset}")
        elif status == 200:
            offset = 0
            mode = "wb"
        else:
            response.close()
            raise ValueError(f"Unexpected HTTP {status} while resuming {url}")
    elif status not in {200, 206}:
        response.close()
        raise ValueError(f"Unexpected HTTP {status} while downloading {url}")

    with response, partial.open(mode) as output:
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    actual_size = partial.stat().st_size
    if actual_size != expected_size:
        raise OSError(f"Partial download has {actual_size} bytes after transfer; expected {expected_size}: {partial}")
    partial.replace(destination)
    return destination


def _praat_string(value: str) -> str:
    value = value.strip()
    if len(value) < 2 or value[0] != '"' or value[-1] != '"':
        raise ValueError(f"Expected a quoted Praat string, got {value!r}")
    return value[1:-1].replace('""', '"')


def parse_long_textgrid(path: Path) -> list[MeetingInterval]:
    """Parse the official long TextGrid format used by both corpora."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    intervals: list[MeetingInterval] = []
    speaker: str | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        if _ITEM_RE.match(line):
            speaker = None
            index += 1
            continue
        match = _ASSIGN_RE.match(line)
        if match and match.group(1) == "name":
            speaker = _praat_string(match.group(2))
            index += 1
            continue
        if not _INTERVAL_RE.match(line):
            index += 1
            continue
        if speaker is None:
            raise ValueError(f"Interval encountered before tier name in {path}:{index + 1}")
        values: dict[str, str] = {}
        index += 1
        while index < len(lines) and len(values) < 3:
            field = _ASSIGN_RE.match(lines[index])
            if field and field.group(1) in {"xmin", "xmax", "text"}:
                values[field.group(1)] = field.group(2)
            index += 1
        if set(values) != {"xmin", "xmax", "text"}:
            raise ValueError(f"Incomplete interval in {path} near line {index + 1}")
        text = _praat_string(values["text"])
        if text:
            start = float(values["xmin"])
            end = float(values["xmax"])
            if end <= start:
                raise ValueError(f"Non-positive interval in {path}: {start}..{end}")
            intervals.append(MeetingInterval(speaker=speaker, start=start, end=end, text=text))
    if not intervals:
        raise ValueError(f"No non-empty speech intervals found in {path}")
    return intervals


def normalize_alimeeting_text(text: str) -> str:
    """Port the complete released AliMeeting ASR text cleanup path."""
    normalized = str(text)
    for marker in _MARKERS:
        normalized = normalized.replace(marker, "")
    normalized = normalized.translate(_FULLWIDTH_ASCII)
    normalized = normalized.translate(str.maketrans("", "", _DELETE_CHARS))
    for marker in ("！", "？", "(~)", "(SIL)", "(/OVERLAP)"):
        normalized = normalized.replace(marker, "")
    normalized = re.sub(r"\s+", "", normalized)
    if re.search(r"[A-Za-z]", normalized):
        normalized = normalized.upper()
    return normalized


def normalize_aishell4_text(text: str) -> str:
    """Apply AISHELL-4's released Han-only U+4E00--U+9FA5 rule."""
    return re.sub(r"[^\u4e00-\u9fa5]", "", str(text))


def normalize_mandarin_meeting_text(text: str, corpus: str) -> str:
    """Select the released corpus-specific reference normalization."""
    if corpus in {"alimeeting", "alimeeting-sot"}:
        return normalize_alimeeting_text(text)
    if corpus in {"aishell4", "aishell4-sot"}:
        return normalize_aishell4_text(text)
    raise ValueError(f"Unsupported Mandarin meeting corpus: {corpus!r}")


def cpcer_token_string(text: str, corpus: str) -> str:
    """Return whitespace-delimited Unicode characters for cpCER edit distance."""
    normalized = normalize_mandarin_meeting_text(text, corpus)
    return " ".join(character for character in normalized if not character.isspace())


def intervals_to_sot(intervals: list[MeetingInterval]) -> tuple[str, int, int]:
    """Serialize turns chronologically with stable first-appearance speaker IDs."""
    ordered = sorted(intervals, key=lambda item: (item.start, item.end, item.speaker))
    speaker_ids: dict[str, int] = {}
    chunks: list[str] = []
    previous: int | None = None
    changes = 0
    for interval in ordered:
        speaker_id = speaker_ids.setdefault(interval.speaker, len(speaker_ids))
        if previous is not None and speaker_id != previous:
            changes += 1
        previous = speaker_id
        chunks.append(f"<spk:{speaker_id}> {interval.text.strip()}")
    return " ".join(chunks), len(speaker_ids), changes


def write_first_channel_wav(source: Path, destination: Path) -> float:
    """Stream channel zero into a PCM16 mono WAV without loading a meeting in RAM."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return float(sf.info(destination).duration)
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.unlink(missing_ok=True)
    with (
        sf.SoundFile(source) as input_audio,
        sf.SoundFile(
            partial,
            mode="w",
            samplerate=input_audio.samplerate,
            channels=1,
            format="WAV",
            subtype="PCM_16",
        ) as output_audio,
    ):
        while True:
            block = input_audio.read(1_048_576, dtype="float32", always_2d=True)
            if len(block) == 0:
                break
            output_audio.write(block[:, :1])
    partial.replace(destination)
    return float(sf.info(destination).duration)


def build_meeting_record(
    *,
    benchmark_name: str,
    session_id: str,
    source_audio: Path,
    textgrid: Path,
    audio_dir: Path,
    audio_prefix: str,
    source_metadata: dict,
    no_audio: bool = False,
) -> dict:
    intervals = parse_long_textgrid(textgrid)
    reference, num_speakers, num_changes = intervals_to_sot(intervals)
    source_info = sf.info(source_audio)
    if source_info.frames <= 0 or source_info.channels <= 0 or source_info.samplerate <= 0:
        raise ValueError(f"Invalid source audio header for {session_id}: {source_info}")
    destination = audio_dir / f"{session_id}.wav"
    if no_audio:
        duration = float(source_info.duration)
    else:
        duration = write_first_channel_wav(source_audio, destination)
    audio_path = f"{audio_prefix.rstrip('/')}/{session_id}.wav"
    audio = {"path": audio_path, "duration": duration}
    return {
        "id": session_id,
        "sample_id": session_id,
        "audio_filepath": audio_path,
        "duration": duration,
        "expected_answer": reference,
        "text": reference,
        "messages": [{"role": "user", "content": SPEAKER_PROMPT, "audio": audio}],
        "source_lang": "zh",
        "target_lang": "zh",
        "task_type": "Mandarin-MS-ASR",
        "category": "nonjudge",
        "mandarin_corpus": benchmark_name,
        "num_speakers": num_speakers,
        "num_changes": num_changes,
        "subset_for_metrics": "test",
        "extra_fields": {
            **source_metadata,
            "source_audio": str(source_audio),
            "source_audio_stem": source_audio.stem,
            "source_audio_channels": source_info.channels,
            "source_audio_frames": source_info.frames,
            "source_audio_sample_rate": source_info.samplerate,
            "source_textgrid": str(textgrid),
            "input_channel": 0,
            "reference_intervals": len(intervals),
            "normalization": f"released-{benchmark_name}",
            "metric": "full-session-cpCER",
        },
    }


def prepare_meeting_tree(
    *,
    benchmark_name: str,
    source_audio_dir: Path,
    source_textgrid_dir: Path,
    output_dir: Path,
    audio_prefix: str,
    source_metadata: dict,
    no_audio: bool = False,
    max_samples: int | None = None,
    expected_session_ids: tuple[str, ...] | None = None,
    audio_stem_by_session: dict[str, str] | None = None,
) -> Path:
    audio_by_stem: dict[str, Path] = {}
    for path in sorted(source_audio_dir.rglob("*")):
        if path.suffix.lower() not in {".wav", ".flac"}:
            continue
        previous = audio_by_stem.setdefault(path.stem, path)
        if previous != path:
            raise ValueError(f"Duplicate source audio stem {path.stem!r}: {previous}, {path}")

    all_textgrids = sorted(source_textgrid_dir.rglob("*.TextGrid"))
    if not all_textgrids:
        raise ValueError(f"No TextGrid files found under {source_textgrid_dir}")
    textgrid_by_session: dict[str, Path] = {}
    for textgrid in all_textgrids:
        previous = textgrid_by_session.setdefault(textgrid.stem, textgrid)
        if previous != textgrid:
            raise ValueError(f"Duplicate TextGrid session {textgrid.stem!r}: {previous}, {textgrid}")

    if expected_session_ids is not None:
        if len(expected_session_ids) != len(set(expected_session_ids)):
            raise ValueError("Expected session contract contains duplicate IDs")
        expected = set(expected_session_ids)
        discovered = set(textgrid_by_session)
        unexpected = sorted(discovered - expected)
        missing = sorted(expected - discovered)
        if unexpected or (max_samples is None and missing):
            raise ValueError(f"{benchmark_name} session contract mismatch: missing={missing}, unexpected={unexpected}")
        if audio_stem_by_session is not None and set(audio_stem_by_session) != expected:
            raise ValueError(f"{benchmark_name} audio mapping does not cover the exact session contract")

    textgrids = all_textgrids[:max_samples] if max_samples is not None else all_textgrids
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    manifest = output_dir / "test.jsonl"
    rows = []
    for textgrid in textgrids:
        session_id = textgrid.stem
        audio_stem = (audio_stem_by_session or {}).get(session_id, session_id)
        source_audio = audio_by_stem.get(audio_stem)
        if source_audio is None:
            raise FileNotFoundError(f"Expected exact source audio stem {audio_stem!r} for session {session_id!r}")
        rows.append(
            build_meeting_record(
                benchmark_name=benchmark_name,
                session_id=session_id,
                source_audio=source_audio,
                textgrid=textgrid,
                audio_dir=audio_dir,
                audio_prefix=audio_prefix,
                source_metadata=source_metadata,
                no_audio=no_audio,
            )
        )

    partial_manifest = manifest.with_suffix(manifest.suffix + ".partial")
    with partial_manifest.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    partial_manifest.replace(manifest)
    receipt = {
        "benchmark": benchmark_name,
        "rows": len(rows),
        "sessions": [row["id"] for row in rows],
        "source_audio_stems": {row["id"]: row["extra_fields"]["source_audio_stem"] for row in rows},
        **source_metadata,
    }
    receipt_path = output_dir / "source_receipt.json"
    partial_receipt = receipt_path.with_suffix(receipt_path.suffix + ".partial")
    partial_receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    partial_receipt.replace(receipt_path)
    return manifest


def safe_stream_extract_tar_gz(archive: Path, destination: Path, required_component: str) -> None:
    """Extract only one subtree from a large gzip tar in one sequential pass."""
    import tarfile

    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tar:
            parts = Path(member.name).parts
            if required_component not in parts:
                continue
            if Path(member.name).is_absolute() or ".." in parts or member.issym() or member.islnk():
                raise ValueError(f"Unsafe archive member: {member.name!r}")
            target = destination.joinpath(*parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            source = tar.extractfile(member)
            if source is None:
                raise ValueError(f"Could not read archive member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".partial")
            with source, partial.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
            partial.replace(target)
