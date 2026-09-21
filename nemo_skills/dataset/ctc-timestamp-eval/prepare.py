# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize the self-contained ``ctc-timestamp-eval`` data closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import unicodedata
import wave
from collections import defaultdict
from pathlib import Path

BENCHMARK_NAME = "ctc-timestamp-eval"
D12_LANGUAGE_CODES = ("ar", "de", "en", "es", "fr", "hi", "it", "ja", "ko", "pt", "ru", "th", "tr", "vi", "zh")
D12_SUBSETS = tuple(f"d12a-{language}" for language in D12_LANGUAGE_CODES)
LSM_MIXES = ("lsm-test-clean-1mix", "lsm-test-clean-2mix", "lsm-test-clean-3mix")
SUBSETS = ("ch109", "ami-ihm-test", "lsm", *D12_SUBSETS)
DEFAULT_AUDIO_PREFIX = Path("/data/ctc-timestamp-eval/audio")
DEFAULT_GT_PREFIX = Path("/data/ctc-timestamp-eval/ground_truth")
_AUDIO_STEM_RE = re.compile(r"^(?P<session>.+)_(?P<offset>\d+(?:\.\d+)?)_(?P<duration>\d+(?:\.\d+)?)_\d+$")
_INTERVAL_RE = re.compile(
    r"intervals \[\d+\]:\s+xmin = (?P<start>[0-9.eE+-]+)\s+"
    r"xmax = (?P<end>[0-9.eE+-]+)\s+text = \"(?P<word>[^\"]*)\"",
    re.DOTALL,
)


def _selected_files() -> dict:
    path = Path(__file__).with_name("selected_files.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audio_info(path: Path) -> tuple[float, int]:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate(), audio.getframerate()


def parse_cropped_audio_name(path: Path) -> tuple[str, float, float]:
    match = _AUDIO_STEM_RE.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"Cannot parse session/offset/duration from {path.name}")
    return match.group("session"), float(match.group("offset")), float(match.group("duration"))


def read_ctm(path: Path) -> list[dict]:
    words = []
    with path.open("rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 5:
                raise ValueError(f"Malformed CTM row at {path}:{line_number}")
            start = float(fields[2])
            duration = float(fields[3])
            words.append(
                {
                    "word": fields[4],
                    "start": start,
                    "end": start + duration,
                    "speaker": fields[7] if len(fields) > 7 else fields[0],
                }
            )
    return words


def crop_and_merge_ctms(paths: list[Path], offset: float, duration: float) -> list[dict]:
    """Merge channel CTMs, retaining words whose midpoint is inside the audio crop."""
    crop_end = offset + duration
    words = []
    for path in paths:
        for word in read_ctm(path):
            midpoint = (word["start"] + word["end"]) / 2
            if not offset <= midpoint < crop_end:
                continue
            words.append(
                {
                    **word,
                    "start": max(0.0, word["start"] - offset),
                    "end": min(duration, word["end"] - offset),
                }
            )
    return canonicalize_speakers(words)


def canonicalize_speakers(words: list[dict]) -> list[dict]:
    words = sorted(words, key=lambda word: (word["start"], word["end"], word["speaker"]))
    speaker_map = {}
    for word in words:
        speaker = str(word["speaker"])
        speaker_map.setdefault(speaker, f"speaker{len(speaker_map)}")
        word["speaker"] = speaker_map[speaker]
        word["start"] = round(float(word["start"]), 3)
        word["end"] = round(float(word["end"]), 3)
    return words


def parse_textgrid_words(path: Path) -> tuple[list[dict], float]:
    text = path.read_text(encoding="utf-8")
    tier = re.search(
        r"item \[\d+\]:\s+class = \"IntervalTier\"\s+name = \"words\"(?P<body>.*?)(?=\n\s*item \[\d+\]:|\Z)",
        text,
        re.DOTALL,
    )
    if tier is None:
        raise ValueError(f"No words IntervalTier found in {path}")
    words = []
    for match in _INTERVAL_RE.finditer(tier.group("body")):
        word = match.group("word").strip()
        if not word or word == "<eps>":
            continue
        words.append(
            {
                "word": word,
                "start": round(float(match.group("start")), 3),
                "end": round(float(match.group("end")), 3),
                "speaker": "speaker0",
            }
        )
    xmax = re.search(r"^xmax = ([0-9.eE+-]+)", text, re.MULTILINE)
    if not words or xmax is None:
        raise ValueError(f"No scoreable word intervals found in {path}")
    return words, float(xmax.group(1))


def _normalized_token(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    start = 0
    end = len(text)
    while start < end and unicodedata.category(text[start])[0] not in "LMN":
        start += 1
    while end > start and unicodedata.category(text[end - 1])[0] not in "LMN":
        end -= 1
    return text[start:end]


def reconcile_textgrid_words(words: list[dict], transcript: str) -> list[dict]:
    """Map TextGrid intervals onto the exact LibriSpeechMix transcript tokens."""
    tokens = transcript.casefold().split()
    reference = [_normalized_token(token) for token in tokens]
    aligned = [_normalized_token(item["word"]) for item in words]
    output = []
    ref_start = aligned_start = 0
    while ref_start < len(reference) and aligned_start < len(aligned):
        ref_count = aligned_count = 1
        if reference[ref_start] != aligned[aligned_start]:
            candidates = []
            for candidate_ref_count in range(1, min(4, len(reference) - ref_start) + 1):
                for candidate_aligned_count in range(1, min(4, len(aligned) - aligned_start) + 1):
                    ref_text = "".join(
                        character
                        for token in reference[ref_start : ref_start + candidate_ref_count]
                        for character in token
                        if unicodedata.category(character)[0] in "LMN"
                    )
                    aligned_text = "".join(
                        character
                        for token in aligned[aligned_start : aligned_start + candidate_aligned_count]
                        for character in token
                        if unicodedata.category(character)[0] in "LMN"
                    )
                    if ref_text == aligned_text and (candidate_ref_count, candidate_aligned_count) != (1, 1):
                        candidates.append(
                            (
                                candidate_ref_count + candidate_aligned_count,
                                candidate_ref_count,
                                candidate_aligned_count,
                            )
                        )
            if candidates:
                _, ref_count, aligned_count = min(candidates)
        ref_end = ref_start + ref_count
        aligned_end = aligned_start + aligned_count
        span_start = float(words[aligned_start]["start"])
        span_end = float(words[aligned_end - 1]["end"])
        weights = [max(1, len(reference[index])) for index in range(ref_start, ref_end)]
        total_weight = sum(weights)
        consumed_weight = 0
        for relative_index, (token, weight) in enumerate(zip(tokens[ref_start:ref_end], weights)):
            token_start = span_start + (span_end - span_start) * consumed_weight / total_weight
            consumed_weight += weight
            token_end = (
                span_end
                if relative_index == len(weights) - 1
                else span_start + (span_end - span_start) * consumed_weight / total_weight
            )
            output.append(
                {
                    "word": token,
                    "start": round(token_start, 3),
                    "end": round(token_end, 3),
                    "speaker": "speaker0",
                }
            )
        ref_start = ref_end
        aligned_start = aligned_end
    if ref_start != len(reference) or aligned_start != len(aligned):
        raise ValueError(
            f"Cannot reconcile trailing transcript tokens {tokens[ref_start:]} with "
            f"TextGrid tokens {[item['word'] for item in words[aligned_start:]]}"
        )
    if [item["word"] for item in output] != tokens:
        raise ValueError("Reconciled TextGrid words do not reproduce the LibriSpeechMix transcript")
    return output


def lsm_selected_indices(selected: dict) -> dict[str, list[int]]:
    rng = random.Random(int(selected["seed"]))
    indices = {}
    for mix in LSM_MIXES:
        config = selected["mixes"][mix]
        indices[mix] = sorted(rng.sample(range(int(config["source_files"])), int(config["selected_files"])))
    return indices


def d12_reference_words(row: dict) -> list[dict]:
    """Convert a D12 alignment row, deriving every end from the next start."""
    alignment = row.get("alignment", {}).get("word")
    if not isinstance(alignment, list) or not alignment:
        raise ValueError("D12 row has no alignment.word entries")
    alignment = [item for item in alignment if item and item[0] != "[Timestamp alignment GT labels]"]
    if not alignment:
        return []
    offset = float(row.get("offset", 0.0))
    segment_end = offset + float(row["duration"])
    words = []
    for index, item in enumerate(alignment):
        if not isinstance(item, list) or len(item) < 2:
            raise ValueError(f"Invalid D12 alignment word at index {index}")
        absolute_start = float(item[1])
        absolute_end = float(alignment[index + 1][1]) if index + 1 < len(alignment) else segment_end
        if absolute_end < absolute_start:
            raise ValueError(f"D12 word starts are not monotonic at index {index}")
        words.append(
            {
                "word": str(item[0]),
                "start": round(max(0.0, absolute_start - offset), 3),
                "end": round(min(float(row["duration"]), absolute_end - offset), 3),
                "speaker": "speaker0",
            }
        )
    return words


def _sot_text(words: list[dict]) -> str:
    parts = []
    last_speaker = None
    for item in words:
        if item["speaker"] != last_speaker:
            parts.append(f"<spk:{int(item['speaker'].removeprefix('speaker'))}>")
            last_speaker = item["speaker"]
        parts.append(item["word"])
    return " ".join(parts)


def _write_ctm(path: Path, sample_id: str, words: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="utf-8") as stream:
        for item in words:
            duration = max(0.0, item["end"] - item["start"])
            stream.write(
                f"{sample_id} 1 {item['start']:.3f} {duration:.3f} {item['word']} 1.0 lex {item['speaker']}\n"
            )


def _record(
    subset: str,
    sample_id: str,
    audio_name: str,
    duration: float,
    words: list[dict],
    audio_prefix: Path,
    gt_prefix: Path,
) -> dict:
    speakers = list(dict.fromkeys(item["speaker"] for item in words))
    multispeaker = len(speakers) > 1
    audio_path = str(audio_prefix / subset / audio_name)
    gt_path = str(gt_prefix / subset / f"{sample_id}.ctm")
    text = _sot_text(words) if multispeaker else " ".join(item["word"] for item in words)
    prompt = (
        "Transcribe the mixed audio with speaker tags like <spk:0> hello <spk:1> yes, "
        "and return word start/end timestamps."
        if multispeaker
        else "Transcribe the audio and return word start/end timestamps."
    )
    return {
        "sample_id": sample_id,
        "dataset_id": f"ctc-timestamp-eval.{subset}",
        "subset_for_metrics": subset,
        "audio_filepath": audio_path,
        "ground_truth_ctm": gt_path,
        "duration": round(duration, 3),
        "text": text,
        "expected_answer": text,
        "reference_words": words,
        "num_speakers": len(speakers),
        "task_type": "MS-ASR" if multispeaker else "ASR",
        "taskname": "asr",
        "category": "nonjudge",
        "messages": [{"role": "user", "content": prompt, "audio": {"path": audio_path, "duration": duration}}],
    }


def _publish_subset(data_dir: Path, subset: str, records: list[dict]) -> None:
    output = data_dir / subset / "test.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".jsonl.tmp")
    with temporary.open("wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(output)


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != source.stat().st_size:
            raise ValueError(f"Existing target has a different size: {target}")
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _extract_audio_segment(source: Path, target: Path, offset: float, duration: float) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    temporary = target.with_suffix(".wav.tmp")
    command = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(offset),
        "-t",
        str(duration),
        "-i",
        str(source),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _materialize_ctm_subset(
    *,
    subset: str,
    audio_root: Path,
    ctm_root: Path,
    selected_names: list[str],
    data_dir: Path,
    audio_prefix: Path,
    gt_prefix: Path,
) -> list[dict]:
    records = []
    for audio_name in selected_names:
        source_audio = audio_root / audio_name
        if not source_audio.is_file():
            raise FileNotFoundError(source_audio)
        session, offset, declared_duration = parse_cropped_audio_name(source_audio)
        if subset == "ch109":
            ctm_paths = [ctm_root / f"{session}_c1.ctm", ctm_root / f"{session}_c2.ctm"]
        else:
            ctm_paths = [ctm_root / f"{session}.ctm"]
        missing = [path for path in ctm_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing CTM inputs: {missing}")
        actual_duration, _ = _audio_info(source_audio)
        if abs(actual_duration - declared_duration) > 0.1:
            raise ValueError(
                f"Filename/audio duration mismatch for {source_audio}: {declared_duration} vs {actual_duration}"
            )
        words = crop_and_merge_ctms(ctm_paths, offset, actual_duration)
        if not words:
            raise ValueError(f"No CTM words overlap {source_audio}")
        target_audio = data_dir / "audio" / subset / audio_name
        target_audio.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_audio, target_audio)
        gt_path = data_dir / "ground_truth" / subset / f"{source_audio.stem}.ctm"
        _write_ctm(gt_path, source_audio.stem, words)
        records.append(_record(subset, source_audio.stem, audio_name, actual_duration, words, audio_prefix, gt_prefix))
    _publish_subset(data_dir, subset, records)
    return records


def _materialize_lsm(
    *,
    textgrid_root: Path,
    skills_root: Path,
    metadata_root: Path,
    selected: dict,
    data_dir: Path,
    audio_prefix: Path,
    gt_prefix: Path,
) -> list[dict]:
    records = []
    selected_indices = lsm_selected_indices(selected)
    for mix in LSM_MIXES:
        config = selected["mixes"][mix]
        skills_manifest = skills_root / mix / "test.jsonl"
        metadata_manifest = metadata_root / f"{mix.removeprefix('lsm-')}.jsonl"
        if _sha256(skills_manifest) != config["manifest_sha256"]:
            raise ValueError(f"LSM source manifest changed: {skills_manifest}")
        if _sha256(metadata_manifest) != config["metadata_manifest_sha256"]:
            raise ValueError(f"LSM metadata manifest changed: {metadata_manifest}")
        skills_rows = _read_jsonl(skills_manifest)
        metadata_rows = _read_jsonl(metadata_manifest)
        expected_source_files = int(config["source_files"])
        if len(skills_rows) != expected_source_files or len(metadata_rows) != expected_source_files:
            raise ValueError(
                f"Expected {expected_source_files} {mix} source rows, found "
                f"{len(skills_rows)} Skills and {len(metadata_rows)} metadata rows"
            )

        for row_index in selected_indices[mix]:
            source_record = skills_rows[row_index]
            metadata = metadata_rows[row_index]
            sample_id = str(source_record["sample_id"])
            audio_name = Path(source_record["audio_filepath"]).name
            session, _, _ = parse_cropped_audio_name(Path(audio_name))
            if Path(metadata["id"]).name != session:
                raise ValueError(f"LSM row mismatch at {mix}[{row_index}]: {session} != {metadata['id']}")

            duration = float(source_record["duration"])
            words = []
            for speaker_index, (wav_path, transcript, delay, source_duration) in enumerate(
                zip(metadata["wavs"], metadata["texts"], metadata["delays"], metadata["durations"])
            ):
                source_stem = Path(wav_path).stem
                textgrid = textgrid_root / source_stem.split("-", 1)[0] / f"{source_stem}.TextGrid"
                aligned_words, textgrid_duration = parse_textgrid_words(textgrid)
                if abs(textgrid_duration - float(source_duration)) > 0.02:
                    raise ValueError(
                        f"TextGrid/source duration mismatch for {textgrid}: {textgrid_duration} vs {source_duration}"
                    )
                for item in reconcile_textgrid_words(aligned_words, transcript):
                    words.append(
                        {
                            "word": item["word"],
                            "start": round(float(item["start"]) + float(delay), 3),
                            "end": round(min(duration, float(item["end"]) + float(delay)), 3),
                            "speaker": f"speaker{speaker_index}",
                        }
                    )
            words.sort(key=lambda item: (item["start"], item["end"], item["speaker"]))

            source_audio = skills_root / "audio" / mix / audio_name
            target_audio = data_dir / "audio" / "lsm" / audio_name
            _link_or_copy(source_audio, target_audio)
            gt_path = data_dir / "ground_truth" / "lsm" / f"{sample_id}.ctm"
            _write_ctm(gt_path, sample_id, words)
            record = _record("lsm", sample_id, audio_name, duration, words, audio_prefix, gt_prefix)
            prompt = (
                "Transcribe the mixed audio with speaker tags like <spk:0> hello <spk:1> yes, "
                "and return word start/end timestamps."
            )
            record.update(
                text=source_record["text"],
                expected_answer=source_record["expected_answer"],
                num_speakers=int(source_record["num_speakers"]),
                task_type="MS-ASR",
                source_lang="en",
                target_lang="en",
                source_mix=mix,
                subset_for_metrics=mix,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "audio": {"path": record["audio_filepath"], "duration": duration},
                    }
                ],
            )
            records.append(record)

    if len(records) != int(selected["expected_files"]):
        raise ValueError(f"Expected {selected['expected_files']} LSM rows, found {len(records)}")
    _publish_subset(data_dir, "lsm", records)
    return records


def _d12_language_sessions(label_root: Path, languages: list[str]) -> dict[str, set[str]]:
    sessions = {}
    owners = {}
    for language in languages:
        manifest = label_root / language / f"d12a_{language}_manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        language_sessions = set()
        with manifest.open("rt", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                session = Path(json.loads(line)["audio_filepath"]).stem
                previous = owners.setdefault(session, language)
                if previous != language:
                    raise ValueError(f"D12 session {session} occurs in both {previous} and {language}")
                language_sessions.add(session)
        sessions[language] = language_sessions
    return sessions


def _materialize_d12a(
    *,
    languages: list[str],
    aligned_manifest: Path,
    label_root: Path,
    segment_audio_root: Path | None,
    selected: dict,
    data_dir: Path,
    audio_prefix: Path,
    gt_prefix: Path,
) -> dict[str, int]:
    label_sessions = _d12_language_sessions(label_root, languages)
    session_to_language = {}
    for language in languages:
        pinned = set(selected["languages"][language]["sessions"])
        if not pinned <= label_sessions[language]:
            missing = sorted(pinned - label_sessions[language])
            raise ValueError(f"Pinned D12 {language} sessions are absent from its language manifest: {missing}")
        session_to_language.update(dict.fromkeys(pinned, language))

    records = defaultdict(list)
    with aligned_manifest.open("rt", encoding="utf-8") as stream:
        for row_index, line in enumerate(stream):
            if not line.strip():
                continue
            row = json.loads(line)
            source_path = Path(row["audio_filepath"])
            language = session_to_language.get(source_path.parent.name)
            if language is None:
                continue
            words = d12_reference_words(row)
            if not words:
                continue
            subset = f"d12a-{language}"
            source_name = f"test.d12a-{row_index:05d}-{source_path.stem}.wav"
            expected_duration = float(row["duration"])
            sample_id = f"d12a-{row_index:05d}-{source_path.stem}"
            audio_name = f"{sample_id}.wav"
            target_audio = data_dir / "audio" / subset / audio_name
            if segment_audio_root is not None:
                source_audio = segment_audio_root / source_name
                if not source_audio.is_file():
                    raise FileNotFoundError(source_audio)
                _link_or_copy(source_audio, target_audio)
            else:
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                _extract_audio_segment(source_path, target_audio, float(row.get("offset", 0.0)), expected_duration)
            audio_duration, _ = _audio_info(target_audio)
            if abs(audio_duration - expected_duration) > 0.1:
                raise ValueError(
                    f"D12 segment duration mismatch for {target_audio}: {expected_duration} vs {audio_duration}"
                )
            gt_path = data_dir / "ground_truth" / subset / f"{sample_id}.ctm"
            _write_ctm(gt_path, sample_id, words)
            record = _record(
                subset,
                sample_id,
                audio_name,
                expected_duration,
                words,
                audio_prefix,
                gt_prefix,
            )
            record.update(source_lang=language, target_lang=language, session_id=source_path.parent.name)
            records[language].append(record)

    counts = {}
    for language in languages:
        expected = int(selected["languages"][language]["expected_files"])
        if len(records[language]) != expected:
            raise ValueError(f"Expected {expected} D12 {language} rows, found {len(records[language])}")
        _publish_subset(data_dir, f"d12a-{language}", records[language])
        counts[f"d12a-{language}"] = len(records[language])
    return counts


def prepare(args: argparse.Namespace) -> dict[str, int]:
    data_dir = Path(args.data_dir).expanduser().resolve()
    selected = _selected_files()
    subset_args = args.subsets or list(SUBSETS)
    subsets = list(dict.fromkeys(part.strip() for value in subset_args for part in value.split(",") if part.strip()))
    unknown = set(subsets) - set(SUBSETS)
    if unknown:
        raise ValueError(f"Unknown subsets: {sorted(unknown)}")
    counts = {}
    if "ch109" in subsets:
        counts["ch109"] = len(
            _materialize_ctm_subset(
                subset="ch109",
                audio_root=Path(args.chaes_audio_root),
                ctm_root=Path(args.chaes_ctm_root),
                selected_names=selected["ch109"],
                data_dir=data_dir,
                audio_prefix=Path(args.audio_prefix),
                gt_prefix=Path(args.gt_prefix),
            )
        )
    if "ami-ihm-test" in subsets:
        counts["ami-ihm-test"] = len(
            _materialize_ctm_subset(
                subset="ami-ihm-test",
                audio_root=Path(args.ami_audio_root),
                ctm_root=Path(args.ami_ctm_root),
                selected_names=selected["ami-ihm-test"],
                data_dir=data_dir,
                audio_prefix=Path(args.audio_prefix),
                gt_prefix=Path(args.gt_prefix),
            )
        )
    if "lsm" in subsets:
        counts["lsm"] = len(
            _materialize_lsm(
                textgrid_root=Path(args.librispeech_textgrid_root),
                skills_root=Path(args.lsm_skills_root),
                metadata_root=Path(args.lsm_metadata_root),
                selected=selected["lsm"],
                data_dir=data_dir,
                audio_prefix=Path(args.audio_prefix),
                gt_prefix=Path(args.gt_prefix),
            )
        )
    d12_languages = [subset.removeprefix("d12a-") for subset in subsets if subset.startswith("d12a-")]
    if d12_languages:
        counts.update(
            _materialize_d12a(
                languages=d12_languages,
                aligned_manifest=Path(args.d12_aligned_manifest),
                label_root=Path(args.d12_label_root),
                segment_audio_root=Path(args.d12_segment_audio_root) if args.d12_segment_audio_root else None,
                selected=selected["d12a"],
                data_dir=data_dir,
                audio_prefix=Path(args.audio_prefix),
                gt_prefix=Path(args.gt_prefix),
            )
        )
    (data_dir / "selected_files.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--subsets", action="append", help="Comma-separated subset names; default: all")
    parser.add_argument("--audio-prefix", default=str(DEFAULT_AUDIO_PREFIX))
    parser.add_argument("--gt-prefix", default=str(DEFAULT_GT_PREFIX))
    parser.add_argument("--chaes-audio-root")
    parser.add_argument("--chaes-ctm-root")
    parser.add_argument("--ami-audio-root")
    parser.add_argument("--ami-ctm-root")
    parser.add_argument("--librispeech-textgrid-root")
    parser.add_argument("--lsm-skills-root")
    parser.add_argument("--lsm-metadata-root")
    parser.add_argument("--d12-aligned-manifest")
    parser.add_argument("--d12-label-root")
    parser.add_argument("--d12-segment-audio-root")
    args = parser.parse_args()
    required = {
        "ch109": ("chaes_audio_root", "chaes_ctm_root"),
        "ami-ihm-test": ("ami_audio_root", "ami_ctm_root"),
        "lsm": ("librispeech_textgrid_root", "lsm_skills_root", "lsm_metadata_root"),
    }
    selected_subsets = {
        part.strip() for value in (args.subsets or list(SUBSETS)) for part in value.split(",") if part.strip()
    }
    missing = [
        option for subset in selected_subsets for option in required.get(subset, ()) if not getattr(args, option)
    ]
    if selected_subsets & set(D12_SUBSETS):
        missing.extend(option for option in ("d12_aligned_manifest", "d12_label_root") if not getattr(args, option))
    if missing:
        parser.error("missing source options: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing))
    return args


if __name__ == "__main__":
    print(json.dumps(prepare(parse_args()), sort_keys=True))
