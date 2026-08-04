# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Prepare multi-speaker SOT ASR manifests for NeMo Skills."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable

BENCHMARK_NAME = "multispeaker-sot"
DEFAULT_IAD_MANIFEST_ROOT = (
    "/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/"
    "data/multi_speaker_asr/eval_manifests"
)
DEFAULT_ORD_MANIFEST_ROOT = (
    "/lustre/fs11/portfolios/llmservice/projects/llmservice_nemo_speechlm/"
    "data/multi_speaker_asr/eval_manifests"
)
DEFAULT_PROMPT = "Transcribe the mixed audio with speaker tags like <spk:0> hello <spk:1> yes."
SPEAKER_TAG_STYLES = ("spk", "s_bracket")
SOT_TAG_RE = re.compile(r"\[(s\d+)\]|<spk:(\d+)>", re.IGNORECASE)
SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")
PROJECT_DATA_ROOTS = (
    Path("/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data"),
    Path("/lustre/fs12/portfolios/llmservice/projects/llmservice_nemo_speechlm/data"),
)
CONTAINER_DATA_ROOT = Path("/data")


class InvalidAudioSegmentError(ValueError):
    """Raised when a manifest row cannot produce a non-empty audio segment."""

BENCHMARK_MANIFESTS = {
    # under20s (10-20s segments)
    "under20s-ami-ihm-dev": "under20s_eval_set/ami_ihm_dev_10-20s.json",
    "under20s-ami-ihm-test": "under20s_eval_set/ami_test_10-20s.json",
    "under20s-ami-sdm-test": "under20s_eval_set/ami_sdm_test_10-20s.json",
    "under20s-chaes-ch109-test": "under20s_eval_set/ch109_test_10-20s.json",
    "under20s-libri-am-dev": "under20s_eval_set/libri-am_dev_10-20s.json",
    "under20s-mixer6-ch04-test": "under20s_eval_set/mixer6_test_CH04mic_10-20s.json",
    # 60-90s segments
    "60-90s-ami-ihm-test": "60s_90s_eval_set/ami_ihm_test_60-90s.json",
    "60-90s-ami-sdm-test": "60s_90s_eval_set/ami_sdm_test_60-90s.json",
    "60-90s-chaes-ch109-test": "60s_90s_eval_set/ch109_test_60-90s.json",
    "60-90s-mixer6-ch04-test": "60s_90s_eval_set/mixer6_ch4_test_60-90s.json",
    # 120-180s segments
    "120-180s-ami-ihm-test": "120s_180s_eval_set/ami_ihm_test_120-180s.json",
    "120-180s-ami-sdm-test": "120s_180s_eval_set/ami_sdm_test_120-180s.json",
    "120-180s-chaes-ch109-test": "120s_180s_eval_set/ch109_test_120-180s.json",
    "120-180s-mixer6-ch04-test": "120s_180s_eval_set/mixer6_ch4_test_120-180s.json",
    # full-length sessions
    "full-ami-ihm-test": "full_length_eval_set/ami_ihm_test_full.json",
    "full-ami-sdm-test": "full_length_eval_set/ami_sdm_test_full.json",
    "full-chaes-ch109-test": "full_length_eval_set/ch109_test_full.json",
    "full-mixer6-ch04-test": "full_length_eval_set/mixer6_ch4_test_full.json",
    # LibriSpeechMix (clean test, 1/2/3 speakers)
    "lsm-test-clean-1mix": "libri_speech_mix/lsm-test-clean-1mix_nemo_sot.json",
    "lsm-test-clean-2mix": "libri_speech_mix/lsm-test-clean-2mix_nemo_sot.json",
    "lsm-test-clean-3mix": "libri_speech_mix/lsm-test-clean-3mix_nemo_sot.json",
}

# All current eval sets embed the SOT reference directly in each manifest's "text"
# field, so no external SegLST companion files are needed. Benchmarks may still opt
# in per-row via "seglst_filepath"/"reference_seglst_filepath".
REFERENCE_SEGLST_FILENAMES: dict[str, str] = {}


def _default_data_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent.parent / f"{parent.name}-data" / BENCHMARK_NAME
    return Path.home() / ".cache" / "nemo-skills-data" / BENCHMARK_NAME


def _resolve_manifest_root(cluster: str | None, manifest_root: str | None) -> Path:
    if manifest_root:
        return Path(manifest_root).expanduser().resolve()
    env_root = os.getenv("MULTISPEAKER_SOT_MANIFEST_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    if cluster == "ord":
        return Path(DEFAULT_ORD_MANIFEST_ROOT)
    return Path(DEFAULT_IAD_MANIFEST_ROOT)


def _resolve_values(values: list[str] | None, allowed: Iterable[str]) -> list[str]:
    allowed_set = set(allowed)
    if values is None:
        return list(allowed)
    out = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if part == "all":
                out.extend(allowed)
            elif part not in allowed_set:
                raise ValueError(f"Unsupported benchmark set {part!r}; expected one of {sorted(allowed_set)}")
            else:
                out.append(part)
    return list(dict.fromkeys(out))


def _safe_id(value: str) -> str:
    return SAFE_ID_RE.sub("_", value).strip("_") or "sample"


def _path_from_row(row: dict, *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value and value != "-":
            return value
    return None


def _remap_project_data_path(path: Path) -> Path:
    for root in PROJECT_DATA_ROOTS:
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        candidate = CONTAINER_DATA_ROOT / rel
        if candidate.exists():
            return candidate
    if path.exists():
        return path
    return path


def _resolve_row_path(path: str | None, manifest_path: Path) -> Path | None:
    if not path:
        return None
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return _remap_project_data_path(resolved)
    candidate = (manifest_path.parent / resolved).resolve()
    if candidate.exists():
        return candidate
    fallback = (manifest_path.parents[1] / resolved).resolve() if len(manifest_path.parents) > 1 else candidate
    return _remap_project_data_path(fallback)


def iter_jsonl(path: Path):
    with path.open("rt", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_seglst(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("segments", "annotations", "entries"):
            if isinstance(data.get(key), list):
                return data[key]
    if not isinstance(data, list):
        raise ValueError(f"Unsupported SegLST format in {path}")
    return data


def _row_session_id(row: dict) -> str | None:
    audio_path = _path_from_row(row, "audio_filepath", "audio_path", "audio")
    if not audio_path:
        return None
    stem = Path(audio_path).stem
    return stem.split(".")[0] if "." in stem else stem


def _filter_seglst_entries_for_row(entries: list[dict], row: dict) -> list[dict]:
    session_id = _row_session_id(row)
    if session_id is None:
        return entries
    filtered = [entry for entry in entries if str(entry.get("session_id")) == session_id]
    if filtered:
        return filtered
    session_ids = {str(entry.get("session_id")) for entry in entries}
    return entries if len(session_ids) <= 1 else []


def _reference_seglst_path(benchmark: str, row: dict, manifest_path: Path) -> Path | None:
    row_path = _resolve_row_path(
        _path_from_row(row, "reference_seglst_filepath", "seglst_filepath", "seglst"), manifest_path
    )
    if row_path is not None:
        return row_path
    filename = REFERENCE_SEGLST_FILENAMES.get(benchmark)
    if filename is None:
        return None
    return _remap_project_data_path(manifest_path.parent / filename)


def format_speaker_tag(tag: str, speaker_tag_style: str = "spk") -> str:
    if speaker_tag_style == "s_bracket":
        return f"[{tag}]"
    if speaker_tag_style == "spk":
        return f"<spk:{int(tag[1:])}>"
    raise ValueError(f"Unsupported speaker tag style: {speaker_tag_style}")


def normalize_sot_text(text: str, speaker_tag_style: str = "spk") -> str:
    def repl(match: re.Match) -> str:
        bracket_tag, spk_tag = match.groups()
        tag = bracket_tag.lower() if bracket_tag is not None else f"s{int(spk_tag)}"
        return format_speaker_tag(tag, speaker_tag_style)

    return re.sub(r"\s+", " ", SOT_TAG_RE.sub(repl, text or "")).strip()


def _speaker_tag_map_from_sot(text: str) -> dict[str, str]:
    mapping = {}
    for match in SOT_TAG_RE.finditer(text or ""):
        bracket_tag, spk_tag = match.groups()
        tag = bracket_tag.lower() if bracket_tag is not None else f"s{int(spk_tag)}"
        mapping.setdefault(tag, tag)
    return mapping


def _sot_speaker_sequence(text: str) -> list[str]:
    seq = []
    for match in SOT_TAG_RE.finditer(text or ""):
        bracket_tag, spk_tag = match.groups()
        seq.append(bracket_tag.lower() if bracket_tag is not None else f"s{int(spk_tag)}")
    return seq


def _num_speaker_changes_from_sequence(sequence: list[str]) -> int:
    return sum(1 for prev, cur in zip(sequence, sequence[1:]) if prev != cur)


def seglst_to_sot(entries: list[dict], speaker_tag_style: str = "spk") -> tuple[str, dict[str, str], int]:
    def start_time(entry: dict) -> float:
        return float(entry.get("start_time", entry.get("start", 0.0)) or 0.0)

    ordered = sorted(entries, key=lambda item: (str(item.get("session_id", "")), start_time(item)))
    speaker_to_tag: dict[str, str] = {}
    sequence: list[str] = []
    chunks: list[str] = []
    for entry in ordered:
        speaker = str(entry.get("speaker", "speaker_0"))
        words = str(entry.get("words", entry.get("text", ""))).strip()
        if not words:
            continue
        if speaker not in speaker_to_tag:
            speaker_to_tag[speaker] = f"s{len(speaker_to_tag)}"
        tag = speaker_to_tag[speaker]
        sequence.append(tag)
        chunks.append(f"{format_speaker_tag(tag, speaker_tag_style)} {words}")
    return " ".join(chunks), speaker_to_tag, _num_speaker_changes_from_sequence(sequence)


def _is_placeholder_sample_id(value: str) -> bool:
    return value.strip().lower() in {"infer"}


def _sample_id(row: dict, benchmark: str, index: int, source_audio: Path | None = None) -> str:
    for key in ("sample_id", "id", "utt", "uniq_id"):
        value = row.get(key)
        if isinstance(value, str) and value and value != "-":
            return _safe_id(value)

    label = row.get("label")
    if isinstance(label, str) and label and label != "-" and not _is_placeholder_sample_id(label):
        return _safe_id(label)

    audio_path = str(source_audio) if source_audio is not None else _path_from_row(row, "audio_filepath", "audio_path")
    stem = Path(audio_path).stem if audio_path else benchmark
    offset = float(row.get("offset", 0.0) or 0.0)
    duration = float(row.get("duration", 0.0) or 0.0)
    return _safe_id(f"{stem}_{offset:.2f}_{duration:.2f}_{index}")


def _public_audio_path(local_audio_path: Path, local_audio_root: Path, audio_prefix: Path) -> str:
    return str(audio_prefix / local_audio_path.relative_to(local_audio_root))


def _try_materialize_pcm_wav_segment(
    source_path: Path, output_path: Path, offset: float, duration: float | None
) -> bool:
    if source_path.suffix.lower() != ".wav":
        return False

    try:
        import wave

        import numpy as np
        import soundfile as sf

        with wave.open(str(source_path), "rb") as fin:
            if fin.getcomptype() != "NONE":
                return False
            sample_rate = fin.getframerate()
            channels = fin.getnchannels()
            sample_width = fin.getsampwidth()
            start_frame = max(0, int(round(offset * sample_rate)))
            total_frames = fin.getnframes()
            if start_frame >= total_frames:
                return False
            frame_count = (
                total_frames - start_frame
                if duration is None or duration <= 0
                else max(1, int(round(duration * sample_rate)))
            )
            if duration is not None and duration > 0 and start_frame + frame_count > total_frames:
                frame_count = total_frames - start_frame
            fin.setpos(start_frame)
            raw = fin.readframes(frame_count)

        if not raw:
            return False
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if channels == 1:
            with wave.open(str(output_path), "wb") as fout:
                fout.setnchannels(channels)
                fout.setsampwidth(sample_width)
                fout.setframerate(sample_rate)
                fout.writeframes(raw)
            return True
        if sample_width == 1:
            audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            audio = (audio - 128.0) / 128.0
        elif sample_width == 2:
            audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            return False
        audio = audio.reshape(-1, channels).mean(axis=1)
        sf.write(str(output_path), audio, sample_rate)
        return True
    except (OSError, EOFError, ValueError, wave.Error):
        return False


def _has_valid_audio_frames(path: Path, min_duration: float | None = None) -> bool:
    if not path.exists():
        return False
    try:
        import wave

        with wave.open(str(path), "rb") as fin:
            frames = fin.getnframes()
            sample_rate = fin.getframerate()
    except (OSError, EOFError, ValueError, wave.Error):
        try:
            import soundfile as sf

            info = sf.info(str(path))
            frames = info.frames
            sample_rate = info.samplerate
        except (OSError, RuntimeError, ValueError):
            return False
    if frames <= 0:
        return False
    if min_duration is not None and min_duration > 0 and sample_rate > 0:
        return frames / sample_rate + 0.05 >= min_duration
    return True


def _audio_duration_seconds(path: Path) -> float | None:
    """Return the audio length in seconds, or None when it cannot be read."""
    if not path.exists():
        return None
    try:
        import wave

        with wave.open(str(path), "rb") as fin:
            frames = fin.getnframes()
            sample_rate = fin.getframerate()
    except (OSError, EOFError, ValueError, wave.Error):
        try:
            import soundfile as sf

            info = sf.info(str(path))
            frames = info.frames
            sample_rate = info.samplerate
        except (OSError, RuntimeError, ValueError):
            return None
    if frames <= 0 or sample_rate <= 0:
        return None
    return frames / sample_rate


def materialize_audio_segment(source_path: Path, output_path: Path, offset: float, duration: float | None) -> bool:
    if _has_valid_audio_frames(output_path, duration):
        return True
    if output_path.exists():
        output_path.unlink()
    if _try_materialize_pcm_wav_segment(source_path, output_path, offset, duration):
        return _has_valid_audio_frames(output_path, duration)

    import numpy as np
    import soundfile as sf

    info = sf.info(str(source_path))
    start = max(0, int(round(offset * info.samplerate)))
    frames = -1 if duration is None or duration <= 0 else max(1, int(round(duration * info.samplerate)))
    if frames > 0 and start + frames > info.frames:
        frames = info.frames - start
        if frames <= 0:
            return False
    audio, sample_rate = sf.read(str(source_path), start=start, frames=frames, dtype="float32")
    if getattr(audio, "size", len(audio)) == 0:
        return False
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sample_rate)
    return _has_valid_audio_frames(output_path, duration)


def _write_reference_seglst(entries: list[dict], dst_path: Path, audio_prefix: Path, data_dir: Path) -> str:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if not dst_path.exists():
        dst_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    rel = dst_path.relative_to(data_dir)
    # audio_prefix is normally /data/<benchmark>/audio; its parent is /data/<benchmark>.
    public_root = audio_prefix.parent
    return str(public_root / rel)


def build_record(
    row: dict,
    *,
    benchmark: str,
    index: int,
    manifest_path: Path,
    data_dir: Path,
    audio_prefix: Path,
    no_audio: bool,
    speaker_tag_style: str,
) -> dict:
    source_audio = _resolve_row_path(_path_from_row(row, "audio_filepath", "audio_path", "audio"), manifest_path)
    if source_audio is None:
        raise ValueError(f"Missing audio path in {manifest_path} row {index}")

    offset = float(row.get("offset", 0.0) or 0.0)
    duration_value = row.get("duration")
    duration = float(duration_value) if duration_value not in (None, "") else None
    sample_id = _sample_id(row, benchmark, index, source_audio)
    local_audio_root = data_dir / "audio"
    local_audio_path = local_audio_root / benchmark / f"{sample_id}.wav"
    if not no_audio and not materialize_audio_segment(source_audio, local_audio_path, offset, duration):
        raise InvalidAudioSegmentError(
            f"Could not materialize non-empty audio for {manifest_path} row {index}: "
            f"source={source_audio} offset={offset} duration={duration}"
        )
    audio_filepath = _public_audio_path(local_audio_path, local_audio_root, audio_prefix)

    # Full-session rows routinely omit "duration" (transcribe the whole file). A 0
    # here would mis-size duration-based chunk balancing and the generation context,
    # so fall back to the real audio length.
    if duration is None:
        if not no_audio and local_audio_path.exists():
            duration = _audio_duration_seconds(local_audio_path)
        else:
            source_seconds = _audio_duration_seconds(source_audio)
            duration = None if source_seconds is None else max(0.0, source_seconds - offset)
    duration_seconds = round(float(duration or 0.0), 4)

    seglst_src = _reference_seglst_path(benchmark, row, manifest_path)
    text = str(row.get("text", "") or "").strip()
    speaker_tag_map = _speaker_tag_map_from_sot(text)
    num_changes = row.get("num_changes")
    reference_seglst_filepath = None
    if seglst_src is not None and seglst_src.exists():
        seglst_entries = _filter_seglst_entries_for_row(load_seglst(seglst_src), row)
        if not seglst_entries:
            raise ValueError(f"Could not find SegLST entries for {manifest_path} row {index}")
        seglst_dst = data_dir / "seglst" / benchmark / f"{sample_id}_ref.seglst.json"
        reference_seglst_filepath = _write_reference_seglst(seglst_entries, seglst_dst, audio_prefix, data_dir)
        if not text or text == "-":
            text, speaker_tag_map, num_changes = seglst_to_sot(seglst_entries, speaker_tag_style)
    else:
        text = normalize_sot_text(text, speaker_tag_style)
        speaker_tag_map = _speaker_tag_map_from_sot(text)

    if not text or text == "-":
        raise ValueError(f"Could not resolve SOT reference text for {manifest_path} row {index}")
    if num_changes is None:
        num_changes = _num_speaker_changes_from_sequence(_sot_speaker_sequence(text))

    record = {
        "audio_filepath": audio_filepath,
        "source_audio_filepath": str(source_audio),
        "offset": offset,
        "duration": duration_seconds,
        "text": text,
        "expected_answer": text,
        "source_lang": row.get("source_lang", "en"),
        "target_lang": row.get("target_lang", "en"),
        "taskname": row.get("taskname", "asr"),
        "pnc": row.get("pnc", "no"),
        "num_speakers": int(row.get("num_speakers") or len(speaker_tag_map) or 1),
        "num_changes": int(num_changes or 0),
        "dataset_id": row.get("dataset_id", f"{BENCHMARK_NAME}:{benchmark}"),
        "sample_id": sample_id,
        "speaker_tag_map": speaker_tag_map,
        "speaker_tag_style": speaker_tag_style,
        "subset_for_metrics": benchmark,
        "task_type": "MS-ASR",
        "category": "nonjudge",
        "messages": [
            {
                "role": "user",
                "content": DEFAULT_PROMPT,
                "audio": {"path": audio_filepath, "duration": duration_seconds},
            }
        ],
    }
    for key in ("rttm_filepath", "uem_filepath", "ctm_filepath", "label", "global_speaker_ids"):
        if key in row:
            record[key] = row[key]
    if reference_seglst_filepath:
        record["reference_seglst_filepath"] = reference_seglst_filepath
    return record


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("wt", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row) + "\n")
            count += 1
    return count


def prepare_manifest_set(
    benchmark: str,
    *,
    manifest_root: Path,
    output_dir: Path,
    data_dir: Path,
    audio_prefix: Path,
    no_audio: bool,
    max_samples: int | None,
    speaker_tag_style: str,
    invalid_audio_policy: str = "skip",
) -> int:
    manifest_rel = BENCHMARK_MANIFESTS[benchmark]
    manifest_path = manifest_root / manifest_rel
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest for {benchmark}: {manifest_path}")
    rows = []
    for index, row in enumerate(iter_jsonl(manifest_path)):
        if max_samples is not None and len(rows) >= max_samples:
            break
        if index % 25 == 0:
            print(f"{benchmark}: row {index}", flush=True)
        try:
            record = build_record(
                row,
                benchmark=benchmark,
                index=index,
                manifest_path=manifest_path,
                data_dir=data_dir,
                audio_prefix=audio_prefix,
                no_audio=no_audio,
                speaker_tag_style=speaker_tag_style,
            )
        except InvalidAudioSegmentError as exc:
            if invalid_audio_policy != "skip":
                raise
            print(f"{benchmark}: skipped row {index}: {exc}", flush=True)
            continue
        rows.append(record)
        if (index + 1) % 25 == 0:
            print(f"{benchmark}: completed {index + 1} rows", flush=True)
    return write_jsonl(output_dir / benchmark / "test.jsonl", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare multi-speaker SOT benchmark manifests.")
    parser.add_argument("--data_dir", type=str, default=None, help="External data/artifact root.")
    parser.add_argument(
        "--dataset-root",
        dest="data_dir",
        type=str,
        default=None,
        help="Alias for --data_dir that does not conflict with ns prepare_data's top-level --data_dir.",
    )
    parser.add_argument(
        "--manifest-root",
        type=str,
        default=None,
        help="Root containing the eval sets (under20s/60s_90s/120s_180s/full_length/libri_speech_mix).",
    )
    parser.add_argument("--cluster", choices=("iad", "ord"), default="iad", help="Default manifest-root selector.")
    parser.add_argument("--audio-prefix", type=str, default=None, help="Audio path prefix embedded in manifests.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where prepared <set>/test.jsonl manifests are written. "
        "Defaults to this dataset package directory to preserve ns prepare_data behavior.",
    )
    parser.add_argument("--no-audio", action="store_true", help="Do not materialize clipped audio.")
    parser.add_argument("--sets", nargs="+", default=["all"], help="Benchmark sets to prepare or 'all'.")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum records per benchmark set.")
    parser.add_argument(
        "--speaker-tag-style",
        choices=SPEAKER_TAG_STYLES,
        default="spk",
        help="Speaker tag style to emit in SOT references and prompts.",
    )
    parser.add_argument(
        "--invalid-audio-policy",
        choices=("skip", "error"),
        default="skip",
        help="How to handle rows whose source audio cannot produce a non-empty segment.",
    )
    args = parser.parse_args()

    manifest_root = _resolve_manifest_root(args.cluster, args.manifest_root)
    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else _default_data_dir()
    audio_prefix = Path(args.audio_prefix).expanduser() if args.audio_prefix else Path(f"/data/{BENCHMARK_NAME}/audio")
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path(__file__).parent
    selected_sets = _resolve_values(args.sets, BENCHMARK_MANIFESTS.keys())
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Manifest root: {manifest_root}")
    print(f"Data/artifacts root: {data_dir}")
    all_counts = {}
    for benchmark in selected_sets:
        print(f"Preparing {benchmark}...", flush=True)
        count = prepare_manifest_set(
            benchmark,
            manifest_root=manifest_root,
            output_dir=output_dir,
            data_dir=data_dir,
            audio_prefix=audio_prefix,
            no_audio=args.no_audio,
            max_samples=args.max_samples,
            speaker_tag_style=args.speaker_tag_style,
            invalid_audio_policy=args.invalid_audio_policy,
        )
        all_counts[benchmark] = count
        print(f"Prepared {benchmark}: {count}", flush=True)

    print("Prepared multi-speaker SOT manifests:")
    for name, count in sorted(all_counts.items()):
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
