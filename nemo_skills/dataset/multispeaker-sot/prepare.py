# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Prepare multi-speaker SOT ASR manifests for NeMo Skills."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

BENCHMARK_NAME = "multispeaker-sot"
TURING_BENCHMARK = "turing-diarization-v1"
MLC_SLM_2_BENCHMARK = "mlc-slm-2-dev-13L"
NOTSOFAR1_BENCHMARKS = ("notsofar1-eval-ihm", "notsofar1-eval-sc")
DFW_MULTISPEAKER_ROOT = (
    "/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr/"
    "data/multi_speaker_asr"
)
DEFAULT_MLC_SLM_2_OPUS_DEV_ROOT = (
    f"{DFW_MULTISPEAKER_ROOT}/MultiSpeakerASR/MLC_SLM_2_opus/dev"
)
DEFAULT_NOTSOFAR1_ROOT = (
    f"{DFW_MULTISPEAKER_ROOT}/diarization_e2e/above4spk_finetune_datasets/notsofar1"
)
DEFAULT_TURING_ROOT = (
    f"{DFW_MULTISPEAKER_ROOT}/diarization_e2e/above4spk_finetune_datasets/"
    "Turing_Batch_Deliveries"
)
MLC_SLM_2_DEV_LOCALES = (
    "de",
    "en",
    "es",
    "es-MX",
    "fr",
    "fr-CA",
    "it",
    "ja",
    "ko",
    "pt",
    "pt-BR",
    "ru",
    "th",
)
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


def available_benchmarks() -> tuple[str, ...]:
    """Return every leaf benchmark supported by this preparation package."""
    return (*BENCHMARK_MANIFESTS, TURING_BENCHMARK, MLC_SLM_2_BENCHMARK, *NOTSOFAR1_BENCHMARKS)


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


def _load_turing_session_entries(session_dir: Path) -> list[dict]:
    """Load every per-speaker SegLST entry for one Turing conversation."""
    entries: list[dict] = []
    seglst_paths = sorted(session_dir.glob("SPK*.seglst.json"))
    if not seglst_paths:
        raise FileNotFoundError(f"No speaker SegLST files found in {session_dir}")
    for seglst_path in seglst_paths:
        entries.extend(load_seglst(seglst_path))
    return entries


def _write_json(path: Path, value: object) -> None:
    """Write human-readable UTF-8 JSON atomically, without ``\\u`` escapes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == payload:
        return
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(path)


def _valid_stg_entries(entries: list[dict]) -> list[dict]:
    """Drop rows that cannot form an STG/Lhotse supervision and RTTM segment."""
    valid = []
    for entry in entries:
        start = float(entry.get("start_time", entry.get("start", 0.0)))
        end = float(entry.get("end_time", entry.get("end", 0.0)))
        speaker = str(entry.get("speaker", "speaker_0"))
        if speaker.lower() != "none" and round(end - start, 3) > 0:
            valid.append(entry)
    return valid


def _write_rttm_from_seglst(entries: list[dict], path: Path, recording_id: str) -> None:
    """Create an RTTM that exactly matches the combined SegLST supervision set."""
    lines = []
    for entry in sorted(entries, key=lambda item: float(item.get("start_time", item.get("start", 0.0)))):
        start = float(entry.get("start_time", entry.get("start", 0.0)))
        end = float(entry.get("end_time", entry.get("end", 0.0)))
        if round(end - start, 3) <= 0:
            continue
        speaker = str(entry.get("speaker", "speaker_0"))
        lines.append(
            f"SPEAKER {recording_id} 1 {start:.3f} {end - start:.3f} "
            f"<NA> <NA> {speaker} <NA> <NA>"
        )
    payload = "\n".join(lines) + ("\n" if lines else "")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(path)


def _run_stg_v4_2(
    input_manifest: Path,
    output_manifest: Path,
    *,
    dataset_id: str,
    max_speakers: int,
    speaker_tag_style: str,
    stg_script: Path | None,
    python_executable: str,
) -> list[dict]:
    """Run the canonical STG 4.2 full-session NeMoSOT generation contract."""
    if stg_script is None:
        raise ValueError(
            "STG 4.2 script is required; pass --stg-v4-2-script or set "
            "NEMO_SOT_STG_V4_2_SCRIPT"
        )
    if not stg_script.is_file():
        raise FileNotFoundError(f"Missing STG 4.2 script: {stg_script}")
    token_style = "<spk:*>" if speaker_tag_style == "spk" else "[s*]"
    command = [
        python_executable,
        str(stg_script),
        "--dataset_id",
        dataset_id,
        "--input_mainfest_filepath",
        str(input_manifest),
        "--output_mainfest_filepath",
        str(output_manifest),
        "--length",
        "0",
        "100000",
        "--min_gap",
        "100000",
        "--seg_hop",
        "0",
        "--word_merge_thres",
        "-1",
        "--edge_ovl_thres",
        "0",
        "--leading_word_thres",
        "0",
        "--max_speakers",
        str(max_speakers),
        "--min_overlap_turn_sec",
        "1.2",
        "--use_word_ts_sim",
        "--speaker_token_style",
        token_style,
        "--speaker_token_position",
        "segment",
        "--full_session",
        "--allow_simultaneous_start",
        "--double_check",
        "--mp",
    ]
    subprocess.run(command, check=True)
    return list(iter_jsonl(output_manifest))


def _speaker_tag_map_from_stg(row: dict) -> dict[str, str]:
    speakers = [str(speaker) for speaker in row.get("global_speaker_ids", [])]
    return {speaker: f"s{index}" for index, speaker in enumerate(speakers)}


def _convert_turing_nsv_text(text: str) -> str:
    converter = importlib.import_module(
        "nemo_skills.dataset.multispeaker-sot.turing-diarization-v1.convert_nsv_tokens"
    )
    return converter.convert_text(text)


def _copy_file_atomic(source: Path, destination: Path) -> None:
    """Copy a file without exposing a partial destination to evaluation jobs."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == source.stat().st_size:
        # Size equality is not content equality. This matters for tiny label files:
        # an adjusted RTTM can have the same length as a stale source RTTM.
        if source.stat().st_size > 16 * 1024 * 1024 or source.read_bytes() == destination.read_bytes():
            return
    tmp_path = destination.with_suffix(f"{destination.suffix}.tmp")
    shutil.copy2(source, tmp_path)
    tmp_path.replace(destination)


def prepare_turing_diarization_v1(
    *,
    source_root: Path,
    output_dir: Path,
    data_dir: Path,
    audio_prefix: Path,
    no_audio: bool,
    max_samples: int | None,
    speaker_tag_style: str,
    stg_script: Path | None = None,
    python_executable: str = sys.executable,
) -> int:
    """Prepare full-conversation Turing diarization data from delivered OPUS mixdowns.

    Only direct ``Delivery_Batch*/NV-*`` session directories are considered, which
    deliberately excludes delivery ``Scratch`` folders. Sessions without a delivered
    ``*_mixed.opus`` are omitted rather than constructing a new mix from speaker tracks.
    """
    if not source_root.is_dir():
        raise FileNotFoundError(f"Missing Turing delivery root: {source_root}")

    session_dirs = sorted(source_root.glob("Delivery_Batch*/NV-*"))
    source_records: list[dict] = []
    skipped_without_mixdown: list[str] = []
    local_audio_root = data_dir / "audio"
    public_root = audio_prefix.parent

    for session_dir in session_dirs:
        if max_samples is not None and len(source_records) >= max_samples:
            break
        mixed_opus = sorted(session_dir.glob("*_mixed.opus"))
        if not mixed_opus:
            skipped_without_mixdown.append(session_dir.name)
            continue
        if len(mixed_opus) != 1:
            raise ValueError(f"Expected one OPUS mixdown in {session_dir}, found {len(mixed_opus)}")

        source_audio = mixed_opus[0]
        duration = _audio_duration_seconds(source_audio)
        if duration is None or duration <= 0:
            raise InvalidAudioSegmentError(f"Could not read non-empty OPUS mixdown: {source_audio}")

        entries = _valid_stg_entries(_load_turing_session_entries(session_dir))
        if not any(str(entry.get("words", entry.get("text", ""))).strip() for entry in entries):
            raise ValueError(f"No reference transcript text resolved for {session_dir}")

        session_id = session_dir.name
        language_code = session_id.split("-")[1].lower()
        language_code = "el" if language_code == "gr" else language_code
        local_audio_path = local_audio_root / TURING_BENCHMARK / f"{session_id}.opus"
        if not no_audio:
            _copy_file_atomic(source_audio, local_audio_path)
        audio_filepath = _public_audio_path(local_audio_path, local_audio_root, audio_prefix)

        seglst_dst = data_dir / "seglst" / TURING_BENCHMARK / f"{session_id}_ref.seglst.json"
        _write_json(seglst_dst, entries)
        reference_seglst_filepath = str(public_root / seglst_dst.relative_to(data_dir))
        rttm_dst = data_dir / "rttm" / TURING_BENCHMARK / f"{session_id}.rttm"
        _write_rttm_from_seglst(entries, rttm_dst, session_id)
        source_records.append(
            {
                "stg_input": {
                    "audio_filepath": str(local_audio_path),
                    "rttm_filepath": str(rttm_dst),
                    "ctm_filepath": None,
                    "seglst_filepath": str(seglst_dst),
                    "source_lang": language_code,
                    "target_lang": language_code,
                    "taskname": "asr",
                    "pnc": "yes",
                    "uniq_id": session_id,
                },
                "audio_filepath": audio_filepath,
                "source_audio_filepath": str(source_audio),
                "sample_id": session_id,
                "duration": round(duration, 4),
                "reference_seglst_filepath": reference_seglst_filepath,
                "rttm_filepath": str(public_root / rttm_dst.relative_to(data_dir)),
            }
        )
        if len(source_records) % 25 == 0:
            print(f"{TURING_BENCHMARK}: staged {len(source_records)} sessions", flush=True)

    if skipped_without_mixdown:
        print(
            f"{TURING_BENCHMARK}: skipped {len(skipped_without_mixdown)} sessions without delivered OPUS mixdown: "
            + ", ".join(skipped_without_mixdown),
            flush=True,
        )
    with tempfile.TemporaryDirectory(prefix=f"{TURING_BENCHMARK}-stg-") as temp_dir:
        stg_input = Path(temp_dir) / "input.jsonl"
        stg_output = Path(temp_dir) / "output.jsonl"
        write_jsonl(stg_input, (record["stg_input"] for record in source_records))
        stg_rows = _run_stg_v4_2(
            stg_input,
            stg_output,
            dataset_id=TURING_BENCHMARK,
            max_speakers=8,
            speaker_tag_style=speaker_tag_style,
            stg_script=stg_script,
            python_executable=python_executable,
        )

    stg_by_audio = {str(row["audio_filepath"]): row for row in stg_rows}
    rows: list[dict] = []
    for source_record in source_records:
        stg_row = stg_by_audio.get(source_record["stg_input"]["audio_filepath"])
        if stg_row is None:
            raise ValueError(f"STG 4.2 did not emit {source_record['sample_id']}")
        text = _convert_turing_nsv_text(
            normalize_sot_text(str(stg_row["text"]), speaker_tag_style)
        )
        duration_seconds = source_record["duration"]
        rows.append(
            {
                **{
                    key: value
                    for key, value in stg_row.items()
                    if key
                    not in {"audio_filepath", "rttm_filepath", "offset", "duration", "text", "dataset_id"}
                },
                "audio_filepath": source_record["audio_filepath"],
                "source_audio_filepath": source_record["source_audio_filepath"],
                "offset": 0.0,
                "duration": duration_seconds,
                "text": text,
                "expected_answer": text,
                "dataset_id": TURING_BENCHMARK,
                "sample_id": source_record["sample_id"],
                "speaker_tag_map": _speaker_tag_map_from_stg(stg_row),
                "speaker_tag_style": speaker_tag_style,
                "subset_for_metrics": TURING_BENCHMARK,
                "task_type": "MS-ASR",
                "category": "nonjudge",
                "reference_seglst_filepath": source_record["reference_seglst_filepath"],
                "rttm_filepath": source_record["rttm_filepath"],
                "messages": [
                    {
                        "role": "user",
                        "content": DEFAULT_PROMPT,
                        "audio": {
                            "path": source_record["audio_filepath"],
                            "duration": duration_seconds,
                        },
                    }
                ],
            }
        )
    return write_jsonl(output_dir / TURING_BENCHMARK / "test.jsonl", rows)


def _index_opus_by_stem(locale_audio_dir: Path) -> dict[str, Path]:
    """Index a locale's OPUS files, rejecting ambiguous duplicate stems."""
    index: dict[str, Path] = {}
    for opus_path in sorted(locale_audio_dir.rglob("*.opus")):
        if opus_path.stem in index:
            raise ValueError(
                f"Duplicate OPUS stem {opus_path.stem!r}: {index[opus_path.stem]} and {opus_path}"
            )
        index[opus_path.stem] = opus_path
    return index


def prepare_mlc_slm_2_dev_13l(
    *,
    opus_dev_root: Path,
    output_dir: Path,
    data_dir: Path,
    audio_prefix: Path,
    no_audio: bool,
    max_samples: int | None,
    speaker_tag_style: str,
    locales: Iterable[str] = MLC_SLM_2_DEV_LOCALES,
    stg_script: Path | None = None,
    python_executable: str = sys.executable,
) -> int:
    """Prepare selected MLC-SLM 2 dev locales with the canonical STG 4.2 route.

    The OPUS delivery contains time-adjusted SegLST/RTTM projections produced from
    ``nemo_labels`` by the documented silence-processing step. Use those processed
    sidecars directly so preparation depends only on the DFW OPUS delivery.
    """
    source_records: list[dict] = []
    counts: dict[str, int] = {}
    local_audio_root = data_dir / "audio"
    public_root = audio_prefix.parent

    for locale in locales:
        locale_audio_dir = opus_dev_root / "data" / locale
        if not locale_audio_dir.is_dir():
            raise FileNotFoundError(f"Missing MLC-SLM 2 OPUS directory: {locale_audio_dir}")

        seglst_dir = locale_audio_dir / "seglsts"
        rttm_dir = locale_audio_dir / "rttms"
        seglst_paths = sorted(seglst_dir.glob("*.seglst.json"))
        if not seglst_paths:
            raise FileNotFoundError(f"Missing processed MLC-SLM 2 SegLST files: {seglst_dir}")
        opus_by_stem = _index_opus_by_stem(locale_audio_dir)
        label_stems = {path.name.removesuffix(".seglst.json") for path in seglst_paths}
        missing_audio = sorted(label_stems - opus_by_stem.keys())
        extra_audio = sorted(opus_by_stem.keys() - label_stems)
        if missing_audio or extra_audio:
            raise ValueError(
                f"MLC-SLM 2 {locale}: NeMo-label/OPUS mismatch; "
                f"missing_audio={missing_audio}, extra_audio={extra_audio}"
            )

        locale_count = 0
        for seglst_src in seglst_paths:
            if max_samples is not None and len(source_records) >= max_samples:
                break
            recording_id = seglst_src.name.removesuffix(".seglst.json")
            source_audio = opus_by_stem[recording_id]
            rttm_src = rttm_dir / f"{recording_id}.rttm"
            if not rttm_src.is_file():
                raise FileNotFoundError(f"Missing processed MLC-SLM 2 RTTM: {rttm_src}")

            duration = _audio_duration_seconds(source_audio)
            if duration is None or duration <= 0:
                raise InvalidAudioSegmentError(f"Could not read non-empty OPUS audio: {source_audio}")
            entries = _valid_stg_entries(load_seglst(seglst_src))
            if not any(str(entry.get("words", entry.get("text", ""))).strip() for entry in entries):
                raise ValueError(f"No reference transcript text resolved from {seglst_src}")

            sample_id = _safe_id(f"{locale}-{recording_id}")
            local_audio_path = local_audio_root / MLC_SLM_2_BENCHMARK / locale / f"{recording_id}.opus"
            if not no_audio:
                _copy_file_atomic(source_audio, local_audio_path)
            audio_filepath = _public_audio_path(local_audio_path, local_audio_root, audio_prefix)

            seglst_dst = (
                data_dir / "seglst" / MLC_SLM_2_BENCHMARK / locale / f"{recording_id}_ref.seglst.json"
            )
            _write_json(seglst_dst, entries)
            reference_seglst_filepath = str(public_root / seglst_dst.relative_to(data_dir))
            rttm_dst = data_dir / "rttm" / MLC_SLM_2_BENCHMARK / locale / f"{recording_id}.rttm"
            _copy_file_atomic(rttm_src, rttm_dst)
            rttm_filepath = str(public_root / rttm_dst.relative_to(data_dir))

            source_lang = locale.split("-", 1)[0]
            duration_seconds = round(duration, 4)
            source_records.append(
                {
                    "stg_input": {
                        "audio_filepath": str(local_audio_path),
                        "rttm_filepath": str(rttm_dst),
                        "ctm_filepath": None,
                        "seglst_filepath": str(seglst_dst),
                        "source_lang": source_lang,
                        "target_lang": source_lang,
                        "taskname": "asr",
                        "pnc": "yes",
                        "uniq_id": recording_id,
                    },
                    "audio_filepath": audio_filepath,
                    "source_audio_filepath": str(source_audio),
                    "duration": duration_seconds,
                    "locale": locale,
                    "sample_id": sample_id,
                    "reference_seglst_filepath": reference_seglst_filepath,
                    "rttm_filepath": rttm_filepath,
                }
            )
            locale_count += 1
        counts[locale] = locale_count
        print(f"{MLC_SLM_2_BENCHMARK}: {locale}: {locale_count} recordings", flush=True)
        if max_samples is not None and len(source_records) >= max_samples:
            break

    print(
        f"{MLC_SLM_2_BENCHMARK}: staged {len(source_records)} recordings across {len(counts)} locales",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix=f"{MLC_SLM_2_BENCHMARK}-stg-") as temp_dir:
        stg_input = Path(temp_dir) / "input.jsonl"
        stg_output = Path(temp_dir) / "output.jsonl"
        write_jsonl(stg_input, (record["stg_input"] for record in source_records))
        stg_rows = _run_stg_v4_2(
            stg_input,
            stg_output,
            dataset_id=MLC_SLM_2_BENCHMARK,
            max_speakers=4,
            speaker_tag_style=speaker_tag_style,
            stg_script=stg_script,
            python_executable=python_executable,
        )

    stg_by_audio = {str(row["audio_filepath"]): row for row in stg_rows}
    rows: list[dict] = []
    for source_record in source_records:
        stg_row = stg_by_audio.get(source_record["stg_input"]["audio_filepath"])
        if stg_row is None:
            raise ValueError(f"STG 4.2 did not emit {source_record['sample_id']}")
        text = normalize_sot_text(str(stg_row["text"]), speaker_tag_style)
        duration_seconds = source_record["duration"]
        rows.append(
            {
                **{
                    key: value
                    for key, value in stg_row.items()
                    if key not in {"audio_filepath", "rttm_filepath", "offset", "duration", "text", "dataset_id"}
                },
                "audio_filepath": source_record["audio_filepath"],
                "source_audio_filepath": source_record["source_audio_filepath"],
                "offset": 0.0,
                "duration": duration_seconds,
                "text": text,
                "expected_answer": text,
                "locale": source_record["locale"],
                "dataset_id": MLC_SLM_2_BENCHMARK,
                "sample_id": source_record["sample_id"],
                "speaker_tag_map": _speaker_tag_map_from_stg(stg_row),
                "speaker_tag_style": speaker_tag_style,
                "subset_for_metrics": f"{MLC_SLM_2_BENCHMARK}.{source_record['locale']}",
                "task_type": "MS-ASR",
                "category": "nonjudge",
                "reference_seglst_filepath": source_record["reference_seglst_filepath"],
                "rttm_filepath": source_record["rttm_filepath"],
                "messages": [
                    {
                        "role": "user",
                        "content": DEFAULT_PROMPT,
                        "audio": {"path": source_record["audio_filepath"], "duration": duration_seconds},
                    }
                ],
            }
        )
    return write_jsonl(output_dir / MLC_SLM_2_BENCHMARK / "test.jsonl", rows)


def prepare_notsofar1_eval(
    benchmark: str,
    *,
    manifest_root: Path,
    output_dir: Path,
    data_dir: Path,
    audio_prefix: Path,
    no_audio: bool,
    max_samples: int | None,
    speaker_tag_style: str,
) -> int:
    """Prepare one NOTSOFAR1 eval microphone view from full-session NeMoSOT.

    The Skills full benchmark uses unpunctuated raw references. Mirror that
    contract with the Granary-v2 ``tn_raw_noPnC`` v11 manifests and retain the
    complete, untruncated eval sessions.
    """
    if benchmark not in NOTSOFAR1_BENCHMARKS:
        raise ValueError(f"Unsupported NOTSOFAR1 benchmark: {benchmark}")

    modality = benchmark.rsplit("-", 1)[-1]
    manifest_path = manifest_root / f"nemoSOT_NOTSOFAR1_{modality}_eval_full_tn_raw_noPnC.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing NOTSOFAR1 source manifest: {manifest_path}")

    rows: list[dict] = []
    local_audio_root = data_dir / "audio"
    for index, source_record in enumerate(iter_jsonl(manifest_path)):
        if max_samples is not None and len(rows) >= max_samples:
            break

        source_audio = Path(str(source_record["audio_filepath"])).expanduser()
        if not source_audio.is_file():
            raise FileNotFoundError(f"Missing NOTSOFAR1 source audio: {source_audio}")
        if float(source_record.get("offset", 0.0) or 0.0) != 0.0:
            raise ValueError(f"NOTSOFAR1 eval row {index} is not a full session: offset is nonzero")

        duration_seconds = round(float(source_record["duration"]), 4)
        if duration_seconds <= 0:
            raise InvalidAudioSegmentError(f"NOTSOFAR1 eval row {index} has invalid duration")
        text = normalize_sot_text(str(source_record.get("text", "")), speaker_tag_style)
        if not text:
            raise ValueError(f"NOTSOFAR1 eval row {index} has no SOT reference")

        sample_id = _safe_id(source_audio.stem)
        local_audio_path = local_audio_root / benchmark / source_audio.name
        if not no_audio:
            _copy_file_atomic(source_audio, local_audio_path)
        audio_filepath = _public_audio_path(local_audio_path, local_audio_root, audio_prefix)
        speaker_tag_map = _speaker_tag_map_from_sot(text)

        record = {
            "audio_filepath": audio_filepath,
            "source_audio_filepath": str(source_audio),
            "offset": 0.0,
            "duration": duration_seconds,
            "text": text,
            "expected_answer": text,
            "source_lang": source_record.get("source_lang", "en"),
            "target_lang": source_record.get("target_lang", "en"),
            "taskname": source_record.get("taskname", "asr"),
            "pnc": "no",
            "num_speakers": int(source_record.get("num_speakers") or len(speaker_tag_map)),
            "num_changes": int(source_record.get("num_changes") or 0),
            "dataset_id": source_record.get("dataset_id", "NOTSOFAR1"),
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
        if "global_speaker_ids" in source_record:
            record["global_speaker_ids"] = source_record["global_speaker_ids"]
        rows.append(record)

    return write_jsonl(output_dir / benchmark / "test.jsonl", rows)


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
    _write_json(dst_path, entries)
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
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
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
    parser.add_argument(
        "--turing-root",
        type=str,
        default=DEFAULT_TURING_ROOT,
        help="Root containing Delivery_Batch*/NV-* Turing conversation directories.",
    )
    parser.add_argument(
        "--mlc-slm-2-opus-dev-root",
        type=str,
        default=DEFAULT_MLC_SLM_2_OPUS_DEV_ROOT,
        help="DFW OPUS dev root containing data/<locale>/{opus,seglsts,rttms}.",
    )
    parser.add_argument(
        "--stg-v4-2-script",
        type=str,
        default=os.getenv("NEMO_SOT_STG_V4_2_SCRIPT"),
        help="Path to canonical serialized_transcription_generator_v4_2.py; required for Turing and MLC-SLM.",
    )
    parser.add_argument(
        "--stg-python",
        type=str,
        default=sys.executable,
        help="Python executable with the STG 4.2 dependencies (lhotse, cytoolz, jiwer).",
    )
    parser.add_argument(
        "--notsofar1-root",
        "--notsofar1-manifest-root",
        dest="notsofar1_root",
        type=str,
        default=DEFAULT_NOTSOFAR1_ROOT,
        help="DFW NOTSOFAR1 dataset root.",
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
    selected_sets = _resolve_values(args.sets, available_benchmarks())
    stg_script = Path(args.stg_v4_2_script).expanduser().resolve() if args.stg_v4_2_script else None
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Manifest root: {manifest_root}")
    print(f"Data/artifacts root: {data_dir}")
    all_counts = {}
    for benchmark in selected_sets:
        print(f"Preparing {benchmark}...", flush=True)
        if benchmark == TURING_BENCHMARK:
            count = prepare_turing_diarization_v1(
                source_root=Path(args.turing_root).expanduser().resolve(),
                output_dir=output_dir,
                data_dir=data_dir,
                audio_prefix=audio_prefix,
                no_audio=args.no_audio,
                max_samples=args.max_samples,
                speaker_tag_style=args.speaker_tag_style,
                stg_script=stg_script,
                python_executable=args.stg_python,
            )
        elif benchmark == MLC_SLM_2_BENCHMARK:
            count = prepare_mlc_slm_2_dev_13l(
                opus_dev_root=Path(args.mlc_slm_2_opus_dev_root).expanduser().resolve(),
                output_dir=output_dir,
                data_dir=data_dir,
                audio_prefix=audio_prefix,
                no_audio=args.no_audio,
                max_samples=args.max_samples,
                speaker_tag_style=args.speaker_tag_style,
                stg_script=stg_script,
                python_executable=args.stg_python,
            )
        elif benchmark in NOTSOFAR1_BENCHMARKS:
            count = prepare_notsofar1_eval(
                benchmark,
                manifest_root=Path(args.notsofar1_root).expanduser().resolve(),
                output_dir=output_dir,
                data_dir=data_dir,
                audio_prefix=audio_prefix,
                no_audio=args.no_audio,
                max_samples=args.max_samples,
                speaker_tag_style=args.speaker_tag_style,
            )
        else:
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
