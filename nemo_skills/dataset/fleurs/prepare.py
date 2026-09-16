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

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from nemo_skills.dataset.fleurs.languages import (
    CER_LOCALES,
    LANGUAGE_SETS,
    LOCALES,
    get_lang_group,
    get_lang_name,
)

HF_REPO_ID = "google/fleurs"
HF_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
OFFICIAL_PRODUCT_LANGUAGES = LANGUAGE_SETS["nemotron-asr"]
EXPECTED_TEST_ASR_ROWS = 29_322
EXPECTED_TEST_ST_ROWS = 51_394
EXPECTED_TEST_LOCALE_COUNTS = {
    "ar_eg": 428,
    "hy_am": 932,
    "bg_bg": 658,
    "cmn_hans_cn": 945,
    "hr_hr": 914,
    "cs_cz": 723,
    "da_dk": 930,
    "nl_nl": 364,
    "en_us": 647,
    "et_ee": 893,
    "fil_ph": 964,
    "fi_fi": 918,
    "fr_fr": 676,
    "de_de": 862,
    "el_gr": 650,
    "he_il": 792,
    "hi_in": 418,
    "hu_hu": 905,
    "id_id": 687,
    "it_it": 865,
    "ja_jp": 650,
    "ko_kr": 382,
    "lv_lv": 851,
    "lt_lt": 986,
    "mt_mt": 926,
    "nb_no": 357,
    "pl_pl": 758,
    "pt_br": 919,
    "ro_ro": 883,
    "ru_ru": 775,
    "sk_sk": 792,
    "sl_si": 834,
    "es_419": 908,
    "sv_se": 759,
    "th_th": 1021,
    "tr_tr": 743,
    "uk_ua": 750,
    "vi_vn": 857,
}
EXPECTED_SAMPLE_RATE = 16_000
EXPECTED_CHANNELS = 1
MINIMUM_AUDIO_SAMPLES = 320
LOCALE_SHARDS_DIRNAME = ".locale-shards"
PROVENANCE_FILENAME = "provenance.json"
DEFAULT_NUM_WORKERS = 8


def parse_tsv(tsv_path: str) -> dict[str, dict]:
    """Parse FLEURS TSV metadata file. Returns dict keyed by audio filename."""
    metadata = {}
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row_id, wav_filename, raw_transcription, transcription, _, _, _ = line.split("\t")
            metadata[wav_filename] = {
                "id": int(row_id),
                "raw_transcription": raw_transcription,
                "transcription": transcription,
                "wav_filename": wav_filename,
            }
    return metadata


def load_fleurs(locale: str, split: str, local_dir: str) -> list[dict]:
    """Download and parse a FLEURS locale/split directly from HuggingFace."""
    tsv_path = hf_hub_download(
        repo_id="google/fleurs",
        filename=f"data/{locale}/{split}.tsv",
        repo_type="dataset",
        revision=HF_REVISION,
        local_dir=local_dir,
    )
    tar_path = hf_hub_download(
        repo_id="google/fleurs",
        filename=f"data/{locale}/audio/{split}.tar.gz",
        repo_type="dataset",
        revision=HF_REVISION,
        local_dir=local_dir,
    )

    metadata = parse_tsv(tsv_path)
    rows = []
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            wav_filename = Path(member.name).name
            if wav_filename not in metadata:
                continue
            audio_bytes = tar.extractfile(member).read()
            y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            row = dict(metadata[wav_filename])
            row["audio"] = {"array": y, "sampling_rate": sr}
            rows.append(row)
    return rows


def index_by_id(rows: list[dict]) -> dict[int, dict]:
    """Index translation text by sentence id without treating recordings as duplicates.

    FLEURS sentence ids are deliberately non-unique: a sentence may have
    multiple independently recorded WAVs.  All recordings for one id must carry
    identical text, so selecting one as the translation reference is lossless.
    Source rows themselves remain row/WAV aligned and are never deduplicated.
    """
    indexed: dict[int, dict] = {}
    for row in rows:
        source_id = row["id"]
        previous = indexed.get(source_id)
        if previous is not None and any(
            previous.get(field) != row.get(field) for field in ("raw_transcription", "transcription")
        ):
            raise ValueError(f"FLEURS source id {source_id} has conflicting translation text")
        indexed.setdefault(source_id, row)
    return indexed


def build_translation_pairs(languages: list[str]) -> list[tuple[str, str]]:
    """Build (en_us -> lang) and (lang -> en_us) pairs for each language."""
    pairs = set()
    for lang in languages:
        if lang == "en_us":
            continue
        pairs.add(("en_us", lang))
        pairs.add((lang, "en_us"))
    return sorted(pairs)


def prepare_audio(item: dict) -> tuple[np.ndarray, int, float]:
    audio_dict = item["audio"]
    y, sr = audio_dict["array"], audio_dict["sampling_rate"]
    duration = float(len(y) / sr)
    return y, sr, duration


def get_container_audio_path(locale: str, wav_filename: str) -> str:
    return f"/dataset/fleurs/audio/{locale}/{wav_filename}"


def save_audio(y: np.ndarray, sr: int, wav_path: Path) -> None:
    if not wav_path.exists():
        sf.write(str(wav_path), y, sr)


def get_st_instruction(target_locale: str) -> str:
    tgt_lang_name = get_lang_name(target_locale)
    return f"Please translate the given speech to {tgt_lang_name}."


def get_asr_instruction() -> str:
    return "Produce a verbatim transcript of the audio."


def _build_record(
    expected_answer: str,
    instruction: str,
    container_audio_path: str,
    duration: float,
    subset_for_metrics: str,
    task_type: str,
    extra_fields: dict,
    source: str | None = None,
    reference: str | None = None,
) -> dict:
    audio_metadata = {"path": container_audio_path, "duration": duration}
    record = {
        "expected_answer": expected_answer,
        "audio_path": container_audio_path,
        "duration": duration,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. /no_think"},
            {"role": "user", "content": instruction, "audio": audio_metadata},
        ],
        "subset_for_metrics": subset_for_metrics,
        "task_type": f"Multilingual-{task_type.upper()}",
        "extra_fields": extra_fields,
    }
    if source is not None:
        record["source"] = source
    if reference is not None:
        record["reference"] = reference
    return record


def _src_extra_fields(source_row: dict, src_locale: str, *, for_asr: bool = False) -> dict:
    extra_fields = {
        "src_text": source_row["transcription"],
        "src_raw_text": source_row["raw_transcription"],
        "src_lang_name": get_lang_name(src_locale),
        "src_lang": src_locale,
        "src_lang_group": get_lang_group(src_locale),
        "use_cer": src_locale in CER_LOCALES,
    }
    # FLEURS Mandarin/Cantonese references may contain character-segmentation
    # spaces. Ignore spaces around Han while preserving any embedded Latin word
    # boundaries. Other CER locales keep their benchmark text unchanged.
    if for_asr and src_locale in {"cmn_hans_cn", "yue_hant_hk"}:
        extra_fields["cer_whitespace_mode"] = "han_adjacent"
    return extra_fields


def _collect_asr_records(
    languages: list[str],
    audio_dir: Path,
    local_dir: Path,
    split: str,
    no_audio: bool,
) -> list[dict]:
    records: list[dict] = []
    for src_locale in languages:
        locale_audio_dir = audio_dir / src_locale
        if not no_audio:
            locale_audio_dir.mkdir(parents=True, exist_ok=True)
        for source_row in tqdm(load_fleurs(src_locale, split, local_dir=local_dir), desc=src_locale):
            y, sr, duration = prepare_audio(source_row)
            wav_filename = source_row["wav_filename"]
            wav_path = locale_audio_dir / wav_filename
            cpath = get_container_audio_path(src_locale, wav_filename)
            if not no_audio:
                save_audio(y, sr, wav_path)
            records.append(
                _build_record(
                    expected_answer=source_row["transcription"],
                    instruction=get_asr_instruction(),
                    container_audio_path=cpath,
                    duration=duration,
                    subset_for_metrics=src_locale,
                    task_type="ASR",
                    extra_fields=_src_extra_fields(source_row, src_locale, for_asr=True),
                )
            )
    return records


def _collect_st_records(
    languages: list[str],
    audio_dir: Path,
    local_dir: Path,
    split: str,
    no_audio: bool,
) -> list[dict]:
    pairs = build_translation_pairs(languages)
    row_cache: dict[str, list[dict]] = {}
    target_cache: dict[str, dict[int, dict]] = {}

    def get_rows(locale: str) -> list[dict]:
        if locale not in row_cache:
            row_cache[locale] = load_fleurs(locale, split, local_dir=local_dir)
        return row_cache[locale]

    def get_target_index(locale: str) -> dict[int, dict]:
        if locale not in target_cache:
            target_cache[locale] = index_by_id(get_rows(locale))
        return target_cache[locale]

    records: list[dict] = []
    for src_locale, tgt_locale in pairs:
        locale_audio_dir = audio_dir / src_locale
        if not no_audio:
            locale_audio_dir.mkdir(parents=True, exist_ok=True)
        target_by_id = get_target_index(tgt_locale)
        tag = f"{src_locale}->{tgt_locale}"
        for source_row in tqdm(get_rows(src_locale), desc=tag):
            target_row = target_by_id.get(source_row["id"])
            if target_row is None:
                continue
            y, sr, duration = prepare_audio(source_row)
            wav_filename = source_row["wav_filename"]
            wav_path = locale_audio_dir / wav_filename
            cpath = get_container_audio_path(src_locale, wav_filename)
            if not no_audio:
                save_audio(y, sr, wav_path)
            extra = _src_extra_fields(source_row, src_locale)
            extra.update(
                {
                    "tgt_text": target_row["transcription"],
                    "tgt_raw_text": target_row["raw_transcription"],
                    "tgt_lang_name": get_lang_name(tgt_locale),
                    "tgt_lang": tgt_locale,
                    "tgt_lang_group": get_lang_group(tgt_locale),
                }
            )
            records.append(
                _build_record(
                    expected_answer=target_row["raw_transcription"],
                    instruction=get_st_instruction(tgt_locale),
                    container_audio_path=cpath,
                    duration=duration,
                    subset_for_metrics=tag,
                    task_type="AST",
                    extra_fields=extra,
                    source=source_row["raw_transcription"],
                    reference=target_row["raw_transcription"],
                )
            )
    return records


def _dump_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as out:
        out.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"FLEURS {field} is not a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"FLEURS {field} is not a SHA-256 digest") from error
    return value


def _inventory_sha256(filenames: list[str] | set[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(filenames), separators=(",", ":")).encode("utf-8")).hexdigest()


def _parse_tsv_rows(tsv_path: Path, locale: str) -> list[dict]:
    rows: list[dict] = []
    seen_wavs: set[str] = set()
    # A FLEURS id identifies a sentence, not an audio row.  The pinned bg_bg
    # test TSV, for example, has 658 real rows/WAVs but only 344 sentence ids.
    # WAV filename is the unique row/audio identity; repeated ids are retained.
    with tsv_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) != 7:
                raise ValueError(f"FLEURS {locale} TSV row {line_number} has {len(fields)} fields; expected 7")
            row_id, wav_filename, raw_transcription, transcription, _phonemes, _samples, _gender = fields
            if (
                not wav_filename
                or PurePosixPath(wav_filename).name != wav_filename
                or not wav_filename.endswith(".wav")
            ):
                raise ValueError(f"FLEURS {locale} TSV row {line_number} has unsafe WAV filename {wav_filename!r}")
            try:
                parsed_id = int(row_id)
            except ValueError as error:
                raise ValueError(
                    f"FLEURS {locale} TSV row {line_number} has non-integer source id {row_id!r}"
                ) from error
            if wav_filename in seen_wavs:
                raise ValueError(f"FLEURS {locale} TSV has duplicate WAV filename at row {line_number}")
            seen_wavs.add(wav_filename)
            rows.append(
                {
                    "id": parsed_id,
                    "raw_transcription": raw_transcription,
                    "transcription": transcription,
                    "wav_filename": wav_filename,
                }
            )
    return rows


def _read_source_audio(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    locale: str,
    filename: str,
) -> tuple[np.ndarray, int]:
    fileobj = archive.extractfile(member)
    if fileobj is None:
        raise ValueError(f"FLEURS {locale} source audio {filename!r} is unreadable")
    try:
        with sf.SoundFile(io.BytesIO(fileobj.read())) as stream:
            sampling_rate = int(stream.samplerate)
            channels = int(stream.channels)
            frames = int(stream.frames)
            samples = stream.read(dtype="float32", always_2d=False)
    except Exception as error:
        raise ValueError(f"FLEURS {locale} source audio {filename!r} has an unreadable header") from error
    if (
        sampling_rate != EXPECTED_SAMPLE_RATE
        or channels != EXPECTED_CHANNELS
        or frames < MINIMUM_AUDIO_SAMPLES
        or getattr(samples, "ndim", None) != 1
        or len(samples) != frames
    ):
        raise ValueError(
            f"FLEURS {locale} source audio {filename!r} must be mono, {EXPECTED_SAMPLE_RATE} Hz, "
            f"and at least {MINIMUM_AUDIO_SAMPLES} samples; got rate={sampling_rate}, "
            f"channels={channels}, frames={frames}"
        )
    return samples, sampling_rate


def _load_product_locale_source(locale: str, split: str, cache_dir: Path) -> tuple[list[dict], dict]:
    tsv_path = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=f"data/{locale}/{split}.tsv",
            repo_type="dataset",
            revision=HF_REVISION,
            local_dir=cache_dir,
        )
    )
    tar_path = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=f"data/{locale}/audio/{split}.tar.gz",
            repo_type="dataset",
            revision=HF_REVISION,
            local_dir=cache_dir,
        )
    )
    metadata = _parse_tsv_rows(tsv_path, locale)
    expected_count = EXPECTED_TEST_LOCALE_COUNTS.get(locale) if split == "test" else None
    if expected_count is not None and len(metadata) != expected_count:
        raise ValueError(f"FLEURS {locale} test row-count drift: expected {expected_count}, got {len(metadata)}")

    metadata_wavs = {row["wav_filename"] for row in metadata}
    rows: list[dict] = []
    with tarfile.open(tar_path, "r:gz") as archive:
        members: dict[str, tarfile.TarInfo] = {}
        for member in archive:
            if not member.isfile():
                continue
            filename = PurePosixPath(member.name).name
            if filename in members:
                raise ValueError(f"FLEURS {locale} tar has duplicate WAV basename {filename!r}")
            members[filename] = member
        if set(members) != metadata_wavs:
            missing = sorted(metadata_wavs - set(members))
            unexpected = sorted(set(members) - metadata_wavs)
            raise ValueError(
                f"FLEURS {locale} TSV/tar inventory mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        for row in metadata:
            filename = row["wav_filename"]
            samples, sampling_rate = _read_source_audio(archive, members[filename], locale, filename)
            rows.append({**row, "samples": samples, "sampling_rate": sampling_rate})

    artifact = {
        "rows": len(rows),
        "tsv_sha256": _sha256_file(tsv_path),
        "audio_tar_sha256": _sha256_file(tar_path),
        "wav_inventory_sha256": _inventory_sha256(metadata_wavs),
        "source_id_inventory_sha256": _inventory_sha256([row["id"] for row in metadata]),
        "unique_source_ids": len({row["id"] for row in metadata}),
    }
    return rows, artifact


def _validate_materialized_audio(path: Path, locale: str, duration: float) -> None:
    if not path.is_file():
        raise ValueError(f"FLEURS requires existing audio file for {locale}/{path.name}: {path}")
    try:
        info = sf.info(str(path))
    except Exception as error:
        raise ValueError(f"FLEURS materialized audio has an unreadable header: {path}") from error
    if (
        int(info.samplerate) != EXPECTED_SAMPLE_RATE
        or int(info.channels) != EXPECTED_CHANNELS
        or int(info.frames) < MINIMUM_AUDIO_SAMPLES
        or round(duration * info.samplerate) != int(info.frames)
    ):
        raise ValueError(
            f"FLEURS materialized audio contract failed for {path}: "
            f"rate={info.samplerate}, channels={info.channels}, frames={info.frames}, duration={duration}"
        )


def _validate_materialized_audio_against_source(
    path: Path,
    samples: np.ndarray,
    sampling_rate: int,
    locale: str,
) -> None:
    duration = len(samples) / sampling_rate
    _validate_materialized_audio(path, locale, duration)
    materialized, observed_rate = sf.read(str(path), dtype="float32", always_2d=False)
    canonical_stream = io.BytesIO()
    sf.write(canonical_stream, samples, sampling_rate, format="WAV", subtype="PCM_16")
    canonical_stream.seek(0)
    expected, expected_rate = sf.read(canonical_stream, dtype="float32", always_2d=False)
    if (
        int(observed_rate) != sampling_rate
        or int(expected_rate) != sampling_rate
        or materialized.shape != expected.shape
        or not bool((materialized == expected).all())
    ):
        raise ValueError(
            f"FLEURS materialized audio differs from canonical pinned source for {locale}/{path.name}; "
            "refusing to overwrite an existing immutable artifact"
        )


def _write_audio_atomic(path: Path, samples: np.ndarray, sampling_rate: int, locale: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.wav")
    try:
        sf.write(str(temporary), samples, sampling_rate, format="WAV", subtype="PCM_16")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    _validate_materialized_audio(path, locale, len(samples) / sampling_rate)


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _locale_shard_paths(data_dir: Path, split: str, locale: str) -> tuple[Path, Path]:
    shard_dir = data_dir / LOCALE_SHARDS_DIRNAME / split
    return shard_dir / f"{locale}.jsonl", shard_dir / f"{locale}.provenance.json"


def _load_product_locale_shard(data_dir: Path, split: str, locale: str) -> dict:
    shard_path, receipt_path = _locale_shard_paths(data_dir, split, locale)
    if not shard_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"FLEURS {locale} has no complete resumable {split} shard")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"FLEURS {locale} has an unreadable shard receipt") from error
    expected_receipt = {
        "schema_version": 2,
        "dataset": "fleurs",
        "locale": locale,
        "split": split,
        "manifest": shard_path.name,
        "source_repo": HF_REPO_ID,
        "source_revision": HF_REVISION,
    }
    for field, expected in expected_receipt.items():
        if receipt.get(field) != expected:
            raise ValueError(
                f"FLEURS {locale} shard receipt drift for {field}: expected {expected!r}, got {receipt.get(field)!r}"
            )
    manifest_sha256 = _require_sha256(receipt.get("manifest_sha256"), f"{locale} manifest_sha256")
    if _sha256_file(shard_path) != manifest_sha256:
        raise ValueError(f"FLEURS {locale} shard hash mismatch")
    source_artifact = receipt.get("source_artifact")
    if not isinstance(source_artifact, dict):
        raise TypeError(f"FLEURS {locale} source-artifact receipt is missing")
    for field in ("tsv_sha256", "audio_tar_sha256", "wav_inventory_sha256", "source_id_inventory_sha256"):
        _require_sha256(source_artifact.get(field), f"{locale} source_artifact.{field}")

    rows: list[dict] = []
    seen_wavs: set[str] = set()
    with shard_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                row = json.loads(line)
            except Exception as error:
                raise ValueError(f"FLEURS {locale} shard row {line_number} is not JSON") from error
            if set(row) != {
                "id",
                "raw_transcription",
                "transcription",
                "wav_filename",
                "duration",
            }:
                raise ValueError(f"FLEURS {locale} shard row {line_number} has unexpected fields")
            source_id = row["id"]
            filename = row["wav_filename"]
            duration = row["duration"]
            if (
                isinstance(source_id, bool)
                or not isinstance(source_id, int)
                or not isinstance(filename, str)
                or PurePosixPath(filename).name != filename
                or not filename.endswith(".wav")
                or filename in seen_wavs
                or not isinstance(duration, (int, float))
                or isinstance(duration, bool)
                or duration <= 0
            ):
                raise ValueError(f"FLEURS {locale} shard row {line_number} violates its identity contract")
            seen_wavs.add(filename)
            _validate_materialized_audio(data_dir / "audio" / locale / filename, locale, float(duration))
            rows.append(row)
    index_by_id(rows)

    expected_count = EXPECTED_TEST_LOCALE_COUNTS.get(locale) if split == "test" else None
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"FLEURS {locale} shard row-count drift: expected {expected_count}, got {len(rows)}")
    if receipt.get("rows") != len(rows) or source_artifact.get("rows") != len(rows):
        raise ValueError(f"FLEURS {locale} shard receipt row-count drift")
    inventory_sha256 = _inventory_sha256(seen_wavs)
    source_id_inventory_sha256 = _inventory_sha256([row["id"] for row in rows])
    unique_source_ids = len({row["id"] for row in rows})
    if (
        receipt.get("wav_inventory_sha256") != inventory_sha256
        or source_artifact.get("wav_inventory_sha256") != inventory_sha256
    ):
        raise ValueError(f"FLEURS {locale} shard WAV inventory drift")
    if (
        receipt.get("source_id_inventory_sha256") != source_id_inventory_sha256
        or source_artifact.get("source_id_inventory_sha256") != source_id_inventory_sha256
        or receipt.get("unique_source_ids") != unique_source_ids
        or source_artifact.get("unique_source_ids") != unique_source_ids
    ):
        raise ValueError(f"FLEURS {locale} shard source-ID inventory drift")
    return {
        "rows": rows,
        "manifest_sha256": manifest_sha256,
        "source_artifact": source_artifact,
        "wav_inventory_sha256": inventory_sha256,
    }


def _prepare_product_locale_shard(
    data_dir: Path,
    cache_dir: Path,
    split: str,
    locale: str,
    no_audio: bool,
) -> dict:
    shard_path, receipt_path = _locale_shard_paths(data_dir, split, locale)
    if shard_path.is_file() and receipt_path.is_file():
        return _load_product_locale_shard(data_dir, split, locale)

    source_rows, source_artifact = _load_product_locale_source(locale, split, cache_dir)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = shard_path.with_name(f".{shard_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for row in source_rows:
                samples = row["samples"]
                sampling_rate = int(row["sampling_rate"])
                duration = len(samples) / sampling_rate
                audio_path = data_dir / "audio" / locale / row["wav_filename"]
                if no_audio or audio_path.exists():
                    _validate_materialized_audio_against_source(audio_path, samples, sampling_rate, locale)
                else:
                    _write_audio_atomic(audio_path, samples, sampling_rate, locale)
                output.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "raw_transcription": row["raw_transcription"],
                            "transcription": row["transcription"],
                            "wav_filename": row["wav_filename"],
                            "duration": duration,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            output.flush()
            os.fsync(output.fileno())
        receipt = {
            "schema_version": 2,
            "dataset": "fleurs",
            "locale": locale,
            "split": split,
            "manifest": shard_path.name,
            "manifest_sha256": _sha256_file(temporary),
            "source_repo": HF_REPO_ID,
            "source_revision": HF_REVISION,
            "rows": len(source_rows),
            "wav_inventory_sha256": source_artifact["wav_inventory_sha256"],
            "source_id_inventory_sha256": source_artifact["source_id_inventory_sha256"],
            "unique_source_ids": source_artifact["unique_source_ids"],
            "audio_mode": "validated-existing" if no_audio else "materialized-or-verified",
            "source_artifact": source_artifact,
        }
        os.replace(temporary, shard_path)
        _write_json_atomic(receipt_path, receipt)
    finally:
        temporary.unlink(missing_ok=True)
    return _load_product_locale_shard(data_dir, split, locale)


def _resolve_num_workers(num_workers: int | None, language_count: int) -> int:
    if num_workers is None:
        raw = os.getenv("NEMO_SKILLS_FLEURS_NUM_WORKERS", str(DEFAULT_NUM_WORKERS))
        try:
            num_workers = int(raw)
        except ValueError as error:
            raise ValueError("NEMO_SKILLS_FLEURS_NUM_WORKERS must be an integer") from error
    if num_workers < 1 or num_workers > 32:
        raise ValueError("FLEURS num_workers must be between 1 and 32")
    return min(num_workers, language_count)


def _is_official_product_contract(split: str, languages: list[str]) -> bool:
    return split == "test" and tuple(languages) == tuple(OFFICIAL_PRODUCT_LANGUAGES)


def _validate_static_product_contract() -> None:
    if tuple(EXPECTED_TEST_LOCALE_COUNTS) != tuple(OFFICIAL_PRODUCT_LANGUAGES):
        raise ValueError("FLEURS pinned locale counts do not match the ordered 38-locale product contract")
    if sum(EXPECTED_TEST_LOCALE_COUNTS.values()) != EXPECTED_TEST_ASR_ROWS:
        raise ValueError("FLEURS pinned locale counts do not sum to the expected ASR total")


def prepare_fleurs(
    data_dir: Path,
    split: str,
    languages: list[str],
    no_audio: bool,
    num_workers: int | None = None,
) -> None:
    if not languages:
        raise ValueError("No languages to process")
    if len(set(languages)) != len(languages):
        raise ValueError("FLEURS languages must be unique")
    official_product = _is_official_product_contract(split, languages)
    if official_product:
        _validate_static_product_contract()

    local_dir = data_dir / "hf-fleurs"
    local_dir.mkdir(parents=True, exist_ok=True)
    asr_dir = data_dir / "asr"
    st_dir = data_dir / "st"
    asr_dir.mkdir(parents=True, exist_ok=True)
    st_dir.mkdir(parents=True, exist_ok=True)

    asr_jsonl = asr_dir / f"{split}.jsonl"
    st_jsonl = st_dir / f"{split}.jsonl"

    worker_count = _resolve_num_workers(num_workers, len(languages))
    summaries: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _prepare_product_locale_shard,
                data_dir,
                local_dir,
                split,
                locale,
                no_audio,
            ): locale
            for locale in languages
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="FLEURS product locales",
        ):
            locale = futures[future]
            summaries[locale] = future.result()

    temporary_asr = asr_jsonl.with_name(f".{asr_jsonl.name}.{os.getpid()}.tmp")
    temporary_st = st_jsonl.with_name(f".{st_jsonl.name}.{os.getpid()}.tmp")
    asr_count = 0
    st_count = 0
    try:
        with temporary_asr.open("w", encoding="utf-8") as output:
            for locale in languages:
                for row in summaries[locale]["rows"]:
                    output.write(
                        json.dumps(
                            _build_record(
                                expected_answer=row["transcription"],
                                instruction=get_asr_instruction(),
                                container_audio_path=get_container_audio_path(locale, row["wav_filename"]),
                                duration=float(row["duration"]),
                                subset_for_metrics=locale,
                                task_type="ASR",
                                extra_fields=_src_extra_fields(row, locale, for_asr=True),
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    asr_count += 1
            output.flush()
            os.fsync(output.fileno())

        with temporary_st.open("w", encoding="utf-8") as output:
            for src_locale, tgt_locale in build_translation_pairs(languages):
                target_by_id = index_by_id(summaries[tgt_locale]["rows"])
                tag = f"{src_locale}->{tgt_locale}"
                for source_row in summaries[src_locale]["rows"]:
                    target_row = target_by_id.get(source_row["id"])
                    if target_row is None:
                        continue
                    extra = _src_extra_fields(source_row, src_locale)
                    extra.update(
                        {
                            "tgt_text": target_row["transcription"],
                            "tgt_raw_text": target_row["raw_transcription"],
                            "tgt_lang_name": get_lang_name(tgt_locale),
                            "tgt_lang": tgt_locale,
                            "tgt_lang_group": get_lang_group(tgt_locale),
                        }
                    )
                    output.write(
                        json.dumps(
                            _build_record(
                                expected_answer=target_row["raw_transcription"],
                                instruction=get_st_instruction(tgt_locale),
                                container_audio_path=get_container_audio_path(src_locale, source_row["wav_filename"]),
                                duration=float(source_row["duration"]),
                                subset_for_metrics=tag,
                                task_type="AST",
                                extra_fields=extra,
                                source=source_row["raw_transcription"],
                                reference=target_row["raw_transcription"],
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    st_count += 1
            output.flush()
            os.fsync(output.fileno())

        if official_product and (asr_count != EXPECTED_TEST_ASR_ROWS or st_count != EXPECTED_TEST_ST_ROWS):
            raise ValueError(
                "FLEURS product closure drift: "
                f"expected ASR/ST={EXPECTED_TEST_ASR_ROWS}/{EXPECTED_TEST_ST_ROWS}, "
                f"got {asr_count}/{st_count}"
            )
        receipt = {
            "schema_version": 2,
            "dataset": "fleurs",
            "source_repo": HF_REPO_ID,
            "source_revision": HF_REVISION,
            "source_split": split,
            "languages": languages,
            "language_count": len(languages),
            "asr_rows": asr_count,
            "st_rows": st_count,
            "asr_manifest": f"asr/{split}.jsonl",
            "asr_manifest_sha256": _sha256_file(temporary_asr),
            "st_manifest": f"st/{split}.jsonl",
            "st_manifest_sha256": _sha256_file(temporary_st),
            "locale_shards": {locale: summaries[locale]["manifest_sha256"] for locale in languages},
            "source_artifacts": {locale: summaries[locale]["source_artifact"] for locale in languages},
            "official_product_contract": official_product,
            "resumable_locale_shards": True,
            "preparation_workers": worker_count,
        }
        os.replace(temporary_asr, asr_jsonl)
        os.replace(temporary_st, st_jsonl)
        _write_json_atomic(data_dir / PROVENANCE_FILENAME, receipt)
    finally:
        temporary_asr.unlink(missing_ok=True)
        temporary_st.unlink(missing_ok=True)

    print(f"Fleurs ASR dataset prepared: {asr_jsonl} ({asr_count} records)")
    print(f"Fleurs ST dataset prepared: {st_jsonl} ({st_count} records)")


def main():
    parser = argparse.ArgumentParser(description="Prepare FLEURS Benchmark")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Output directory (defaults to $NEMO_SKILLS_DATA_DIR/fleurs or this package directory)",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "dev", "test"],
        help="Dataset split to process",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Languages to process",
    )
    parser.add_argument(
        "--language-set",
        choices=sorted(LANGUAGE_SETS),
        default="all",
        help="Named language set to process when --languages is not supplied",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip saving audio files (only create manifests)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Concurrent per-locale workers (default: NEMO_SKILLS_FLEURS_NUM_WORKERS or 8).",
    )
    args = parser.parse_args()

    languages = args.languages if args.languages is not None else list(LANGUAGE_SETS[args.language_set])

    unknown = set(languages) - LOCALES
    if unknown:
        raise ValueError(f"Unknown language(s): {', '.join(sorted(unknown))}. Available: {', '.join(sorted(LOCALES))}")

    if args.data_dir:
        data_dir = Path(args.data_dir) / "fleurs"
    else:
        data_dir = Path(__file__).parent
    data_dir.mkdir(parents=True, exist_ok=True)

    prepare_fleurs(
        data_dir=data_dir,
        split=args.split,
        languages=languages,
        no_audio=args.no_audio,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
