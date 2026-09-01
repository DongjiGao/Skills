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

"""Prepare the pinned merged FLEURS-LID test set (all 102 locales)."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
import shutil
import tarfile
from pathlib import Path, PurePosixPath

import soundfile as sf
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from nemo_skills.dataset.fleurs.languages import FLEURS_LANGUAGES, LANGUAGE_SETS, get_lang_group, get_lang_name

HF_REPO_ID = "google/fleurs"
HF_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
OFFICIAL_SPLIT = "test"
CANONICAL_LABELS = LANGUAGE_SETS["all"]
EXPECTED_TOTAL_ROWS = 77810
EXPECTED_LOCALE_COUNTS = {
    "af_za": 264,
    "am_et": 516,
    "ar_eg": 428,
    "as_in": 984,
    "ast_es": 946,
    "az_az": 923,
    "be_by": 967,
    "bg_bg": 658,
    "bn_in": 920,
    "bs_ba": 925,
    "ca_es": 940,
    "ceb_ph": 541,
    "ckb_iq": 922,
    "cmn_hans_cn": 945,
    "cs_cz": 723,
    "cy_gb": 1021,
    "da_dk": 930,
    "de_de": 862,
    "el_gr": 650,
    "en_us": 647,
    "es_419": 908,
    "et_ee": 893,
    "fa_ir": 871,
    "ff_sn": 660,
    "fi_fi": 918,
    "fil_ph": 964,
    "fr_fr": 676,
    "ga_ie": 842,
    "gl_es": 927,
    "gu_in": 1000,
    "ha_ng": 621,
    "he_il": 792,
    "hi_in": 418,
    "hr_hr": 914,
    "hu_hu": 905,
    "hy_am": 932,
    "id_id": 687,
    "ig_ng": 969,
    "is_is": 46,
    "it_it": 865,
    "ja_jp": 650,
    "jv_id": 728,
    "ka_ge": 979,
    "kam_ke": 827,
    "kea_cv": 864,
    "kk_kz": 856,
    "km_kh": 771,
    "kn_in": 838,
    "ko_kr": 382,
    "ky_kg": 977,
    "lb_lu": 934,
    "lg_ug": 723,
    "ln_cd": 478,
    "lo_la": 405,
    "lt_lt": 986,
    "luo_ke": 256,
    "lv_lv": 851,
    "mi_nz": 1008,
    "mk_mk": 973,
    "ml_in": 958,
    "mn_mn": 949,
    "mr_in": 1015,
    "ms_my": 749,
    "mt_mt": 926,
    "my_mm": 880,
    "nb_no": 357,
    "ne_np": 726,
    "nl_nl": 364,
    "nso_za": 790,
    "ny_mw": 761,
    "oc_fr": 998,
    "om_et": 41,
    "or_in": 883,
    "pa_in": 574,
    "pl_pl": 758,
    "ps_af": 512,
    "pt_br": 919,
    "ro_ro": 883,
    "ru_ru": 775,
    "sd_in": 980,
    "sk_sk": 792,
    "sl_si": 834,
    "sn_zw": 925,
    "so_so": 1019,
    "sr_rs": 700,
    "sv_se": 759,
    "sw_ke": 487,
    "ta_in": 591,
    "te_in": 472,
    "tg_tj": 600,
    "th_th": 1021,
    "tr_tr": 743,
    "uk_ua": 750,
    "umb_ao": 379,
    "ur_pk": 299,
    "uz_uz": 862,
    "vi_vn": 857,
    "wo_sn": 371,
    "xh_za": 1041,
    "yo_ng": 831,
    "yue_hant_hk": 819,
    "zu_za": 854,
}
# SHA-256 of canonical JSON {locale: sorted(test TSV wav_filename values)} at
# HF_REVISION. This authenticates the exact 102-locale merged test inventory,
# not merely its aggregate row count.
EXPECTED_WAV_INVENTORY_SHA256 = "58ed6f8d74f07dc401a4844b24587576a2aced7fd00fbf741fe89eb278e8d27d"
EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHANNELS = 1
PROVENANCE_FILENAME = "provenance.json"
LOCALE_SHARDS_DIRNAME = ".locale-shards"
DEFAULT_NUM_WORKERS = 8
PROMPT = (
    "Identify the language spoken in this audio. Respond with exactly one of the following locale labels and "
    "nothing else: " + ", ".join(CANONICAL_LABELS) + "."
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory_sha256(inventory: dict[str, list[str] | set[str]]) -> str:
    canonical = {locale: sorted(filenames) for locale, filenames in inventory.items()}
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_tsv_strict(tsv_path: Path, locale: str) -> list[dict]:
    """Parse one pinned TSV without silently overwriting duplicate WAVs."""
    rows = []
    seen_wavs: set[str] = set()
    with tsv_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) != 7:
                raise ValueError(f"FLEURS-LID {locale} TSV row {line_number} has {len(fields)} fields; expected 7")
            row_id, wav_filename, raw_transcription, transcription, _phonemes, _samples, _gender = fields
            if (
                not wav_filename
                or PurePosixPath(wav_filename).name != wav_filename
                or not wav_filename.endswith(".wav")
            ):
                raise ValueError(f"FLEURS-LID {locale} TSV row {line_number} has unsafe WAV filename {wav_filename!r}")
            if wav_filename in seen_wavs:
                raise ValueError(f"FLEURS-LID {locale} TSV has duplicate WAV filename {wav_filename!r}")
            seen_wavs.add(wav_filename)
            try:
                parsed_row_id = int(row_id)
            except ValueError as error:
                raise ValueError(
                    f"FLEURS-LID {locale} TSV row {line_number} has non-integer source id {row_id!r}"
                ) from error
            rows.append(
                {
                    "id": parsed_row_id,
                    "raw_transcription": raw_transcription,
                    "transcription": transcription,
                    "wav_filename": wav_filename,
                }
            )
    return rows


def _read_source_audio(archive: tarfile.TarFile, member: tarfile.TarInfo, locale: str, filename: str):
    fileobj = archive.extractfile(member)
    if fileobj is None:
        raise ValueError(f"FLEURS-LID {locale} tar member {member.name!r} is unreadable")
    payload = fileobj.read()
    try:
        with sf.SoundFile(io.BytesIO(payload)) as stream:
            sampling_rate = int(stream.samplerate)
            channels = int(stream.channels)
            frames = int(stream.frames)
            samples = stream.read(dtype="float32", always_2d=False)
    except Exception as error:
        raise ValueError(f"FLEURS-LID {locale} source audio {filename!r} has an unreadable header") from error
    if sampling_rate != EXPECTED_SAMPLE_RATE or channels != EXPECTED_CHANNELS or frames <= 0:
        raise ValueError(
            f"FLEURS-LID {locale} source audio {filename!r} must be nonempty "
            f"{EXPECTED_SAMPLE_RATE}-Hz mono; got rate={sampling_rate}, channels={channels}, frames={frames}"
        )
    if getattr(samples, "ndim", None) != 1 or len(samples) != frames:
        raise ValueError(f"FLEURS-LID {locale} source audio {filename!r} decoded inconsistently")
    return samples, sampling_rate


def _load_locale(locale: str, cache_dir: Path) -> tuple[list[dict], dict]:
    tsv_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=f"data/{locale}/{OFFICIAL_SPLIT}.tsv",
        repo_type="dataset",
        revision=HF_REVISION,
        local_dir=cache_dir,
    )
    tar_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=f"data/{locale}/audio/{OFFICIAL_SPLIT}.tar.gz",
        repo_type="dataset",
        revision=HF_REVISION,
        local_dir=cache_dir,
    )
    tsv_path = Path(tsv_path)
    tar_path = Path(tar_path)
    metadata = _parse_tsv_strict(tsv_path, locale)
    expected_count = EXPECTED_LOCALE_COUNTS.get(locale)
    if expected_count is None:
        raise ValueError(f"FLEURS-LID has no pinned inventory contract for locale {locale!r}")
    if len(metadata) != expected_count:
        raise ValueError(f"FLEURS-LID {locale} TSV row-count drift: expected {expected_count}, got {len(metadata)}")

    metadata_wavs = {row["wav_filename"] for row in metadata}
    rows: list[dict] = []
    with tarfile.open(tar_path, "r:gz") as archive:
        members: dict[str, tarfile.TarInfo] = {}
        for member in archive:
            if not member.isfile():
                continue
            filename = PurePosixPath(member.name).name
            if filename in members:
                raise ValueError(f"FLEURS-LID {locale} tar has duplicate WAV basename {filename!r}")
            members[filename] = member
        tar_wavs = set(members)
        if tar_wavs != metadata_wavs:
            missing = sorted(metadata_wavs - tar_wavs)
            unexpected = sorted(tar_wavs - metadata_wavs)
            raise ValueError(
                f"FLEURS-LID {locale} TSV/tar inventory mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        for row in metadata:
            filename = row["wav_filename"]
            samples, sampling_rate = _read_source_audio(archive, members[filename], locale, filename)
            rows.append({**row, "samples": samples, "sampling_rate": sampling_rate})

    artifact = {
        "rows": len(rows),
        "tsv_sha256": _sha256_file(tsv_path),
        "audio_tar_sha256": _sha256_file(tar_path),
        "wav_inventory_sha256": hashlib.sha256(
            json.dumps(sorted(metadata_wavs), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    return rows, artifact


def _container_audio_path(locale: str, filename: str) -> str:
    audio_root = os.getenv("NEMO_SKILLS_AUDIO_ROOT", "/data").rstrip("/")
    return f"{audio_root}/fleurs-lid/audio/{locale}/{filename}"


def _build_record(locale: str, row: dict) -> dict:
    audio_path = _container_audio_path(locale, row["wav_filename"])
    duration = len(row["samples"]) / row["sampling_rate"]
    audio = {"path": audio_path, "duration": duration}
    return {
        # Source sentence ids intentionally repeat across speakers. Locale+WAV
        # is the unique public test-example identity.
        "id": f"{locale}:{row['wav_filename']}",
        "expected_answer": locale,
        "audio_path": audio_path,
        "duration": duration,
        "messages": [{"role": "user", "content": PROMPT, "audio": audio}],
        "subset_for_metrics": locale,
        "task_type": "FLEURS-LID",
        "extra_fields": {
            "locale": locale,
            "locale_name": get_lang_name(locale),
            "lang_group": get_lang_group(locale),
            "source_id": row["id"],
            "source_wav_filename": row["wav_filename"],
            "source_wav_inventory_sha256": EXPECTED_WAV_INVENTORY_SHA256,
            "source_repo": HF_REPO_ID,
            "source_revision": HF_REVISION,
            "source_split": OFFICIAL_SPLIT,
        },
    }


def _validate_materialized_audio(path: Path, locale: str, filename: str) -> None:
    if not path.is_file():
        raise ValueError(f"FLEURS-LID --no-audio requires existing audio file for {locale}/{filename}: {path}")
    try:
        info = sf.info(str(path))
    except Exception as error:
        raise ValueError(f"FLEURS-LID materialized audio has an unreadable header: {path}") from error
    if (
        int(info.samplerate) != EXPECTED_SAMPLE_RATE
        or int(info.channels) != EXPECTED_CHANNELS
        or int(info.frames) <= 0
    ):
        raise ValueError(
            f"FLEURS-LID materialized audio must be nonempty {EXPECTED_SAMPLE_RATE}-Hz mono: "
            f"{path} (rate={info.samplerate}, channels={info.channels}, frames={info.frames})"
        )


def _validate_materialized_audio_against_source(path: Path, samples, sampling_rate: int, locale: str) -> None:
    """Prove that an interrupted run's WAV is the deterministic PCM16 source materialization."""
    _validate_materialized_audio(path, locale, path.name)
    try:
        materialized, observed_rate = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as error:
        raise ValueError(f"FLEURS-LID materialized audio is unreadable: {path}") from error
    canonical_stream = io.BytesIO()
    sf.write(canonical_stream, samples, sampling_rate, format="WAV", subtype="PCM_16")
    canonical_stream.seek(0)
    try:
        expected_samples, expected_rate = sf.read(canonical_stream, dtype="float32", always_2d=False)
    except Exception as error:
        raise ValueError(f"FLEURS-LID could not canonicalize pinned source audio for {locale}/{path.name}") from error
    if int(observed_rate) != int(sampling_rate) or int(expected_rate) != int(sampling_rate):
        raise ValueError(
            f"FLEURS-LID materialized audio sample-rate drift for {locale}/{path.name}: "
            f"expected {sampling_rate}, got materialized={observed_rate}, canonical={expected_rate}"
        )
    if materialized.shape != expected_samples.shape or not bool((materialized == expected_samples).all()):
        raise ValueError(
            f"FLEURS-LID materialized audio differs from canonical pinned source for {locale}/{path.name}; "
            "refusing to overwrite an existing immutable artifact"
        )


def _write_audio_atomic(path: Path, samples, sampling_rate: int, locale: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.wav")
    try:
        sf.write(str(temporary), samples, sampling_rate, format="WAV", subtype="PCM_16")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    _validate_materialized_audio(path, locale, path.name)


def _write_json_atomic(path: Path, value: dict) -> None:
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


def _validate_static_contract() -> None:
    labels = set(CANONICAL_LABELS)
    if labels != set(FLEURS_LANGUAGES):
        raise ValueError("FLEURS-LID requires the exact 102 canonical FLEURS locales")
    if labels != set(EXPECTED_LOCALE_COUNTS):
        raise ValueError("FLEURS-LID pinned locale-count contract does not match the canonical labels")
    if sum(EXPECTED_LOCALE_COUNTS.values()) != EXPECTED_TOTAL_ROWS:
        raise ValueError("FLEURS-LID pinned locale counts do not sum to the expected total")


def _locale_shard_paths(data_dir: Path, locale: str) -> tuple[Path, Path]:
    shard_dir = data_dir / LOCALE_SHARDS_DIRNAME
    return shard_dir / f"{locale}.jsonl", shard_dir / f"{locale}.provenance.json"


def _require_sha256(value, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"FLEURS-LID {field} is not a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"FLEURS-LID {field} is not a SHA-256 digest") from error
    return value


def _load_locale_shard(data_dir: Path, locale: str) -> dict:
    shard_path, receipt_path = _locale_shard_paths(data_dir, locale)
    if not shard_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"FLEURS-LID locale {locale} has no complete resumable shard")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"FLEURS-LID locale {locale} has an unreadable shard receipt") from error

    expected_receipt = {
        "schema_version": 1,
        "dataset": "fleurs-lid",
        "locale": locale,
        "manifest": shard_path.name,
        "source_repo": HF_REPO_ID,
        "source_revision": HF_REVISION,
        "source_split": OFFICIAL_SPLIT,
        "rows": EXPECTED_LOCALE_COUNTS[locale],
    }
    for field, expected in expected_receipt.items():
        if receipt.get(field) != expected:
            raise ValueError(
                f"FLEURS-LID locale {locale} shard receipt drift for {field}: "
                f"expected {expected!r}, got {receipt.get(field)!r}"
            )
    manifest_sha256 = _require_sha256(receipt.get("manifest_sha256"), f"{locale} manifest_sha256")
    if _sha256_file(shard_path) != manifest_sha256:
        raise ValueError(f"FLEURS-LID locale {locale} shard hash mismatch")

    source_artifact = receipt.get("source_artifact")
    if not isinstance(source_artifact, dict) or source_artifact.get("rows") != EXPECTED_LOCALE_COUNTS[locale]:
        raise ValueError(f"FLEURS-LID locale {locale} has an invalid source-artifact receipt")
    for field in ("tsv_sha256", "audio_tar_sha256", "wav_inventory_sha256"):
        _require_sha256(source_artifact.get(field), f"{locale} source_artifact.{field}")

    filenames: list[str] = []
    record_ids: set[str] = set()
    with shard_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except Exception as error:
                raise ValueError(f"FLEURS-LID locale {locale} shard row {line_number} is not JSON") from error
            extra_fields = record.get("extra_fields")
            filename = extra_fields.get("source_wav_filename") if isinstance(extra_fields, dict) else None
            if (
                not isinstance(filename, str)
                or PurePosixPath(filename).name != filename
                or not filename.endswith(".wav")
            ):
                raise ValueError(f"FLEURS-LID locale {locale} shard row {line_number} has an unsafe WAV filename")
            expected_id = f"{locale}:{filename}"
            expected_audio_path = _container_audio_path(locale, filename)
            if (
                record.get("id") != expected_id
                or record.get("expected_answer") != locale
                or record.get("subset_for_metrics") != locale
                or record.get("task_type") != "FLEURS-LID"
                or record.get("audio_path") != expected_audio_path
                or extra_fields.get("locale") != locale
                or extra_fields.get("source_repo") != HF_REPO_ID
                or extra_fields.get("source_revision") != HF_REVISION
                or extra_fields.get("source_split") != OFFICIAL_SPLIT
                or extra_fields.get("source_wav_inventory_sha256") != EXPECTED_WAV_INVENTORY_SHA256
            ):
                raise ValueError(f"FLEURS-LID locale {locale} shard row {line_number} violates the pinned contract")
            if expected_id in record_ids:
                raise ValueError(f"FLEURS-LID locale {locale} shard has duplicate record id {expected_id!r}")
            record_ids.add(expected_id)
            filenames.append(filename)

            audio_path = data_dir / "audio" / locale / filename
            _validate_materialized_audio(audio_path, locale, filename)
            info = sf.info(str(audio_path))
            duration = record.get("duration")
            if not isinstance(duration, (int, float)) or round(float(duration) * EXPECTED_SAMPLE_RATE) != int(
                info.frames
            ):
                raise ValueError(f"FLEURS-LID locale {locale} shard row {line_number} duration/audio-frame mismatch")

    expected_count = EXPECTED_LOCALE_COUNTS[locale]
    if len(filenames) != expected_count or len(record_ids) != expected_count:
        raise ValueError(
            f"FLEURS-LID locale {locale} shard row-count drift: expected {expected_count}, "
            f"got rows={len(filenames)}, unique_ids={len(record_ids)}"
        )
    inventory_sha256 = hashlib.sha256(json.dumps(sorted(filenames), separators=(",", ":")).encode("utf-8")).hexdigest()
    if source_artifact["wav_inventory_sha256"] != inventory_sha256:
        raise ValueError(f"FLEURS-LID locale {locale} source/shard WAV inventory mismatch")
    if receipt.get("wav_inventory_sha256") != inventory_sha256:
        raise ValueError(f"FLEURS-LID locale {locale} receipt/shard WAV inventory mismatch")
    return {
        "rows": expected_count,
        "filenames": filenames,
        "source_artifact": source_artifact,
        "manifest_sha256": manifest_sha256,
    }


def _prepare_locale_shard(data_dir: Path, cache_dir: Path, locale: str, no_audio: bool) -> dict:
    shard_path, receipt_path = _locale_shard_paths(data_dir, locale)
    if shard_path.is_file() and receipt_path.is_file():
        return _load_locale_shard(data_dir, locale)

    rows, source_artifact = _load_locale(locale, cache_dir)
    expected_count = EXPECTED_LOCALE_COUNTS[locale]
    if len(rows) != expected_count or source_artifact.get("rows") != expected_count:
        raise ValueError(
            f"FLEURS-LID locale {locale} row-count drift before publication: "
            f"expected {expected_count}, got rows={len(rows)}, artifact={source_artifact.get('rows')}"
        )

    shard_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_shard = shard_path.with_name(f".{shard_path.name}.{os.getpid()}.tmp")
    filenames: list[str] = []
    record_ids: set[str] = set()
    try:
        with temporary_shard.open("w", encoding="utf-8") as output:
            for row in rows:
                filename = row["wav_filename"]
                record = _build_record(locale, row)
                if record["id"] in record_ids:
                    raise ValueError(f"FLEURS-LID locale {locale} has duplicate record id {record['id']!r}")
                record_ids.add(record["id"])
                filenames.append(filename)

                audio_path = data_dir / "audio" / locale / filename
                if no_audio or audio_path.exists():
                    _validate_materialized_audio_against_source(
                        audio_path, row["samples"], row["sampling_rate"], locale
                    )
                else:
                    _write_audio_atomic(audio_path, row["samples"], row["sampling_rate"], locale)
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())

        if len(filenames) != expected_count or len(record_ids) != expected_count:
            raise ValueError(f"FLEURS-LID locale {locale} did not produce its exact pinned row count")
        inventory_sha256 = hashlib.sha256(
            json.dumps(sorted(filenames), separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if source_artifact.get("wav_inventory_sha256") != inventory_sha256:
            raise ValueError(f"FLEURS-LID locale {locale} source/output WAV inventory mismatch")
        receipt = {
            "schema_version": 1,
            "dataset": "fleurs-lid",
            "locale": locale,
            "manifest": shard_path.name,
            "manifest_sha256": _sha256_file(temporary_shard),
            "source_repo": HF_REPO_ID,
            "source_revision": HF_REVISION,
            "source_split": OFFICIAL_SPLIT,
            "rows": expected_count,
            "wav_inventory_sha256": inventory_sha256,
            "audio_mode": "validated-existing" if no_audio else "materialized-or-verified",
            "source_artifact": source_artifact,
        }
        os.replace(temporary_shard, shard_path)
        _write_json_atomic(receipt_path, receipt)
    finally:
        temporary_shard.unlink(missing_ok=True)
    return _load_locale_shard(data_dir, locale)


def _resolve_num_workers(num_workers: int | None) -> int:
    if num_workers is None:
        raw_value = os.getenv("NEMO_SKILLS_FLEURS_LID_NUM_WORKERS", str(DEFAULT_NUM_WORKERS))
        try:
            num_workers = int(raw_value)
        except ValueError as error:
            raise ValueError("NEMO_SKILLS_FLEURS_LID_NUM_WORKERS must be an integer") from error
    if num_workers < 1 or num_workers > 32:
        raise ValueError("FLEURS-LID num_workers must be between 1 and 32")
    return min(num_workers, len(CANONICAL_LABELS))


def prepare_fleurs_lid(data_dir: Path, no_audio: bool = False, num_workers: int | None = None) -> Path:
    _validate_static_contract()

    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = data_dir / "hf-fleurs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / "test.jsonl"
    provenance_path = data_dir / PROVENANCE_FILENAME
    temporary_output = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    locale_counts: dict[str, int] = {}
    source_artifacts: dict[str, dict] = {}
    inventory: dict[str, list[str]] = {}
    record_ids: set[str] = set()
    locale_wavs: set[tuple[str, str]] = set()
    total_rows = 0
    try:
        worker_count = _resolve_num_workers(num_workers)
        summaries: dict[str, dict] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_prepare_locale_shard, data_dir, cache_dir, locale, no_audio): locale
                for locale in CANONICAL_LABELS
            }
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="FLEURS-LID locales",
            ):
                locale = futures[future]
                summaries[locale] = future.result()

        with temporary_output.open("wb") as output:
            for locale in CANONICAL_LABELS:
                summary = summaries[locale]
                shard_path, _receipt_path = _locale_shard_paths(data_dir, locale)
                with shard_path.open("rb") as source:
                    shutil.copyfileobj(source, output)
                locale_counts[locale] = summary["rows"]
                source_artifacts[locale] = summary["source_artifact"]
                inventory[locale] = summary["filenames"]
                for filename in summary["filenames"]:
                    locale_wav = (locale, filename)
                    record_id = f"{locale}:{filename}"
                    if locale_wav in locale_wavs or record_id in record_ids:
                        raise ValueError(f"FLEURS-LID has duplicate locale+WAV identity {locale_wav!r}")
                    locale_wavs.add(locale_wav)
                    record_ids.add(record_id)
                total_rows += summary["rows"]
            output.flush()
            os.fsync(output.fileno())

        if locale_counts != EXPECTED_LOCALE_COUNTS:
            raise ValueError(
                "FLEURS-LID observed locale counts do not match the pinned contract: "
                f"expected={EXPECTED_LOCALE_COUNTS}, observed={locale_counts}"
            )
        if total_rows != EXPECTED_TOTAL_ROWS or len(record_ids) != EXPECTED_TOTAL_ROWS:
            raise ValueError(
                f"FLEURS-LID total-row drift: expected {EXPECTED_TOTAL_ROWS}, "
                f"got rows={total_rows}, unique_ids={len(record_ids)}"
            )
        inventory_digest = _inventory_sha256(inventory)
        if inventory_digest != EXPECTED_WAV_INVENTORY_SHA256:
            raise ValueError(
                "FLEURS-LID merged WAV inventory drift: "
                f"expected {EXPECTED_WAV_INVENTORY_SHA256}, got {inventory_digest}"
            )

        manifest_sha256 = _sha256_file(temporary_output)
        receipt = {
            "schema_version": 1,
            "dataset": "fleurs-lid",
            "manifest": output_path.name,
            "manifest_sha256": manifest_sha256,
            "source_repo": HF_REPO_ID,
            "source_revision": HF_REVISION,
            "source_split": OFFICIAL_SPLIT,
            "total_rows": total_rows,
            "locale_counts": locale_counts,
            "wav_inventory_sha256": inventory_digest,
            "audio_mode": "validated-existing" if no_audio else "materialized",
            "audio_contract": {
                "sample_rate": EXPECTED_SAMPLE_RATE,
                "channels": EXPECTED_CHANNELS,
                "nonempty": True,
            },
            "source_artifacts": source_artifacts,
            "locale_shards": {locale: summaries[locale]["manifest_sha256"] for locale in CANONICAL_LABELS},
            "resumable_locale_shards": True,
            "preparation_workers": worker_count,
        }
        os.replace(temporary_output, output_path)
        _write_json_atomic(provenance_path, receipt)
    finally:
        temporary_output.unlink(missing_ok=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official FLEURS-LID test data for all 102 locales")
    parser.add_argument("--data_dir", default=os.getenv("NEMO_SKILLS_DATA_DIR"))
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help=(
            "Do not materialize WAVs. Every referenced file must already exist under "
            "<data_dir>/fleurs-lid/audio and pass the 16-kHz mono header contract."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Concurrent per-locale workers (default: NEMO_SKILLS_FLEURS_LID_NUM_WORKERS or 8).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir) / "fleurs-lid" if args.data_dir else Path(__file__).parent
    output_path = prepare_fleurs_lid(data_dir, no_audio=args.no_audio, num_workers=args.num_workers)
    print(f"FLEURS-LID prepared at {output_path}")


if __name__ == "__main__":
    main()
