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

"""Prepare the official merged FLEURS-LID test set (all 102 locales)."""

from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
from pathlib import Path

import soundfile as sf
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from nemo_skills.dataset.fleurs.languages import FLEURS_LANGUAGES, LANGUAGE_SETS, get_lang_group, get_lang_name
from nemo_skills.dataset.fleurs.prepare import parse_tsv

HF_REPO_ID = "google/fleurs"
HF_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
OFFICIAL_SPLIT = "test"
CANONICAL_LABELS = LANGUAGE_SETS["all"]
PROMPT = (
    "Identify the language spoken in this audio. Respond with exactly one of the following locale labels and "
    "nothing else: " + ", ".join(CANONICAL_LABELS) + "."
)


def _load_locale(locale: str, cache_dir: Path) -> list[dict]:
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
    metadata = parse_tsv(tsv_path)
    rows = []
    with tarfile.open(tar_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            filename = Path(member.name).name
            if filename not in metadata:
                continue
            fileobj = archive.extractfile(member)
            if fileobj is None:
                continue
            samples, sampling_rate = sf.read(io.BytesIO(fileobj.read()), dtype="float32")
            rows.append({**metadata[filename], "samples": samples, "sampling_rate": int(sampling_rate)})
    return rows


def _container_audio_path(locale: str, filename: str) -> str:
    audio_root = os.getenv("NEMO_SKILLS_AUDIO_ROOT", "/data").rstrip("/")
    return f"{audio_root}/fleurs-lid/audio/{locale}/{filename}"


def _build_record(locale: str, row: dict) -> dict:
    audio_path = _container_audio_path(locale, row["wav_filename"])
    duration = len(row["samples"]) / row["sampling_rate"]
    audio = {"path": audio_path, "duration": duration}
    return {
        "id": f"{locale}-{row['id']}",
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
            "source_repo": HF_REPO_ID,
            "source_revision": HF_REVISION,
            "source_split": OFFICIAL_SPLIT,
        },
    }


def prepare_fleurs_lid(data_dir: Path, no_audio: bool = False) -> Path:
    if len(CANONICAL_LABELS) != 102 or set(CANONICAL_LABELS) != set(FLEURS_LANGUAGES):
        raise ValueError("FLEURS-LID requires the exact 102 canonical FLEURS locales")

    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = data_dir / "hf-fleurs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / "test.jsonl"
    locale_counts: dict[str, int] = {}
    with open(output_path, "w", encoding="utf-8") as output:
        for locale in tqdm(CANONICAL_LABELS, desc="FLEURS-LID locales"):
            rows = _load_locale(locale, cache_dir)
            locale_counts[locale] = len(rows)
            audio_dir = data_dir / "audio" / locale
            if not no_audio:
                audio_dir.mkdir(parents=True, exist_ok=True)
            for row in rows:
                if not no_audio:
                    sf.write(audio_dir / row["wav_filename"], row["samples"], row["sampling_rate"])
                output.write(json.dumps(_build_record(locale, row), ensure_ascii=False) + "\n")

    missing = set(CANONICAL_LABELS) - {locale for locale, count in locale_counts.items() if count > 0}
    if missing:
        raise ValueError(f"Official FLEURS-LID preparation has empty locales: {sorted(missing)}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official FLEURS-LID test data for all 102 locales")
    parser.add_argument("--data_dir", default=os.getenv("NEMO_SKILLS_DATA_DIR"))
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) / "fleurs-lid" if args.data_dir else Path(__file__).parent
    output_path = prepare_fleurs_lid(data_dir, no_audio=args.no_audio)
    print(f"FLEURS-LID prepared at {output_path}")


if __name__ == "__main__":
    main()
