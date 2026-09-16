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

from collections import Counter

from nemo_skills.dataset.fleurs import prepare
from nemo_skills.dataset.fleurs.languages import CER_LOCALES, FLEURS_LANGUAGES, LANGUAGE_SETS, NEMOTRON_ASR_LOCALES


def test_nemotron_asr_language_set_uses_available_prd_table_locales():
    additions = {"hy_am", "id_id", "nb_no", "tr_tr", "fil_ph", "vi_vn"}

    assert len(NEMOTRON_ASR_LOCALES) == 38
    assert set(NEMOTRON_ASR_LOCALES) <= set(FLEURS_LANGUAGES)
    assert additions <= set(NEMOTRON_ASR_LOCALES)
    assert "nn_no" not in NEMOTRON_ASR_LOCALES
    assert LANGUAGE_SETS["nemotron-asr"] == NEMOTRON_ASR_LOCALES


def test_nemotron_asr_additions_have_expected_error_rate_type():
    additions = {"hy_am", "id_id", "nb_no", "tr_tr", "fil_ph", "vi_vn"}

    assert additions & CER_LOCALES == {"vi_vn"}
    assert additions - CER_LOCALES == {"hy_am", "id_id", "nb_no", "tr_tr", "fil_ph"}


def test_st_preparation_loads_each_locale_archive_once(monkeypatch, tmp_path):
    calls = Counter()

    def load_fleurs(locale, split, local_dir):
        calls[locale] += 1
        return [
            {
                "id": 1,
                "raw_transcription": f"raw-{locale}",
                "transcription": f"text-{locale}",
                "wav_filename": f"{locale}.wav",
                "audio": {"array": [0.0, 0.0], "sampling_rate": 2},
            }
        ]

    monkeypatch.setattr(prepare, "load_fleurs", load_fleurs)

    records = prepare._collect_st_records(
        ["en_us", "de_de", "fr_fr"],
        tmp_path / "audio",
        tmp_path / "cache",
        "test",
        no_audio=True,
    )

    assert len(records) == 4
    assert calls == {"en_us": 1, "de_de": 1, "fr_fr": 1}
