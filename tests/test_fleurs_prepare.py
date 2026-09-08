# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_skills.dataset.fleurs.prepare import _src_extra_fields


def _row() -> dict:
    return {"transcription": "transcript", "raw_transcription": "raw transcript"}


def test_mandarin_and_cantonese_opt_in_to_han_adjacent_cer():
    for locale in ("cmn_hans_cn", "yue_hant_hk"):
        assert _src_extra_fields(_row(), locale, for_asr=True)["cer_whitespace_mode"] == "han_adjacent"


def test_other_cer_locales_do_not_inherit_han_adjacent_cer():
    for locale in ("ja_jp", "ko_kr", "vi_vn"):
        assert "cer_whitespace_mode" not in _src_extra_fields(_row(), locale, for_asr=True)


def test_translation_records_do_not_carry_cer_whitespace_metadata():
    assert "cer_whitespace_mode" not in _src_extra_fields(_row(), "cmn_hans_cn")
