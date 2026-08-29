# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_skills.dataset.numb3rs.prepare import HF_REPO_ID, HF_REVISION, save_audio_and_format_entry


def test_numb3rs_source_revision_is_pinned_in_every_formatted_row(tmp_path):
    row = {
        "original_text": "$100",
        "text": "one hundred dollars",
        "file_name": "MONEY/MONEY_1.wav",
        "duration": 1.25,
        "audio": None,
    }
    formatted = save_audio_and_format_entry(
        row,
        "MONEY",
        tmp_path,
        0,
        with_audio=False,
        audio_prefix="/data/numb3rs",
    )

    assert HF_REPO_ID == "nvidia/Numb3rs"
    assert HF_REVISION == "192908075e1bd293914cc7b508f4a183ba6ef2b8"
    assert formatted["extra_fields"] == {
        "source_repo": HF_REPO_ID,
        "source_revision": HF_REVISION,
    }
    assert formatted["audio_filepath"] == "/data/numb3rs/Numb3rs/MONEY/MONEY_1.wav"
    assert formatted["text_tn"] == "$100"
    assert formatted["text_itn"] == "one hundred dollars"
