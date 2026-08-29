# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_skills.dataset.numb3rs.prepare import (
    EXPECTED_MANIFEST_ROWS,
    HF_REPO_ID,
    HF_REVISION,
    MANIFEST_FILENAME,
    save_audio_and_format_entry,
)


def test_numb3rs_source_revision_is_pinned_in_every_formatted_row(tmp_path):
    row = {
        "original_text": "$100",
        "text": "one hundred dollars",
        "audio_filepath": "./MONEY/MONEY_1.wav",
        "duration": 1.25,
        "info": "MONEY",
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
    assert MANIFEST_FILENAME == "manifest.jsonl"
    assert EXPECTED_MANIFEST_ROWS == 10131
    assert formatted["extra_fields"] == {
        "source_repo": HF_REPO_ID,
        "source_revision": HF_REVISION,
        "source_manifest": MANIFEST_FILENAME,
    }
    assert formatted["audio_filepath"] == "/data/numb3rs/Numb3rs/MONEY/MONEY_1.wav"
    assert formatted["text_tn"] == "$100"
    assert formatted["text_itn"] == "one hundred dollars"


def test_numb3rs_audio_is_copied_from_pinned_snapshot_without_decoding(tmp_path):
    source_root = tmp_path / "snapshot"
    source_audio = source_root / "MONEY" / "MONEY_1.wav"
    source_audio.parent.mkdir(parents=True)
    source_audio.write_bytes(b"unchanged wav bytes")
    audio_dir = tmp_path / "prepared" / "MONEY"
    audio_dir.mkdir(parents=True)
    row = {
        "original_text": "$100",
        "text": "one hundred dollars",
        "audio_filepath": "./MONEY/MONEY_1.wav",
        "duration": 1.25,
        "info": "MONEY",
    }

    save_audio_and_format_entry(
        row,
        "MONEY",
        audio_dir,
        0,
        with_audio=True,
        audio_prefix="/data/numb3rs",
        source_root=source_root,
    )

    assert (audio_dir / "MONEY_1.wav").read_bytes() == b"unchanged wav bytes"
