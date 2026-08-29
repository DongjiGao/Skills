# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import importlib

import numpy as np
import pytest

PROFILE_MODULE = "nemo_skills.dataset.librispeech-test-clean-musan-freesound-0db"
profile_config = importlib.import_module(PROFILE_MODULE)
profile = importlib.import_module(f"{PROFILE_MODULE}.prepare")


def test_zero_db_profile_registration_is_distinct_standard_wer():
    assert profile_config.METRICS_TYPE == "audio"
    assert "normalization_mode=standard" in profile_config.EVAL_ARGS
    assert "enable_audio=true" in profile_config.GENERATION_ARGS
    assert profile.BENCHMARK_NAME != "musan"
    assert profile.LIBRISPEECH_EXPECTED_UTTERANCES == 2620
    assert profile.LIBRISPEECH_MD5 == "32fa31d27d2e1cad72775fee3f4849a9"
    assert profile.MUSAN_HF_REVISION == "3edcfdf89b56dbe6a395ff29f9c29489e03d1321"


def test_noise_inventory_validation_is_order_independent_and_frozen(monkeypatch):
    paths = ["noise/free-sound/b.wav", "noise/free-sound/a.wav"]
    monkeypatch.setattr(profile, "MUSAN_FREESOUND_EXPECTED_FILES", len(paths))
    monkeypatch.setattr(profile, "MUSAN_FREESOUND_INVENTORY_SHA256", profile.inventory_digest(paths))
    assert profile.validate_noise_inventory(reversed(paths)) == sorted(paths)

    with pytest.raises(ValueError, match="inventory digest mismatch"):
        profile.validate_noise_inventory([paths[0], "noise/free-sound/c.wav"])


def test_selection_and_circular_crop_are_deterministic():
    paths = ["noise/free-sound/a.wav", "noise/free-sound/b.wav"]
    first = profile.select_noise("1089-134686-0000", paths, 5)
    second = profile.select_noise("1089-134686-0000", list(reversed(paths)), 5)
    assert first == profile.select_noise("1089-134686-0000", paths, 5)
    assert first == second
    assert profile.circular_crop(np.array([10.0, 20.0, 30.0]), 5, 2).tolist() == [30.0, 10.0, 20.0, 30.0, 10.0]


def test_mix_is_exact_zero_db_and_peak_gain_preserves_snr():
    phase = np.linspace(0.0, 8.0 * np.pi, 1600, endpoint=False)
    clean = 0.9 * np.sin(phase)
    noise = 0.7 * np.cos(phase)

    mixture, metadata = profile.mix_at_zero_db(clean, noise, crop_offset=0)

    assert abs(metadata["measured_snr_db"]) < 1e-12
    assert metadata["peak_gain"] < 1.0
    assert np.max(np.abs(mixture)) <= profile.PEAK_LIMIT + 1e-7
    assert mixture.dtype == np.float32


def test_zero_db_manifest_records_full_reproduction_recipe():
    metadata = {
        "speech_rms": 0.1,
        "noise_crop_rms": 0.2,
        "noise_scale": 0.5,
        "pre_peak_abs_max": 1.2,
        "peak_gain": 0.8325,
        "measured_snr_db": 0.0,
    }
    row = profile.create_manifest_entry(
        utterance_id="1089-134686-0000",
        transcript="HELLO WORLD",
        duration=1.0,
        audio_path="/data/librispeech-test-clean-musan-freesound-0db/audio/1089-134686-0000.wav",
        noise_path="noise/free-sound/noise-free-sound-0000.wav",
        crop_offset=17,
        digest_hex="ab" * 32,
        mix_metadata=metadata,
    )
    assert row["messages"][0]["content"] == "Transcribe the following audio."
    assert row["expected_answer"] == "HELLO WORLD"
    assert row["task_type"] == "ASR"
    assert row["target_snr_db"] == 0.0
    assert row["noise_crop_offset_samples"] == 17
    assert row["profile_version"] == profile.PROFILE_VERSION
    assert row["noise_inventory_sha256"] == profile.MUSAN_FREESOUND_INVENTORY_SHA256


def test_official_musan_hallucination_threshold_is_strict():
    from nemo_skills.evaluation.evaluator.audio import evaluate_hallucination

    at_threshold = evaluate_hallucination("", "x" * 25, {"audio_duration": 1.0})
    above_threshold = evaluate_hallucination("", "x" * 26, {"audio_duration": 1.0})
    assert at_threshold["char_rate"] == 1500.0
    assert at_threshold["hallucination_rate"] == 0.0
    assert at_threshold["is_correct"] is True
    assert above_threshold["char_rate"] == 1560.0
    assert above_threshold["hallucination_rate"] == 1.0
    assert above_threshold["is_correct"] is False
