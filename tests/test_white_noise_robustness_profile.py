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

PROFILE_MODULE = "nemo_skills.dataset.librispeech-test-clean-white-noise-0db"
profile_config = importlib.import_module(PROFILE_MODULE)
profile = importlib.import_module(f"{PROFILE_MODULE}.prepare")


def test_profile_matches_whisper_figure5_registration_and_sources():
    assert profile_config.METRICS_TYPE == "audio"
    assert "normalization_mode=standard" in profile_config.EVAL_ARGS
    assert "enable_audio=true" in profile_config.GENERATION_ARGS
    assert profile.LIBRISPEECH_EXPECTED_UTTERANCES == 2620
    assert profile.LIBRISPEECH_MD5 == "32fa31d27d2e1cad72775fee3f4849a9"
    assert profile.TARGET_SNR_DB == 0.0
    assert profile.WHISPER_PROTOCOL_SECTION == "3.7 / Figure 5"
    assert profile.ADT_REVISION == "6f06bcc4ad9c4234f4a9ec655814889ba712c8e7"
    assert set(profile.ADT_SOURCE_SHA256) == {
        "degradationUnit_addNoise.m",
        "degradationUnit_addSound.m",
        "adthelper_normalizeAudio.m",
    }


def test_white_noise_realization_is_order_independent_uniform_and_frozen():
    first, digest, seed = profile.generate_white_noise("1089-134686-0000", 100_000)
    again, again_digest, again_seed = profile.generate_white_noise("1089-134686-0000", 100_000)
    other, _, _ = profile.generate_white_noise("1089-134686-0001", 100_000)

    np.testing.assert_array_equal(first, again)
    assert digest == again_digest
    assert seed == again_seed
    assert digest == "a97332073a01d79f577206e03757c6161991bd6e8bf87d8d9250792fb01d9138"
    assert np.all(first >= -0.5)
    assert np.all(first < 0.5)
    assert abs(float(first.mean())) < 0.005
    assert not np.array_equal(first, other)


def test_literal_adt_port_hits_zero_db_and_always_peak_normalizes():
    clean = np.array([0.02, -0.01, 0.03, -0.04], dtype=np.float64)
    noise = np.array([-0.4, 0.1, 0.2, -0.3], dtype=np.float64)
    mixture, metadata = profile.mix_adt_white_noise(clean, noise)

    expected_scale = np.sqrt(np.sum(clean**2) / np.sum(noise**2))
    unnormalized = clean + expected_scale * noise
    expected_gain = profile.PEAK_AMPLITUDE / np.max(np.abs(unnormalized))
    expected = (unnormalized * expected_gain).astype(np.float32)

    np.testing.assert_allclose(mixture, expected, rtol=0.0, atol=1e-7)
    assert abs(metadata["measured_snr_db"]) < 1e-12
    assert metadata["normalization_gain"] > 1.0
    assert np.max(np.abs(mixture)) <= profile.PEAK_AMPLITUDE + 1e-7


def test_manifest_records_every_reproduction_choice():
    metadata = {
        "speech_power": 1.0,
        "raw_noise_power": 2.0,
        "noise_scale": 2**-0.5,
        "pre_normalization_peak": 1.1,
        "normalization_gain": profile.PEAK_AMPLITUDE / 1.1,
        "measured_snr_db": 0.0,
    }
    row = profile.create_manifest_entry(
        utterance_id="1089-134686-0000",
        transcript="HELLO WORLD",
        duration=1.0,
        audio_path=f"/data/{profile.BENCHMARK_NAME}/audio/1089-134686-0000.wav",
        realization_sha256="ab" * 32,
        realization_seed=17,
        mix_metadata=metadata,
    )

    assert row["expected_answer"] == "HELLO WORLD"
    assert row["messages"][0]["content"] == "Transcribe the following audio."
    assert row["subset_for_metrics"] == "test-clean_white-noise_0db"
    assert row["target_snr_db"] == 0.0
    assert row["noise_distribution"] == "uniform[-0.5,0.5)"
    assert row["noise_prng"] == "numpy.PCG64"
    assert row["realization_seed_uint128"] == "17"
    assert row["protocol_source_section"] == "3.7 / Figure 5"
    assert row["adt_source_revision"] == profile.ADT_REVISION
