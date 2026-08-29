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

import sys

from nemo_skills.dataset import musan
from nemo_skills.dataset.musan import prepare


def test_musan_is_regular_four_generation_audio_benchmark():
    assert not hasattr(musan, "IS_BENCHMARK_GROUP")
    assert not hasattr(musan, "BENCHMARKS")
    assert musan.NUM_SAMPLES == 4
    assert musan.METRICS_TYPE == "audio"
    assert "enable_audio=true" in musan.GENERATION_ARGS


def test_musan_manifest_matches_pinned_gym_protocol_for_every_category():
    for category in prepare.ALL_CATEGORIES:
        row = prepare.create_manifest_entry(
            audio_filename=f"musan_{category}_000000.wav",
            duration=2.5,
            category=category,
            sample_id=0,
            label=f"source-{category}",
        )
        assert row["messages"][0] == {"role": "system", "content": prepare.SYSTEM_PROMPT}
        assert row["messages"][1]["content"] == prepare.USER_PROMPT
        assert row["expected_answer"] == ""
        assert row["task_type"] == "Hallucination"
        assert row["subset_for_metrics"] == f"musan_{category}"
        assert row["protocol_revision"] == "68cb9969d656d259846c45e7369acfe6529f0e97"
        assert row["hallucination_char_rate_threshold"] == 1500.0
        assert row["hallucination_threshold_operator"] == ">"
        assert row["num_repeated_generations"] == 4


def test_musan_prepare_defaults_to_complete_openslr(monkeypatch, tmp_path):
    observed = {}

    def fake_load(source, output_dir):
        observed["source"] = source
        return tmp_path, source

    monkeypatch.setattr(prepare, "load_dataset_from_source", fake_load)
    monkeypatch.setattr(prepare, "process_category", lambda **kwargs: (0, []))
    monkeypatch.setattr(sys, "argv", ["prepare", "--output-dir", str(tmp_path)])
    prepare.main()

    assert observed["source"] == "openslr"
    assert prepare.DEFAULT_SOURCE == "openslr"
    assert prepare.ALL_CATEGORIES == ("noise", "music", "speech")
