# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""multi-speaker SOT ASR benchmark group."""

IS_BENCHMARK_GROUP = True
REQUIRES_DATA_DIR = True
SCORE_MODULE = "nemo_skills.dataset.multispeaker-sot.multispeaker_sot_score"

BENCHMARK_MANIFESTS = {
    # under20s (10-20s segments)
    "under20s-ami-ihm-dev": "under20s_eval_set/ami_ihm_dev_10-20s.json",
    "under20s-ami-ihm-test": "under20s_eval_set/ami_test_10-20s.json",
    "under20s-ami-sdm-test": "under20s_eval_set/ami_sdm_test_10-20s.json",
    "under20s-chaes-ch109-test": "under20s_eval_set/ch109_test_10-20s.json",
    "under20s-libri-am-dev": "under20s_eval_set/libri-am_dev_10-20s.json",
    "under20s-mixer6-ch04-test": "under20s_eval_set/mixer6_test_CH04mic_10-20s.json",
    # 60-90s segments
    "60-90s-ami-ihm-test": "60s_90s_eval_set/ami_ihm_test_60-90s.json",
    "60-90s-ami-sdm-test": "60s_90s_eval_set/ami_sdm_test_60-90s.json",
    "60-90s-chaes-ch109-test": "60s_90s_eval_set/ch109_test_60-90s.json",
    "60-90s-mixer6-ch04-test": "60s_90s_eval_set/mixer6_ch4_test_60-90s.json",
    # 120-180s segments
    "120-180s-ami-ihm-test": "120s_180s_eval_set/ami_ihm_test_120-180s.json",
    "120-180s-ami-sdm-test": "120s_180s_eval_set/ami_sdm_test_120-180s.json",
    "120-180s-chaes-ch109-test": "120s_180s_eval_set/ch109_test_120-180s.json",
    "120-180s-mixer6-ch04-test": "120s_180s_eval_set/mixer6_ch4_test_120-180s.json",
    # full-length sessions
    "full-ami-ihm-test": "full_length_eval_set/ami_ihm_test_full.json",
    "full-ami-sdm-test": "full_length_eval_set/ami_sdm_test_full.json",
    "full-chaes-ch109-test": "full_length_eval_set/ch109_test_full.json",
    "full-mixer6-ch04-test": "full_length_eval_set/mixer6_ch4_test_full.json",
    # LibriSpeechMix (clean test, 1/2/3 speakers)
    "lsm-test-clean-1mix": "libri_speech_mix/lsm-test-clean-1mix_nemo_sot.json",
    "lsm-test-clean-2mix": "libri_speech_mix/lsm-test-clean-2mix_nemo_sot.json",
    "lsm-test-clean-3mix": "libri_speech_mix/lsm-test-clean-3mix_nemo_sot.json",
}

EVAL_GROUPS = (
    "under20s",
    "60-90s",
    "120-180s",
    "full",
    "lsm",
    "turing-diarization-v1",
    "mlc-slm-2-dev-13L",
    "notsofar1-eval-ihm",
    "notsofar1-eval-sc",
)

# A regular `multispeaker-sot` evaluation runs the five merged legacy benchmarks
# plus the Turing, MLC-SLM 2, and NOTSOFAR1 full-conversation benchmarks. Each
# merged legacy manifest retains `subset_for_metrics`, so all 21 underlying eval
# sets remain visible in the per-group metrics without launching 21 generation
# jobs.
BENCHMARKS = {f"multispeaker-sot.{name}": {} for name in EVAL_GROUPS}

FORMAT_EXPECTATIONS = {
    "multispeaker-sot": {
        "task_type": "MS-ASR",
        "speaker_label_namespace": "ordinal_speakers",
        "allowed_output_dialects": ["spk_tag", "s_bracket", "speaker_N_colon", "speaker_N_dash"],
        "preferred_output_dialect": "spk_tag",
        "repeated_speaker_turns_legal": True,
        "flattened_diagnostic_allowed": True,
    },
}
