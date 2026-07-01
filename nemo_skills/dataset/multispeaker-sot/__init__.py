# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""multi-speaker SOT ASR benchmark group."""

IS_BENCHMARK_GROUP = True
REQUIRES_DATA_DIR = True
SCORE_MODULE = "nemo_skills.dataset.multispeaker-sot.multispeaker_sot_score"

BENCHMARK_MANIFESTS = {
    "under20s-ami-ihm-dev": "under20s_eval_set/ami_ihm_dev_10-20s.json",
    "under20s-ami-sdm-test": "under20s_eval_set/ami_sdm_test_10-20s.json",
    "under20s-ami-ihm-test": "under20s_eval_set/ami_test_10-20s.json",
    "under20s-chaes-ch109-test": "under20s_eval_set/ch109_test_10-20s.json",
    "under20s-chaes-ch109-shuf10-test": "under20s_eval_set/ch109_10-20s_shuf10.json",
    "under20s-libri-am-dev": "under20s_eval_set/libri-am_dev_10-20s.json",
    "under20s-lsm-test-clean-1mix": "under20s_eval_set/lsm-test-clean-1mix_nemo_sot.json",
    "under20s-lsm-test-clean-2mix": "under20s_eval_set/lsm-test-clean-2mix_nemo_sot.json",
    "under20s-lsm-test-clean-3mix": "under20s_eval_set/lsm-test-clean-3mix_nemo_sot.json",
    "under20s-mixer6-ch04-test": "under20s_eval_set/mixer6_test_CH04mic_10-20s.json",
    "over20s-ami-ihm-test": "over20s_eval_set/ami_ihm_test.json",
    "over20s-ami-sdm-test": "over20s_eval_set/ami_sdm_test.json",
    "over20s-chaes-ch109-test": "over20s_eval_set/ch109_test.json",
    "over20s-mixer6-ch04-test": "over20s_eval_set/mixer6_ch4_eval.json",
}

BENCHMARKS = {f"multispeaker-sot.{name}": {} for name in BENCHMARK_MANIFESTS}

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
