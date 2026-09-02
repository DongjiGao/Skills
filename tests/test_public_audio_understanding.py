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

import json

from nemo_skills.dataset.mmar.prepare import (
    HF_REVISION as MMAR_REVISION,
)
from nemo_skills.dataset.mmar.prepare import (
    OFFICIAL_SCORER_REVISION as MMAR_SCORER_REVISION,
)
from nemo_skills.dataset.mmar.prepare import (
    format_entry as format_mmar_entry,
)
from nemo_skills.dataset.mmar.prepare import (
    select_rows as select_mmar_rows,
)
from nemo_skills.dataset.mmau.prepare import (
    HF_REVISION as MMAU_REVISION,
)
from nemo_skills.dataset.mmau.prepare import (
    OFFICIAL_REPO_REVISION as MMAU_REPO_REVISION,
)
from nemo_skills.dataset.mmau.prepare import (
    format_entry as format_mmau_entry,
)
from nemo_skills.dataset.mmau.prepare import (
    select_rows as select_mmau_rows,
)
from nemo_skills.evaluation.evaluator import _resolve_eval_type
from nemo_skills.evaluation.evaluator.set_token_mcq import eval_set_token_mcq, official_set_token_match
from nemo_skills.evaluation.metrics.map_metrics import get_metrics
from nemo_skills.evaluation.metrics.set_token_mcq_metrics import SetTokenMCQMetrics


def test_official_set_token_match_parity_cases():
    choices = ["Too little", "Just right, no opinion", "Too much, need to share", "Cannot tell"]

    assert official_set_token_match("Too little", "The answer is: too little.", choices)
    assert not official_set_token_match("Too little", "Too little, not too much.", choices)
    assert not official_set_token_match("Too little", "A", choices)
    assert not official_set_token_match("Too little", "", choices)


def test_set_token_evaluator_retains_official_output_key(tmp_path):
    output_path = tmp_path / "output.jsonl"
    sample = {
        "generation": "I choose Open Grassland.",
        "answer": "Open Grassland",
        "expected_answer": "Open Grassland",
        "choices": ["Mountain Canyon", "Dense Forest", "Open Grassland", "City Streets"],
    }
    output_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    eval_set_token_mcq({"input_file": str(output_path), "official_output_key": "answer_prediction"})

    evaluated = json.loads(output_path.read_text(encoding="utf-8"))
    assert evaluated["answer_prediction"] == sample["generation"]
    assert evaluated["predicted_answer"] == sample["generation"]
    assert evaluated["symbolic_correct"] is True
    assert evaluated["match"] == 1


def test_set_token_metric_reports_micro_accuracy():
    metrics = SetTokenMCQMetrics()
    metrics.update([{"predicted_answer": "yes", "symbolic_correct": True}])
    metrics.update([{"predicted_answer": "no", "symbolic_correct": False}])

    result = metrics.get_metrics()["pass@1"]
    assert result["accuracy"] == 50.0
    assert result["num_entries"] == 2


def test_evaluator_and_metric_are_registered():
    evaluator, is_class = _resolve_eval_type("set-token-mcq")
    assert evaluator is eval_set_token_mcq
    assert is_class is False
    assert isinstance(get_metrics("set_token_mcq"), SetTokenMCQMetrics)


def _mmar_row(modality: str) -> dict:
    return {
        "id": f"id-{modality}",
        "audio_path": f"./audio/{modality}.wav",
        "question": "What does the speaker imply?",
        "choices": ["Too little", "Too much", "Just enough", "Cannot tell"],
        "answer": "Too little",
        "modality": modality,
        "category": "Semantic Layer",
        "sub-category": "Content Analysis",
        "timestamp": "00:00:01,00:00:07",
        "thinking": "Private reference rationale that must not enter the prompt.",
    }


def test_mmar_selects_only_pure_speech_and_preserves_official_fields():
    speech = _mmar_row("speech")
    selected = select_mmar_rows([speech, _mmar_row("mix-sound-speech"), _mmar_row("music")], max_samples=5)
    assert selected == [speech]

    formatted = format_mmar_entry(speech, "/data/mmar")
    assert formatted["answer"] == speech["answer"]
    assert formatted["expected_answer"] == speech["answer"]
    assert formatted["choices"] == speech["choices"]
    assert formatted["modality"] == "speech"
    assert formatted["subset_for_metrics"] == "Semantic Layer"
    assert formatted["messages"][0]["audio"] == {"path": "/data/mmar/audio/speech.wav", "duration": 6.0}
    assert "Private reference rationale" not in formatted["messages"][0]["content"]
    assert formatted["dataset_revision"] == MMAR_REVISION
    assert formatted["official_scorer_revision"] == MMAR_SCORER_REVISION


def _mmau_row(task: str) -> dict:
    return {
        "id": f"id-{task}",
        "audio_id": f"./test-mini-audios/{task}.wav",
        "question": "Which source is audible?",
        "choices": ["Man", "Woman", "Child", "Robot"],
        "answer": "Man",
        "dataset": "AudioSet",
        "task": task,
        "split": "test-mini",
        "category": "Reasoning",
        "sub-category": "Acoustic Source Inference",
        "difficulty": "medium",
    }


def test_mmau_selects_public_test_mini_speech_and_sound_and_preserves_fields():
    sound = _mmau_row("sound")
    speech = _mmau_row("speech")
    music = _mmau_row("music")
    wrong_split = {**_mmau_row("sound"), "id": "wrong-split", "split": "test"}

    selected = select_mmau_rows([sound, speech, music, wrong_split], max_samples=10)
    assert selected == [sound, speech]

    formatted = format_mmau_entry(sound, "/data/mmau")
    assert formatted["answer"] == sound["answer"]
    assert formatted["expected_answer"] == sound["answer"]
    assert formatted["choices"] == sound["choices"]
    assert formatted["task"] == "sound"
    assert formatted["subset_for_metrics"] == "sound"
    assert formatted["messages"][0]["audio"] == {"path": "/data/mmau/audio/sound.wav"}
    assert formatted["dataset_revision"] == MMAU_REVISION
    assert formatted["official_repo_revision"] == MMAU_REPO_REVISION


def test_revision_pins_are_immutable_commit_hashes():
    pins = [MMAR_REVISION, MMAR_SCORER_REVISION, MMAU_REVISION, MMAU_REPO_REVISION]
    assert all(len(pin) == 40 and set(pin) <= set("0123456789abcdef") for pin in pins)
