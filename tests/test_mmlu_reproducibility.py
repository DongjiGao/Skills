import importlib.util
from pathlib import Path

from nemo_skills.dataset.mmlu import prepare as mmlu

ROOT = Path(__file__).parents[1]


def _load_mmlu_pro_prepare():
    path = ROOT / "nemo_skills" / "dataset" / "mmlu-pro" / "prepare.py"
    spec = importlib.util.spec_from_file_location("mmlu_pro_prepare", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_mmlu_uses_pinned_official_source_and_preserves_schema():
    assert mmlu.DATASET_REPO_ID == "cais/mmlu"
    assert mmlu.DATASET_REVISION == "c30699e8356da336a370243923dbaf21066bb9fe"
    assert mmlu.EXPECTED_ROWS == {"dev": 285, "test": 14042, "val": 1531}

    formatted = mmlu.format_entry(
        {
            "question": "Question?",
            "subject": "abstract_algebra",
            "choices": ["zero", "four", "two", "six"],
            "answer": 1,
        }
    )
    assert formatted["expected_answer"] == "B"
    assert formatted["subtopic"] == "abstract_algebra"
    assert formatted["subset_for_metrics"] == "math"
    assert formatted["examples_type"] == "mmlu_few_shot_abstract_algebra"
    assert formatted["B"] == "four"
    assert formatted["extra_fields"] == {
        "source_repo": mmlu.DATASET_REPO_ID,
        "source_revision": mmlu.DATASET_REVISION,
    }


def test_mmlu_pro_uses_pinned_official_source_and_preserves_schema():
    mmlu_pro = _load_mmlu_pro_prepare()
    assert mmlu_pro.DATASET_REPO_ID == "TIGER-Lab/MMLU-Pro"
    assert mmlu_pro.DATASET_REVISION == "b189ec765aa7ed75c8acfea42df31fdae71f97be"
    assert mmlu_pro.EXPECTED_ROWS == {"validation": 70, "test": 12032}

    formatted = mmlu_pro.format_entry(
        {
            "question": "Question?",
            "options": [
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
                "ten",
            ],
            "answer": "I",
            "category": "computer science",
        }
    )
    assert formatted["expected_answer"] == "I"
    assert formatted["subset_for_metrics"] == "computer_science"
    assert formatted["examples_type"] == "mmlu_pro_few_shot_computer_science"
    assert formatted["I"] == "nine"
    assert formatted["extra_fields"] == {
        "source_repo": mmlu_pro.DATASET_REPO_ID,
        "source_revision": mmlu_pro.DATASET_REVISION,
    }
