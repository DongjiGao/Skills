# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from nemo_skills.dataset.utils import get_mcq_fields

DATASET_REPO_ID = "cais/mmlu"
DATASET_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
EXPECTED_ROWS = {"dev": 285, "test": 14042, "val": 1531}

# mmlu subcategories from https://github.com/hendrycks/test/blob/master/categories.py
subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}


def format_entry(entry):
    subject = entry["subject"]
    answer_index = int(entry["answer"])
    if answer_index not in range(4):
        raise ValueError(f"Invalid MMLU answer index: {answer_index}")
    return {
        "expected_answer": chr(ord("A") + answer_index),
        "subtopic": subject,
        "subset_for_metrics": subcategories[subject][0],
        "examples_type": f"mmlu_few_shot_{subject}",
        "extra_fields": {
            "source_repo": DATASET_REPO_ID,
            "source_revision": DATASET_REVISION,
        },
        **get_mcq_fields(entry["question"], entry["choices"]),
    }


def save_data(split):
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    output_file = str(data_dir / f"{split}.jsonl")

    dataset_split = "validation" if split == "val" else split
    data = load_dataset(
        DATASET_REPO_ID,
        "all",
        split=dataset_split,
        revision=DATASET_REVISION,
    )
    if len(data) != EXPECTED_ROWS[split]:
        raise ValueError(
            f"Expected {EXPECTED_ROWS[split]} {split} rows from "
            f"{DATASET_REPO_ID}@{DATASET_REVISION}, found {len(data)}"
        )

    with open(output_file, "wt", encoding="utf-8") as fout:
        fout.writelines(json.dumps(format_entry(entry)) + "\n" for entry in data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="test",
        choices=("dev", "test", "val"),
    )
    args = parser.parse_args()

    save_data(args.split)
