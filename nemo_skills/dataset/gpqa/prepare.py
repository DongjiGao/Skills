# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import csv
import hashlib
import json
import random
import re
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from nemo_skills.dataset.utils import get_mcq_fields

"""
Preprocessing adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gpqa/generative/utils.py
"""

SOURCE_REPOSITORY = "Idavidrein/gpqa"
SOURCE_REVISION = "633f5ee89ab8ad4522a9f850766b73f62147ffdd"
SOURCE_FILES = {
    "extended": {
        "filename": "gpqa_extended.csv",
        "rows": 546,
        "sha256": "0926ee24949d02ed6748eb75a2611546c34479e30ddc42efd01d6f1681aaa48a",
    },
    "main": {
        "filename": "gpqa_main.csv",
        "rows": 448,
        "sha256": "acdeeac8f622267f2cd727d7d474202ea08dec80f7d3c3593b3ef8644f19b8e3",
    },
    "diamond": {
        "filename": "gpqa_diamond.csv",
        "rows": 198,
        "sha256": "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305",
    },
}


def _validate_source_csv(path: Path, split: str) -> dict:
    contract = SOURCE_FILES[split]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open("rt", encoding="utf-8", newline="") as stream:
        rows = sum(1 for _ in csv.DictReader(stream))
    if digest != contract["sha256"] or rows != contract["rows"]:
        raise RuntimeError(
            f"GPQA {split} source contract changed: sha256={digest}, rows={rows}; "
            f"expected sha256={contract['sha256']}, rows={contract['rows']}"
        )
    return {**contract, "split": split}


def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = text.replace("  ", " ")
    return text


def format_entry(entry):
    choices = [
        preprocess(entry["Incorrect Answer 1"]),
        preprocess(entry["Incorrect Answer 2"]),
        preprocess(entry["Incorrect Answer 3"]),
        preprocess(entry["Correct Answer"]),
    ]

    random.shuffle(choices)
    correct_answer_index = choices.index(preprocess(entry["Correct Answer"]))
    return {
        "expected_answer": f"{chr(65 + correct_answer_index)}",
        "explanation": preprocess(entry["Explanation"]),
        "subset_for_metrics": entry["Subdomain"],
        "difficulty": (
            re.split(r"\s*\(", entry["Writer's Difficulty Estimate"])[0]
            if entry["Writer's Difficulty Estimate"] is not None
            else None
        ),
        **get_mcq_fields(entry["Question"], choices),
    }


def write_data_to_file(output_file, data):
    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in tqdm(data, desc=f"Writing {output_file.name}"):
            json.dump(format_entry(entry), fout)
            fout.write("\n")


def save_data(split, random_seed) -> dict:
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)

    output_file = data_dir / f"{split}.jsonl"
    random.seed(random_seed)

    contract = SOURCE_FILES[split]
    source_path = Path(
        hf_hub_download(
            repo_id=SOURCE_REPOSITORY,
            filename=contract["filename"],
            repo_type="dataset",
            revision=SOURCE_REVISION,
        )
    )
    source_metadata = _validate_source_csv(source_path, split)
    dataset = load_dataset(
        SOURCE_REPOSITORY,
        f"gpqa_{split}",
        revision=SOURCE_REVISION,
    )["train"]
    if len(dataset) != contract["rows"]:
        raise RuntimeError(
            f"GPQA {split} loader returned {len(dataset)} rows; expected {contract['rows']}"
        )
    write_data_to_file(output_file, dataset)
    source_metadata["prepared_sha256"] = hashlib.sha256(output_file.read_bytes()).hexdigest()
    return source_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="all",
        choices=("all", "extended", "main", "diamond"),
        help="Dataset split to process.",
    )
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    splits = ["extended", "main", "diamond"] if args.split == "all" else [args.split]
    prepared = {split: save_data(split, args.random_seed) for split in splits}
    provenance = {
        "benchmark": "GPQA",
        "dataset": SOURCE_REPOSITORY,
        "dataset_revision": SOURCE_REVISION,
        "preprocessing": (
            "EleutherAI lm-evaluation-harness GPQA generative preprocessing; "
            "choice order shuffled deterministically"
        ),
        "random_seed": args.random_seed,
        "splits": prepared,
    }
    (Path(__file__).absolute().parent / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
