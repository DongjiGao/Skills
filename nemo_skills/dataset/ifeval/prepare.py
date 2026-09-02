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

import hashlib
import json
import urllib.request
from pathlib import Path

SOURCE_REPOSITORY = "google-research/google-research"
SOURCE_REVISION = "5b09c22d73a9d35eb6c5d2a99b95677a45053466"
SOURCE_PATH = "instruction_following_eval/data/input_data.jsonl"
SOURCE_URL = f"https://raw.githubusercontent.com/{SOURCE_REPOSITORY}/{SOURCE_REVISION}/{SOURCE_PATH}"
SOURCE_SHA256 = "67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49"
EXPECTED_ROWS = 541
EXPECTED_INSTRUCTION_INSTANCES = 834
EXPECTED_INSTRUCTION_TYPES = 25


def _validate_source(path: Path) -> list[dict]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open("rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    unique_keys = {row["key"] for row in rows}
    instruction_ids = [
        instruction_id
        for row in rows
        for instruction_id in row["instruction_id_list"]
    ]
    observed = {
        "sha256": digest,
        "rows": len(rows),
        "unique_keys": len(unique_keys),
        "instruction_instances": len(instruction_ids),
        "instruction_types": len(set(instruction_ids)),
    }
    expected = {
        "sha256": SOURCE_SHA256,
        "rows": EXPECTED_ROWS,
        "unique_keys": EXPECTED_ROWS,
        "instruction_instances": EXPECTED_INSTRUCTION_INSTANCES,
        "instruction_types": EXPECTED_INSTRUCTION_TYPES,
    }
    if observed != expected:
        raise RuntimeError(
            f"IFEval source contract changed: observed={observed}; expected={expected}"
        )
    return rows


if __name__ == "__main__":
    data_dir = Path(__file__).absolute().parent
    original_file = str(data_dir / "input_data.jsonl")
    data_dir.mkdir(exist_ok=True)
    output_file = str(data_dir / "test.jsonl")

    urllib.request.urlretrieve(SOURCE_URL, original_file)
    original_data = _validate_source(Path(original_file))

    data = []
    for original_entry in original_data:
        new_entry = original_entry
        new_entry["question"] = original_entry["prompt"]
        data.append(new_entry)

    with open(output_file, "wt", encoding="utf-8") as fout:
        fout.writelines(json.dumps(entry) + "\n" for entry in data)

    provenance = {
        "benchmark": "IFEval",
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_path": SOURCE_PATH,
        "source_sha256": SOURCE_SHA256,
        "rows": len(data),
        "prepared_sha256": hashlib.sha256(Path(output_file).read_bytes()).hexdigest(),
        "instruction_instances": EXPECTED_INSTRUCTION_INSTANCES,
        "instruction_types": EXPECTED_INSTRUCTION_TYPES,
        "evaluator": "google-research/instruction_following_eval strict and loose",
    }
    (data_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
