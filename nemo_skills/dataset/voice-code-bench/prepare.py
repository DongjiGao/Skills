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

"""Prepare the official VoiceCodeBench test set for NeMo Skills.

Protocol invariants:
* the model receives audio plus a generic transcription instruction only;
* target entities, transcript layers, domain, and difficulty remain scorer-only;
* the exact dataset and scorer revision is pinned;
* hypotheses are treated as text and are never executed.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

HF_REPO_ID = "besimple-ai/voice-code-bench"
OFFICIAL_REVISION = "af309592118b83fdf9b0ad896b564a142657aac8"
BENCHMARK_NAME = "voice-code-bench"
EXPECTED_ROWS = 300
EXPECTED_ENTITIES = 1482
SYSTEM_PROMPT = "You are a helpful assistant. /no_think"
USER_PROMPT = "Produce a verbatim transcript of the audio."


def _default_data_dir() -> Path:
    root = os.environ.get("NEMO_SKILLS_DATA_DIR")
    if root:
        return Path(root) / BENCHMARK_NAME
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent.parent / f"{parent.name}-data" / BENCHMARK_NAME
    return Path.home() / ".cache" / "nemo-skills-data" / BENCHMARK_NAME


def _download_official_snapshot(data_dir: Path, revision: str) -> None:
    from huggingface_hub import snapshot_download

    data_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        revision=revision,
        local_dir=str(data_dir),
        allow_patterns=[
            "LICENSE",
            "DATASET_CARD.md",
            "README.md",
            "pyproject.toml",
            "data/metadata.jsonl",
            "data/audio/*.wav",
            "scripts/voice_code_bench/**",
        ],
    )


def _read_metadata(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} must contain a JSON object")
            rows.append(row)
    return rows


def _validate_rows(rows: list[dict[str, Any]], data_dir: Path) -> None:
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} VoiceCodeBench rows, found {len(rows)}")
    audio_ids: set[str] = set()
    entity_total = 0
    for index, row in enumerate(rows, start=1):
        audio_id = row.get("audio_id")
        if not isinstance(audio_id, str) or not audio_id:
            raise ValueError(f"VoiceCodeBench row {index} is missing audio_id")
        if audio_id in audio_ids:
            raise ValueError(f"Duplicate VoiceCodeBench audio_id: {audio_id}")
        audio_ids.add(audio_id)
        transcripts = row.get("transcripts")
        if not isinstance(transcripts, dict) or any(
            not isinstance(transcripts.get(name), str) or not transcripts[name]
            for name in ("template", "acoustic", "canonical")
        ):
            raise ValueError(f"VoiceCodeBench row {audio_id} has invalid transcript layers")
        entities = row.get("entities")
        if not isinstance(entities, list) or len(entities) != row.get("entity_count"):
            raise ValueError(f"VoiceCodeBench row {audio_id} has inconsistent entities")
        entity_total += len(entities)
        file_name = row.get("file_name")
        if not isinstance(file_name, str) or not (data_dir / "data" / file_name).is_file():
            raise FileNotFoundError(f"Missing VoiceCodeBench audio for {audio_id}: {file_name}")
    if entity_total != EXPECTED_ENTITIES:
        raise ValueError(f"Expected {EXPECTED_ENTITIES} target entities, found {entity_total}")


def _format_entry(row: dict[str, Any], audio_prefix: str) -> dict[str, Any]:
    file_name = row["file_name"]
    audio_path = f"{audio_prefix.rstrip('/')}/data/{file_name}"
    scorer_metadata = {
        key: row[key]
        for key in (
            "audio_id",
            "domain",
            "scenario",
            "difficulty",
            "transcripts",
            "entities",
            "entity_types",
            "entity_count",
        )
    }
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT,
                "audio": {"path": audio_path, "duration": float(row["duration"])},
            }
        ],
        "expected_answer": row["transcripts"]["canonical"],
        "audio_id": row["audio_id"],
        "audio_duration": float(row["duration"]),
        "audio_filepath": audio_path,
        "subset_for_metrics": row["domain"],
        "voice_code_bench_revision": OFFICIAL_REVISION,
        "voice_code_bench": scorer_metadata,
        "task_type": "VoiceCodeBench-ASR",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare VoiceCodeBench")
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--audio-prefix", default=f"/data/{BENCHMARK_NAME}")
    parser.add_argument("--revision", default=OFFICIAL_REVISION)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    if args.revision != OFFICIAL_REVISION:
        raise ValueError(
            "VoiceCodeBench comparability requires the pinned revision "
            f"{OFFICIAL_REVISION}; received {args.revision}"
        )
    data_dir = args.data_dir or _default_data_dir()
    if not args.skip_download:
        _download_official_snapshot(data_dir, args.revision)
    metadata_path = data_dir / "data" / "metadata.jsonl"
    rows = _read_metadata(metadata_path)
    _validate_rows(rows, data_dir)

    output_path = Path(__file__).parent / "test.jsonl"
    with output_path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(_format_entry(row, args.audio_prefix), ensure_ascii=False) + "\n")
    (data_dir / "NEMO_SKILLS_REVISION").write_text(args.revision + "\n", encoding="utf-8")
    print(
        f"VoiceCodeBench: wrote {len(rows)} rows / {EXPECTED_ENTITIES} entities to {output_path}; "
        f"official revision={args.revision}"
    )


if __name__ == "__main__":
    main()
