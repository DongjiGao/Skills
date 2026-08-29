# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Prepare Numb3rs dataset for TN/ITN evaluation.

Numb3rs is a speech dataset for text normalization (TN) and inverse text normalization (ITN) tasks,
containing paired written/spoken forms with corresponding synthetic audio.

Dataset: https://huggingface.co/datasets/nvidia/Numb3rs

Output structure:
- categories/{category}_neutral.jsonl: Per-category files with neutral prompt
- categories/{category}_tn.jsonl: Per-category files with TN prompt (written form)
- categories/{category}_itn.jsonl: Per-category files with ITN prompt (spoken form)
- test_neutral.jsonl: Combined file with neutral prompt (all categories)
- test_tn.jsonl: Combined file with TN prompt (all categories)
- test_itn.jsonl: Combined file with ITN prompt (all categories)

Usage:
    ns prepare_data numb3rs --no-audio
    ns prepare_data numb3rs --categories CARDINAL DATE MONEY --no-audio
    ns prepare_data numb3rs --no-audio --audio-prefix /data/numb3rs

Audio path in JSONL files: {audio_prefix}/Numb3rs/{CATEGORY}/{name}.wav
Default audio_prefix: /data/numb3rs (maps to container mount)
"""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm

HF_REPO_ID = "nvidia/Numb3rs"
HF_REVISION = "192908075e1bd293914cc7b508f4a183ba6ef2b8"
MANIFEST_FILENAME = "manifest.jsonl"
EXPECTED_MANIFEST_ROWS = 10131

SYSTEM_MESSAGE = "You are a helpful assistant. /no_think"

# Prompt variants for TN/ITN evaluation
PROMPT_NEUTRAL = "Transcribe the audio file into English text."
PROMPT_TN = "Transcribe the audio into written form with numbers as digits (e.g., '$100', '3.14', 'Jan 1')."
PROMPT_ITN = "Transcribe the audio into spoken form with numbers spelled out (e.g., 'one hundred dollars', 'three point one four', 'january first')."

MIN_AUDIO_DURATION = 0.1  # Skip audio shorter than this

# Prompt variant definitions
PROMPT_VARIANTS = {
    "neutral": PROMPT_NEUTRAL,
    "tn": PROMPT_TN,
    "itn": PROMPT_ITN,
}


def build_messages_with_prompt(audio_metadata, prompt_text):
    """Build OpenAI format messages with a specific prompt and audio."""
    system_message = {"role": "system", "content": SYSTEM_MESSAGE}
    user_message = {
        "role": "user",
        "content": prompt_text,
        "audio": audio_metadata,
    }
    return [system_message, user_message]


def _relative_audio_path(entry):
    raw_path = entry.get("audio_filepath")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("Numb3rs manifest row has no audio_filepath")
    relative_path = Path(raw_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Numb3rs audio path must be repository-relative: {raw_path!r}")
    if len(relative_path.parts) != 2 or relative_path.suffix.lower() != ".wav":
        raise ValueError(f"Unexpected Numb3rs audio path: {raw_path!r}")
    return relative_path


def load_official_manifest():
    """Load the pinned, transcript-bearing manifest rather than AudioFolder rows.

    The Hub dataset's automatic AudioFolder view contains only ``audio`` and
    ``label``.  The official manifest is the source of written/spoken references
    and identifies the 10,131 human-verified examples described by the card.
    """
    manifest_path = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MANIFEST_FILENAME,
            repo_type="dataset",
            revision=HF_REVISION,
        )
    )
    with manifest_path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]

    if len(rows) != EXPECTED_MANIFEST_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_MANIFEST_ROWS} rows in pinned Numb3rs manifest, "
            f"found {len(rows)}"
        )

    seen_audio_paths = set()
    required_fields = {"audio_filepath", "duration", "info", "original_text", "text"}
    for row_index, row in enumerate(rows):
        missing = required_fields - row.keys()
        if missing:
            raise ValueError(
                f"Numb3rs manifest row {row_index} is missing fields: {sorted(missing)}"
            )
        relative_path = _relative_audio_path(row)
        if relative_path in seen_audio_paths:
            raise ValueError(f"Duplicate Numb3rs audio path: {relative_path}")
        seen_audio_paths.add(relative_path)
    return rows


def download_manifest_audio(rows):
    """Download exactly the WAVs referenced by the pinned official manifest."""
    allow_patterns = sorted(str(_relative_audio_path(row)) for row in rows)
    return Path(
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            revision=HF_REVISION,
            allow_patterns=allow_patterns,
        )
    )


def save_audio_and_format_entry(
    entry,
    category,
    audio_dir,
    sample_idx,
    with_audio=True,
    audio_prefix="/data/numb3rs",
    source_root=None,
):
    """Format a dataset entry and optionally save audio file.

    Returns a base entry dict with audio metadata. Messages are added separately
    based on prompt variant when writing to files.

    Args:
        audio_prefix: Prefix for audio paths in the generated JSONL files.
                      Default is '/data/numb3rs' which maps to container mount.
                      Set to your local audio storage path as needed.
    """
    # Extract fields from Numb3rs dataset
    # original_text = written form (TN), text = spoken form (ITN)
    original_text = entry["original_text"].strip()
    text = entry["text"].strip()
    category = category.upper()
    if entry["info"].upper() != category:
        raise ValueError(
            f"Numb3rs category mismatch: requested {category}, row has {entry['info']}"
        )

    relative_audio_path = _relative_audio_path(entry)
    if relative_audio_path.parts[0].upper() != category:
        raise ValueError(
            f"Numb3rs category/path mismatch: {category} vs {relative_audio_path}"
        )
    audio_filename = relative_audio_path.name
    sample_id = relative_audio_path.stem

    duration = float(entry["duration"])
    if duration < MIN_AUDIO_DURATION:
        return None

    if with_audio:
        if source_root is None:
            raise ValueError("source_root is required when saving Numb3rs audio")
        source_path = Path(source_root) / relative_audio_path
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        shutil.copy2(source_path, audio_dir / audio_filename)

    audio_filepath = f"{audio_prefix}/Numb3rs/{category}/{audio_filename}"

    # Build audio metadata (to be embedded in messages later)
    audio_metadata = {
        "path": audio_filepath,
        "duration": float(duration),
    }

    # Return base entry without messages (messages will be added per variant)
    formatted_entry = {
        "audio_filepath": audio_filepath,
        "duration": float(duration),
        "text_tn": original_text,
        "text_itn": text,
        "task_type": "ASR_LEADERBOARD",
        "category": category,
        "sample_id": sample_id,
        "subset_for_metrics": f"numb3rs_{category}",
        "extra_fields": {
            "source_repo": HF_REPO_ID,
            "source_revision": HF_REVISION,
            "source_manifest": MANIFEST_FILENAME,
        },
        "audio_metadata": audio_metadata,  # Store for message building
    }

    return formatted_entry


def prepare_category(
    category, dataset, output_dir, with_audio=True, audio_prefix="/data/numb3rs", source_root=None
):
    """Prepare a single category from the Numb3rs dataset.

    Generates 3 files per category in categories/ subfolder:
    - categories/{category}_neutral.jsonl
    - categories/{category}_tn.jsonl
    - categories/{category}_itn.jsonl

    Args:
        audio_prefix: Prefix for audio paths in the generated JSONL files.
    """
    print(f"\nProcessing category: {category}")

    # Filter dataset by category
    category_samples = [s for s in dataset if s["info"].upper() == category.upper()]

    if not category_samples:
        print(f"No samples found for category: {category}")
        return 0

    print(f"Found {len(category_samples)} samples")

    # Create output directories
    audio_dir = output_dir / "Numb3rs" / category  # Match cluster structure
    category_dir = output_dir / "categories"  # Save category files in subfolder

    if with_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving audio files to {audio_dir}")

    category_dir.mkdir(parents=True, exist_ok=True)

    # Process samples into base entries (without messages)
    base_entries = []
    skipped = 0

    for idx, entry in enumerate(tqdm(category_samples, desc=f"Processing {category}")):
        formatted = save_audio_and_format_entry(
            entry,
            category,
            audio_dir,
            idx,
            with_audio=with_audio,
            audio_prefix=audio_prefix,
            source_root=source_root,
        )
        if formatted is None:
            skipped += 1
            continue
        base_entries.append(formatted)

    if skipped > 0:
        print(f"Skipped {skipped} samples (short audio or invalid)")

    # Write 3 variant files with different prompts
    variant_counts = {}
    for variant_name, prompt_text in PROMPT_VARIANTS.items():
        output_file = category_dir / f"{category}_{variant_name}.jsonl"
        count = 0

        with open(output_file, "w", encoding="utf-8") as fout:
            for base_entry in base_entries:
                # Build complete entry with messages for this variant
                entry_with_messages = base_entry.copy()

                # Set expected_answer based on variant
                if variant_name == "tn":
                    entry_with_messages["expected_answer"] = base_entry["text_tn"]
                else:  # neutral and itn both expect spoken form
                    entry_with_messages["expected_answer"] = base_entry["text_itn"]

                # Build messages with the prompt for this variant
                entry_with_messages["messages"] = build_messages_with_prompt(base_entry["audio_metadata"], prompt_text)

                # Remove audio_metadata (only needed for message building)
                del entry_with_messages["audio_metadata"]

                fout.write(json.dumps(entry_with_messages) + "\n")
                count += 1

        variant_counts[variant_name] = count
        print(f"  Saved {count} samples to {output_file}")

    return len(base_entries)


def main():
    parser = argparse.ArgumentParser(description="Prepare Numb3rs dataset for TN/ITN evaluation")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["all"],
        help="Categories to prepare (default: all). Available: ADDRESS, CARDINAL, DATE, DECIMAL, DIGIT, FRACTION, MEASURE, MONEY, ORDINAL, PLAIN, TELEPHONE, TIME",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip saving audio files (JSONL still includes audio paths)",
    )
    parser.add_argument(
        "--audio-prefix",
        default="/data/numb3rs",
        help="Prefix for audio paths in JSONL files (default: /data/numb3rs). "
        "Examples: /data/numb3rs, /dataset/numb3rs",
    )
    args = parser.parse_args()

    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with_audio = not args.no_audio
    audio_prefix = args.audio_prefix.rstrip("/")  # Remove trailing slash if present

    print(f"Audio path prefix: {audio_prefix}")
    print(f"Audio paths in JSONL: {audio_prefix}/Numb3rs/{{CATEGORY}}/{{name}}.wav")
    if args.no_audio:
        print("Running without saving audio files.")
    else:
        print("Running with audio. Saving to Numb3rs/{category}/")

    print("\nLoading the official Numb3rs manifest from Hugging Face...")
    dataset = load_official_manifest()
    source_root = download_manifest_audio(dataset) if with_audio else None
    print(f"Loaded {len(dataset)} human-verified samples")

    # Get all available categories
    all_categories = sorted({sample["info"].upper() for sample in dataset})
    print(f"Available categories: {', '.join(all_categories)}")

    # Determine which categories to process
    if "all" in args.categories:
        categories_to_prepare = all_categories
    else:
        categories_to_prepare = [c.upper() for c in args.categories]
        # Validate categories
        invalid = set(categories_to_prepare) - set(all_categories)
        if invalid:
            invalid_str = ", ".join(sorted(invalid))
            available_str = ", ".join(all_categories)
            raise ValueError(f"Unknown categories: {invalid_str}. Available categories: {available_str}")

    if not categories_to_prepare:
        print("No valid categories to process")
        return

    # Process each category
    total_samples = 0
    for category in categories_to_prepare:
        total_samples += prepare_category(
            category,
            dataset,
            output_dir,
            with_audio=with_audio,
            audio_prefix=audio_prefix,
            source_root=source_root,
        )

    # Combine all category variant files into test variant files
    print("\nCreating combined test files for each variant...")

    categories_dir = output_dir / "categories"

    for variant_name in PROMPT_VARIANTS:
        combined_file = output_dir / f"test_{variant_name}.jsonl"

        # Get all category files for this variant from categories/ subfolder
        variant_pattern = f"*_{variant_name}.jsonl"
        category_files = sorted(categories_dir.glob(variant_pattern)) if categories_dir.exists() else []

        combined_count = 0
        with open(combined_file, "w", encoding="utf-8") as fout:
            for category_file in category_files:
                with open(category_file, encoding="utf-8") as fin:
                    for line in fin:
                        fout.write(line)
                        combined_count += 1

        print(f"  {combined_file.name}: Combined {combined_count} samples from {len(category_files)} categories")

    print(f"\nTotal: {total_samples} samples prepared across {len(PROMPT_VARIANTS)} prompt variants")


if __name__ == "__main__":
    main()
