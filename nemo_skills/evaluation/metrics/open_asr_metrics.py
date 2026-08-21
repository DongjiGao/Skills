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

"""Metrics for the current Hugging Face Open ASR Leaderboard protocol."""

import logging
from collections import defaultdict

from nemo_skills.evaluation.evaluator.audio import evaluate_asr
from nemo_skills.evaluation.metrics.audio_metrics import AudioMetrics
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


class OpenASRLeaderboardMetrics(AudioMetrics):
    """Audio metrics with session-level merging for chunked leaderboard sets.

    Regular datasets are aggregated exactly like AudioMetrics. Rows carrying
    parent_id and chunk_index are buffered, ordered by chunk index, and
    collapsed to one synthetic prediction per parent session before WER is
    accumulated. This mirrors merge_chunked_manifest in
    huggingface/open_asr_leaderboard PR #202.
    """

    def reset(self):
        super().reset()
        self._chunk_groups: dict[tuple[str, str], list[list[dict]]] = defaultdict(list)
        self._chunks_finalized = False

    def update(self, predictions):
        if self._chunks_finalized:
            raise RuntimeError("Cannot update OpenASRLeaderboardMetrics after get_metrics()")

        chunk_flags = [prediction.get("parent_id") is not None for prediction in predictions]
        if not any(chunk_flags):
            return super().update(predictions)
        if not all(chunk_flags):
            raise ValueError("All generations for one sample must agree on chunk metadata")

        parent_ids = {str(prediction["parent_id"]) for prediction in predictions}
        chunk_indices = {int(prediction["chunk_index"]) for prediction in predictions}
        subsets = {str(prediction.get("subset_for_metrics", "_all_")) for prediction in predictions}
        if len(parent_ids) != 1 or len(chunk_indices) != 1 or len(subsets) != 1:
            raise ValueError("All generations for one sample must have identical chunk metadata")

        key = (subsets.pop(), parent_ids.pop())
        self._chunk_groups[key].append([prediction.copy() for prediction in predictions])

    @staticmethod
    def _sum_field(chunks: list[dict], field: str):
        values = [chunk.get(field) for chunk in chunks]
        if not any(value is not None for value in values):
            return None
        return sum(value or 0 for value in values)

    def _merge_chunk_group(self, key: tuple[str, str], chunk_prediction_sets: list[list[dict]]) -> list[dict]:
        subset, parent_id = key
        chunk_prediction_sets = sorted(
            chunk_prediction_sets,
            key=lambda prediction_set: int(prediction_set[0]["chunk_index"]),
        )
        indices = [int(prediction_set[0]["chunk_index"]) for prediction_set in chunk_prediction_sets]
        if indices != list(range(len(indices))):
            LOG.warning(
                "Chunk indices for session %s are not contiguous from 0 (%s); "
                "some chunks may be missing, which will inflate deletions.",
                parent_id,
                indices,
            )

        num_attempts = len(chunk_prediction_sets[0])
        if any(len(prediction_set) != num_attempts for prediction_set in chunk_prediction_sets):
            raise ValueError(f"Inconsistent generation counts across chunks for session {parent_id}")

        merged_predictions = []
        for attempt_index in range(num_attempts):
            chunks = [prediction_set[attempt_index] for prediction_set in chunk_prediction_sets]
            references = {str(chunk.get("expected_answer", "")).strip() for chunk in chunks}
            if len(references) != 1:
                raise ValueError(f"Inconsistent parent references across chunks for session {parent_id}")
            reference = references.pop()
            generation = " ".join(text for chunk in chunks if (text := str(chunk.get("generation") or "").strip()))

            merged = chunks[0].copy()
            for field in tuple(merged):
                if (
                    field == "wer"
                    or field.startswith("wer_")
                    or field
                    in {
                        "is_correct",
                        "predicted_answer",
                        "text",
                        "pred_text",
                    }
                ):
                    merged.pop(field, None)

            merged["id"] = parent_id
            merged["parent_id"] = parent_id
            merged["subset_for_metrics"] = subset
            merged["expected_answer"] = reference
            merged["generation"] = generation
            merged["predicted_answer"] = generation
            merged.update(evaluate_asr(reference, generation, normalization_mode="hf_leaderboard"))

            for field in ("num_generated_tokens", "num_reasoning_tokens", "num_answer_tokens", "audio_duration"):
                value = self._sum_field(chunks, field)
                if value is not None:
                    merged[field] = value

            start_times = [chunk["generation_start_time"] for chunk in chunks if "generation_start_time" in chunk]
            end_times = [chunk["generation_end_time"] for chunk in chunks if "generation_end_time" in chunk]
            if start_times:
                merged["generation_start_time"] = min(start_times)
            if end_times:
                merged["generation_end_time"] = max(end_times)

            merged_predictions.append(merged)

        return merged_predictions

    def _finalize_chunk_groups(self):
        if self._chunks_finalized:
            return
        for key, chunk_prediction_sets in sorted(self._chunk_groups.items()):
            super().update(self._merge_chunk_group(key, chunk_prediction_sets))
        self._chunks_finalized = True

    def get_metrics(self):
        self._finalize_chunk_groups()
        return super().get_metrics()
