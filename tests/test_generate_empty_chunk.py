# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

from types import SimpleNamespace

from nemo_skills.inference.generate import GenerationTask


def test_empty_chunk_materializes_output_before_inline_evaluation(tmp_path):
    output_file = tmp_path / "output_chunk_87.jsonl"
    task = GenerationTask.__new__(GenerationTask)
    task.cfg = SimpleNamespace(
        output_file=str(output_file),
        dry_run=False,
        skip_filled=True,
    )
    task.load_data = list
    task.skip_completed_samples = lambda data: data
    task.should_run_evaluation = True
    task.evaluator = None
    evaluated = []

    def run_batch_evaluation():
        assert output_file.is_file()
        assert output_file.read_text() == ""
        evaluated.append(True)

    task.run_batch_evaluation = run_batch_evaluation
    task.postprocess = lambda: None

    task.generate()

    assert evaluated == [True]
    assert output_file.is_file()
    assert output_file.read_text() == ""
