from nemo_skills.dataset.logenbench.metrics import LoGenBenchMetrics


def test_logenbench_counts_any_length_terminated_chunk():
    metrics = LoGenBenchMetrics()
    metrics.update([{"finish_reason": "stop", "num_generated_tokens": 2}])
    metrics.update(
        [
            {
                "finish_reason": "stop",
                "chunk_finish_reasons": ["length", "stop"],
                "num_generated_tokens": 10,
            }
        ]
    )
    result = metrics.get_metrics()["pass@1"]
    assert result["num_entries"] == 2
    assert result["unbounded_examples"] == 1
    assert result["unbounded_generation"] == 50.0
