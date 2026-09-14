# LoGenBench

LoGenBench is a regression benchmark for length-terminated audio generation.
Its frozen selection contains all 1,162 audio-bearing request observations that
ended with `finish_reason=length` across the accepted HR9b step-4000 evaluation
artifacts audited on 2026-09-14. The 494 text-only observations were excluded.

The manifest preserves duplicate waveforms when they appeared under distinct
prompt or task conditions. Each row retains its original generation ceiling.
Recordings longer than 55 minutes are chunked, and a logical example is counted
as unbounded when any chunk finishes by length.

The headline metric is `unbounded_generation` (percent). Metrics are also
reported for `short-form` (<60 seconds), `mid-form` (60 to <600 seconds), and
`long-form` (>=600 seconds) through `subset_for_metrics`.

Prepare on DFW with:

```bash
python -m nemo_skills.dataset.logenbench.prepare --output-dir /lustre/.../immutable-logenbench
```

Preparation resolves `/data` and `/dataset` aliases plus symlinks to absolute,
existing `/lustre/...` regular files and writes a SHA-256 receipt.
