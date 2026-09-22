# CTC timestamp evaluation: LSM canonical multispeaker prompt

Results: `/home/taejinp/projects/Skills_MSASR-eval/Skills-N3NT-fork/results/ctc-timestamp-eval/lsm/hr9a-13k-ntok32k-full-salm-default-prompt`

This is live end-to-end `SALMAutomodel.generate(generate_timestamps=True)` inference. The prompt is loaded from `/home/taejinp/projects/ctc-canary-dev/canary-dev/speechlm-2026h1/prompts/default.yaml` using `multispeaker-sot.lsm-test-clean-1mix, multispeaker-sot.lsm-test-clean-2mix, multispeaker-sot.lsm-test-clean-3mix`. Model loading and the maximum-batch probe are excluded from inference timing.

## Headline accuracy

| Metric | Value |
| --- | --- |
| Samples | 393 |
| Speaker-permuted WER | 12.03% |
| Timestamp F1 @120 ms | 51.99% |
| Timestamp F1 @240 ms | 78.31% |
| Timestamp F1 @360 ms | 84.47% |
| Timestamp precision / recall @240 ms | 78.43% / 78.18% |
| Start MAE | 312.58 ms |
| End MAE | 357.11 ms |
| Exact WER=0 samples | 148 |
| Exact WER=0 and all boundaries @240 ms | 42 |
| Alignment failures | 3 |

## Accuracy by mixture

| Mix | Samples | WER % | F1 @120 % | F1 @240 % | F1 @360 % | Start MAE ms | End MAE ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1mix | 131 | 2.13 | 61.80 | 90.55 | 97.24 | 54.52 | 104.38 |
| 2mix | 131 | 9.56 | 53.16 | 80.64 | 86.84 | 206.37 | 253.28 |
| 3mix | 131 | 16.52 | 48.31 | 73.18 | 79.15 | 469.74 | 510.92 |

## Accuracy by duration

| Duration | Samples | WER % | F1 @120 % | F1 @240 % | F1 @360 % | Start MAE ms | End MAE ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <5s | 75 | 13.79 | 58.07 | 82.20 | 87.17 | 65.68 | 107.23 |
| 5-10s | 111 | 10.27 | 55.64 | 81.36 | 87.32 | 113.94 | 157.89 |
| 10-20s | 151 | 13.66 | 51.90 | 77.74 | 83.32 | 272.58 | 316.80 |
| 20-30s | 47 | 11.65 | 48.87 | 76.20 | 83.09 | 521.64 | 567.27 |
| >=30s | 9 | 3.89 | 48.97 | 78.76 | 88.52 | 593.91 | 640.87 |

## Matched-word boundary distribution

| Boundary | Mean ms | P50 ms | P90 ms | P95 ms | P99 ms | Max ms | Signed bias ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Start | 312.58 | 49.40 | 147.65 | 394.76 | 8349.76 | 27668.72 | 34.46 |
| End | 357.11 | 92.06 | 269.52 | 484.07 | 8287.38 | 28378.72 | -78.27 |

### Overlap versus non-overlap (exact word matches only)

| Reference word | Matches | Start MAE ms | End MAE ms | Both @120 % | Both @240 % | Both @360 % |
| --- | --- | --- | --- | --- | --- | --- |
| Overlapped | 4584 | 812.28 | 849.51 | 48.23 | 75.28 | 82.57 |
| Non-overlapped | 9824 | 79.41 | 127.36 | 61.91 | 92.01 | 98.62 |

The references contain 5908 overlapped words across 246 samples. These percentages are conditional on exact word matches; corpus F1 above also includes substitutions, insertions, and deletions.

## Generation and validity diagnostics

| Metric | Value |
| --- | --- |
| Generated tokens | 22128 |
| Samples hitting 256-token limit | 3 |
| Timestamp speaker count exactly matches reference | 386 / 393 |
| Malformed timestamp words | 9 |
| Negative starts | 0 |
| End before start | 0 |
| Boundaries past audio end | 0 |
| Samples non-monotonic within speaker | 0 |

## Throughput and memory

| Metric | Value |
| --- | --- |
| Audio | 1.311 h |
| Measured end-to-end inference | 147.410 s |
| Timestamp generation within total | 32.946 s (22.35%) |
| End-to-end speed | 32.02x realtime |
| Samples/s | 2.67 |
| Generated tokens/s | 150.11 |
| Maximum working batch | 84 |
| Measured batches | 5 |
| GPU baseline / peak / increment | 65.73 / 83.46 / 17.73 GiB |
| Host RSS baseline / peak / increment | 6.85 / 22.05 / 15.20 GiB |

## py-spy inclusive profile highlights

| Frame | Samples | Inclusive % |
| --- | --- | --- |
| _sample (transformers/generation/utils.py:2750) | 9724 | 66.33 |
| _timestamp_with_failure_isolation (profile_salm_ctc_timestamps.py:569) | 3175 | 21.66 |
| generate_ctc_timestamps (nemo/collections/speechlm2/models/salm_automodel.py:1035) | 1601 | 10.92 |

## Eyeball artifacts

- `eyeball.html`: all samples, worst WER first, with audio, references, transcripts, and matched-word errors.
- `per_sample_stats.csv`: one row per sample.
- `word_boundary_errors.csv`: one row per exact matched word with signed boundary errors.
- `worst_30_by_wer.jsonl` and `worst_30_by_boundary_mae.jsonl`: targeted failure review.
- `examples/`: best, median, worst-WER, and worst-boundary JSON examples for each mixture.
- `detailed_stats.json`: all aggregate breakdowns, validity checks, performance, and py-spy frames.
- `batch.total.svg`: py-spy flamegraph for the measured inference pass.

## Alignment failures

- `test-clean-3mix-0115_0.00_4.95_115`: Speaker 1 transcript is too long for the CTC timeline.
- `test-clean-3mix-0305_0.00_18.05_305`: Speaker 0 transcript is too long for the CTC timeline.
- `test-clean-3mix-1860_0.00_14.15_1860`: Speaker 0 transcript is too long for the CTC timeline.
