# CTC timestamp evaluation

`ctc-timestamp-eval` evaluates word start/end timestamps independently of model-specific timestamp tokens. The
prepared manifests contain the audio path, a merged CTM, and the same word intervals inline as `reference_words`.

## Aliases

| Alias | Selection | Speakers | Samples |
| --- | --- | ---: | ---: |
| `ctc-timestamp-eval.ch109` | Fixed random seed `20260920` | 2 | 10 |
| `ctc-timestamp-eval.ami-ihm-test` | Fixed random seed `20260920` | multiple | 4 |
| `ctc-timestamp-eval.lsm` | Seeded 5% sample, balanced across test-clean 1mix/2mix/3mix | 1–3 | 393 |
| `ctc-timestamp-eval.d12a-{ar,de,en,es,fr,hi,it,ja,ko,pt,ru,th,tr,vi,zh}` | Language-mapped, word-aligned rows from the full test manifest | 1 | 7,951 total |

The exact selections and aligned-LibriSpeech source commit are pinned in `selected_files.json`.

CHAES channel CTMs (`c1` and `c2`) are merged by word start time. CHAES and AMI timestamps are shifted from the
original session clock to the prepared full-session audio clock.

LSM contains 131 samples from each of `lsm-test-clean-{1,2,3}mix`, selected with seed `20260920`. The supplied
LibriSpeechMix CTM directory contains only `train-960` alignments, so test-clean word boundaries are reconstructed
from the `words` tier of the pinned aligned-LibriSpeech TextGrids using the mixture delays; `<eps>` intervals are
excluded. TextGrid tokens are reconciled to the exact LibriSpeechMix transcripts before the per-speaker intervals
are shifted into the mixed-audio timeline.

## Prepare data

```bash
python nemo_skills/dataset/ctc-timestamp-eval/prepare.py \
  --data-dir=/path/to/skills_data/ctc-timestamp-eval \
  --chaes-audio-root=/path/to/full-chaes-ch109-test \
  --chaes-ctm-root=/path/to/CHAES/wav2vec_fa_ctms \
  --ami-audio-root=/path/to/full-ami-ihm-test \
  --ami-ctm-root=/path/to/AMIcorpus/ctms \
  --librispeech-textgrid-root=/path/to/aligned-librispeech/librispeech_aligned \
  --lsm-skills-root=/path/to/skills_data/multispeaker-sot \
  --lsm-metadata-root=/path/to/LibriSpeechMix/nemo_manifests/list \
  --d12-aligned-manifest=/path/to/whole/d12a_test.json \
  --d12-label-root=/path/to/d12a_info/nemo_labels
```

D12a rows are split into aliases by matching each aligned row's parent-session UUID against the per-language
manifest. Preparation extracts the aligned segment from the source audio path in each row with `ffmpeg`. If the
same segments were already materialized with names such as
`test.d12a-00000-<source-stem>.wav`, pass `--d12-segment-audio-root` to hard-link or copy them instead.

The resulting closure is:

```text
ctc-timestamp-eval/
├── audio/<alias>/*.{wav,opus}
├── ground_truth/<alias>/*.ctm
├── <alias>/test.jsonl
└── selected_files.json
```

The paths written into manifests default to `/data/ctc-timestamp-eval/...` so the data root can be mounted at
`/data` in an evaluation container.

## Prediction schema and scoring

Predictions may store timestamps at `speaker_word_timestamps`, `timestamps.speaker_word_timestamps`,
`ctc_timestamps.speaker_word_timestamps`, `word_timestamps`, or `words`. A speaker mapping contains lists such as:

```json
{
  "sample_id": "EN2002a.Mix-Headset_0.96_2141.41_0",
  "timestamps": {
    "speaker_word_timestamps": {
      "speaker0": [{"word": "hello", "start": 1.04, "end": 1.28}]
    }
  }
}
```

Run the standalone evaluator with:

```bash
python nemo_skills/dataset/ctc-timestamp-eval/score_ctc_timestamps.py \
  --reference=/data/ctc-timestamp-eval/ch109/test.jsonl \
  --predictions=/path/to/output.jsonl \
  --scored-output=/path/to/scored.jsonl \
  --metrics-output=/path/to/metrics.json
```

Speaker labels are permutation-invariant. The scorer first chooses the speaker permutation with minimum word edit
distance, then uses deterministic Levenshtein backtracing to pair exact words. It reports speaker-permuted WER,
start/end mean absolute error, and timestamp precision/recall/F1. It reports F1 independently at 120, 240, and
360 ms and `timestamp_f1_average`, the unweighted mean of those three F1 scores. A timestamp true positive requires
both word boundaries to be within the corresponding tolerance. The unsuffixed precision/recall/F1 fields retain the
240 ms default for compatibility; override those fields with `--tolerance-seconds` when using the standalone scorer.

## D12a language coverage

`prepare.d12_reference_words()` implements the requested end-time rule: each word ends at the following word's
start, and the final word ends at the segment boundary. Fifteen language manifests intersect the full aligned
test split, producing these counts: `ar` 409, `de` 520, `en` 280, `es` 273, `fr` 248, `hi` 676, `it` 67, `ja`
1,225, `ko` 416, `pt` 948, `ru` 361, `th` 738, `tr` 842, `vi` 358, and `zh` 590. Event-only rows
whose alignment contains only the `[Timestamp alignment GT labels]` sentinel are excluded because they have no
word timestamp to evaluate.
