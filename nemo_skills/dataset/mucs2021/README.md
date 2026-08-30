# MUCS 2021 Subtask 2 blind-test ASR

This package evaluates the public blind-test release of the Interspeech 2021
Multilingual and Code-Switching ASR Challenge (MUCS), Subtask 2:

- `mucs2021.hindi-english`: 4,034 released segments from 35 source WAVs.
- `mucs2021.bengali-english`: 3,130 released segments from 32 source WAVs.
- `mucs2021`: convenience group that runs both leaves but does **not** invent an
  overall score.

Source: [OpenSLR SLR104](https://www.openslr.org/104/), archive
`subtask2_blindtest_wReadme.tar.gz`. The recipe pins the URL, exact byte length,
SHA-256, full 88-member archive inventory, every released Kaldi metadata hash,
row and recording counts, empty-reference counts, and official scoring assets.
It slices the source recordings at the released `segments` boundaries without
resampling and writes all archive/cache/temporary/output files below the
supplied data directory.

## Preparation

```bash
ns prepare_data mucs2021 --data_dir=/path/to/skills_data
```

For an offline preparation using the exact pinned archive:

```bash
python -m nemo_skills.dataset.mucs2021.prepare \
  --data_dir=/path/to/skills_data \
  --archive=/path/to/subtask2_blindtest_wReadme.tar.gz
```

The prepared manifests are:

```text
mucs2021/hindi-english/test.jsonl
mucs2021/bengali-english/test.jsonl
```

`mucs2021/provenance.json` records source and prepared-manifest hashes, exact
contracts, durations, and the scoring protocol.

## Official scoring protocol

The release's `inference/compute_wer.py` is authoritative. This implementation
matches its operations in order:

1. Lowercase references and hypotheses and split on arbitrary whitespace.
2. Replace hypothesis tokens only with `punct.mapping.txt`; score corpus
   `wer` against the untouched lowercased/whitespace-tokenized references.
3. Starting with the punctuation-mapped hypothesis, replace hypothesis tokens
   only with `hin2engwords.txt` or `ben2engwords.txt`; score corpus `twer`
   against the same untouched references.
4. Skip an utterance entirely when its reference is empty. The release has
   eight such Hindi rows and four Bengali rows; even hypothesis insertions on
   those rows do not affect either denominator or numerator.
5. Report Hindi-English and Bengali-English independently. The release does
   not specify an official cross-language aggregate.

Both `wer` (Normal WER) and `twer` (Transliterated WER) are percentages, rounded
to two decimal places like the released script. Additive error and reference
word counts are retained in the metrics JSON for auditing.

The three mapping files and the released scorer are vendored byte-for-byte
under `assets/`; `assets/README.md` records their source and checksums. The
runtime implementation avoids a dependency on a historical `jiwer` version,
but uses equivalent word-level Levenshtein distance and is regression-tested
against the released transformations and empty-reference behavior.
