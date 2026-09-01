# NFA timestamp public reconstruction v1

This benchmark is an explicit reconstruction/proxy for the public portion of
the NFA word-timestamp evaluation. It is not an exact reproduction of the
paper's numbers.

The public protocol facts implemented here are:

- LibriSpeech `test-other`;
- Mozilla Common Voice 16.1 `test` in English, Spanish, French, and German;
- numeric timestamp tokens quantized at 80 ms;
- a timestamp true positive only when the word matches exactly and both the
  start and end error are strictly less than 240 ms;
- timestamp precision and recall, mean start/end deviation over exact word
  matches that have valid timestamp spans, and corpus WER after numeric
  timestamp tokens are stripped.

The authors did not publish CTMs, alignments, alignment configuration, or
artifact/configuration pins. This adapter therefore downloads or aligns
nothing. It requires a locally prepared artifact, validates every pin, and
writes `comparability=public-reconstruction-proxy` into both the records and
`provenance.json`. Scores from different alignment contracts are not directly
interchangeable and must not be described as paper-equivalent.

## Input artifact

Pass a JSON contract and the SHA-256 of the exact contract bytes:

```bash
ns prepare_data nfa-timestamp-reconstruction \
  --cluster=local \
  --data_dir=/path/to/skills_data \
  --artifact-contract=/path/to/artifact/contract.json \
  --artifact-contract-sha256=<sha256-of-contract.json>
```

The artifact directory contains `contract.json`, its JSONL manifest, and the
relative audio paths named by the manifest. Preparation copies validated audio
under the benchmark data directory so the normal `/data` mount is sufficient.

The contract schema is exact (unknown/missing fields fail):

```json
{
  "schema_version": "nfa-timestamp-local-artifact-v1",
  "protocol_version": "nfa-timestamp-public-reconstruction-v1",
  "manifest": "aligned.jsonl",
  "manifest_sha256": "<sha256>",
  "manifest_rows": 5,
  "sources": {
    "librispeech-test-other": {
      "dataset": "LibriSpeech",
      "release": "OpenSLR-12",
      "split": "test-other",
      "language": "en",
      "rows": 1,
      "source_manifest": "sources/librispeech-test-other.trans.txt",
      "source_manifest_sha256": "<sha256>"
    },
    "common-voice-16.1-en": {
      "dataset": "Mozilla Common Voice",
      "release": "16.1",
      "split": "test",
      "language": "en",
      "rows": 1,
      "source_manifest": "sources/common-voice-16.1-en.tsv",
      "source_manifest_sha256": "<sha256>"
    },
    "common-voice-16.1-es": {
      "dataset": "Mozilla Common Voice",
      "release": "16.1",
      "split": "test",
      "language": "es",
      "rows": 1,
      "source_manifest": "sources/common-voice-16.1-es.tsv",
      "source_manifest_sha256": "<sha256>"
    },
    "common-voice-16.1-fr": {
      "dataset": "Mozilla Common Voice",
      "release": "16.1",
      "split": "test",
      "language": "fr",
      "rows": 1,
      "source_manifest": "sources/common-voice-16.1-fr.tsv",
      "source_manifest_sha256": "<sha256>"
    },
    "common-voice-16.1-de": {
      "dataset": "Mozilla Common Voice",
      "release": "16.1",
      "split": "test",
      "language": "de",
      "rows": 1,
      "source_manifest": "sources/common-voice-16.1-de.tsv",
      "source_manifest_sha256": "<sha256>"
    }
  },
  "alignment": {
    "tool": "locally-selected-aligner",
    "revision": "<immutable-tool-revision>",
    "config": "alignment/config.yaml",
    "config_sha256": "<sha256-of-exact-alignment-config>"
  }
}
```

Each JSONL row also has an exact schema:

```json
{
  "id": "unique-id",
  "source": "librispeech-test-other",
  "audio_filepath": "audio/example.flac",
  "audio_sha256": "<sha256>",
  "audio_bytes": 1234,
  "duration": 2.4,
  "words": [
    {"word": "hello", "start": 0.08, "end": 0.4},
    {"word": "world", "start": 0.48, "end": 0.96}
  ]
}
```

Audio paths must be safe and relative. Every audio size and SHA-256 is checked.
Words must be whitespace-free and intervals must be positive, monotonic,
non-overlapping, and within the declared duration. Seconds are quantized using
decimal round-half-up at 80 ms. The source row counts, complete five-source
inventory, manifest hash, local source manifests and their hashes, aligner
revision, and the local aligner configuration plus its hash are all mandatory
and validated before any output manifest is published.

## Output and scoring grammar

The canonical reference grammar is one timestamp pair per word:

```text
<|0|>hello<|5|> <|6|>world<|12|>
```

A boundary token may also be shared (`<|0|>hello<|5|>world<|12|>`). For WER,
all numeric `<|N|>` tokens are removed before exact whitespace tokenization.
A deterministic Levenshtein alignment establishes exact word matches.
Timestamp precision uses all stripped hypothesis words as its denominator, so
plain or malformed untimestamped words cannot improve precision. Recall uses
all reference words. An error of exactly three ticks (240 ms) fails the strict
true-positive criterion. Mean start/end deviations include every exact matched
word with a valid timestamp pair, even when it misses that tolerance.

The scorer reports overall micro-aggregates and one breakdown for each of the
five public sources. Raw WER counts, reference/hypothesis word counts, true
positives, timestamped exact-word matches, untimestamped words, invalid spans,
and unparsed timestamp tokens are retained for auditability.
