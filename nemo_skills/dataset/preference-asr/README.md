# Preference ASR: seven-corpus release

`preference-asr.contextless` and `preference-asr.fine` implement the 5,170-row
seven-corpus source release used by HR9d. This adapter is distinct from the
2,822-row, four-corpus `preference-asr-public` release. It accepts locally
available source data; no annotations or audio are vendored here.

Both profiles score against `preference_text` with the author's preference-aware
normalization and corpus-micro WER. Case/punctuation, number rendering, and
requested fillers must survive normalization when the active preference tests
them. The original `target_case`, `direction`, and `sub_category` are retained;
the shared normalizer can recover missing case metadata from `sub_category`.
Valid empty references for disfluency removal remain in the manifest, and any
inserted words contribute to the corpus error numerator. Metrics reject legacy
ContextASR-normalized counts. Contextless and fine scores remain separate.

The versioned `preference-asr-recording-prefix-v1` prompt policy changes only:

- `Transcribe the speech in the input audio.` to `Transcribe this recording.`
- `Transcribe the speech in the input English audio` to `Transcribe this English recording.`

Every following preference instruction or entity list is preserved exactly.
`source_instruction` retains the source wording; `instruction` and the fine user
message contain the repaired wording. Contextless remains `Transcribe this
recording.`. The public instructed adapter uses the same opening repair and
records its policy, without changing its release census or references.

A step15000 HR9d paired test over the historical 5,163 prepared examples changed
1,523 openings. In that subset, empty answers fell 1,032 to zero; the 3,640
unchanged controls differed by two word errors. Overall preference-aware WER
fell 21.06% to 6.79%. This is a measured prompt workaround for premature EOS,
not proof that the upstream training issue has been repaired.

## Prepare

Provide a YAML list with `corpus`, `manifest_filepath`, and `audio_filepaths`
(the root for relative audio paths). The prior corpus YAML `paths` mapping is
also supported through `--path-key`.

```bash
python -m nemo_skills.dataset.preference-asr.prepare \
  --corpus-yaml=/lustre/data/preference_asr_bench.yaml \
  --path-key=dfw --copy-audio \
  --output-dir=/lustre/eval/new-immutable-root/preference-asr --dry-run
# Review source hashes and row counts, then remove --dry-run.
```

Dry-run checks the sources and audio paths without writes. Real preparation
refuses to replace existing profile manifests and emits `provenance.json` with
source/manifest hashes, row counts, scoring version, and prompt policy. The
complete source release should yield 5,170 rows per profile, including seven
empty references and 1,582 case targets. Use `/data` as the audio mount for the
prepared root (or set `NEMO_SKILLS_AUDIO_ROOT`).

Historical results used 5,163 rows and ContextASR normalization. New results need
fresh inference for the revised prompts and restored rows, followed by the new
scorer; merely re-summarizing old output files is insufficient. Keep historical
and repaired results in distinct namespaces with new contracts.

Existing external `Skills-N3NT-private/benchmark_map.json` entries can shadow
this built-in adapter. Remove its three `preference-asr*` entries, or point them
at this checkout's `nemo_skills/dataset/preference-asr` directories. Leave
unrelated external benchmark entries intact.
