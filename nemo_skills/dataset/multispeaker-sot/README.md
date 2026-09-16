# Multispeaker SOT benchmark layout

The newer full-conversation evaluations use the same parent-plus-leaves layout
as the existing duration groups. Run a parent alias for one generation job and
one summary job. Every record carries its exact leaf alias in
`subset_for_metrics`, so the parent metrics retain a separate micro cpWER for
each language, locale, or microphone view. Leaf aliases remain directly
addressable when an isolated run is needed.

## NOTSOFAR1 eval

Parent alias: `multispeaker-sot.notsofar1-eval`

| Leaf alias | View |
| --- | --- |
| `multispeaker-sot.notsofar1-eval-ihm` | Individual headset microphone |
| `multispeaker-sot.notsofar1-eval-sc` | Single-channel distant microphone |

## Turing Diarization v1

Parent alias: `multispeaker-sot.turing-diarization-v1`

There are 10 language leaves:

`ar`, `el`, `en`, `es`, `fr`, `it`, `ja`, `ko`, `pt`, and `ru`.

Each leaf is named `multispeaker-sot.turing-diarization-v1-<language>`. Turing
session names use `GR` for Greek; preparation normalizes that delivery code to
the ISO 639-1 code `el`.

## MLC-SLM 2 dev 13L

Parent alias: `multispeaker-sot.mlc-slm-2-dev-13L`

There are exactly 13 locale leaves:

`de`, `en`, `es`, `es-MX`, `fr`, `fr-CA`, `it`, `ja`, `ko`, `pt`, `pt-BR`,
`ru`, and `th`.

Each leaf is named `multispeaker-sot.mlc-slm-2-dev-13L-<locale>`. Locale
variants remain separate leaves even when they share the same base scoring
language.

## Materialized data

Under the Skills data root, every leaf has matching names across its manifest
and artifacts:

```text
multispeaker-sot/<leaf>/test.jsonl
multispeaker-sot/audio/<leaf>/...
multispeaker-sot/seglst/<leaf>/...
multispeaker-sot/rttm/<leaf>/...
```

Each parent has a merged `multispeaker-sot/<parent>/test.jsonl` built by
`build_group_manifests.py`. The aggregate `allsets/test.jsonl` merges the eight
regular parent benchmarks; it does not concatenate the language/locale leaves a
second time.
