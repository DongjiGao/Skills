# Official MUCS scoring assets

These files are copied byte-for-byte from `inference/` in the official
[OpenSLR SLR104](https://www.openslr.org/104/) archive
`subtask2_blindtest_wReadme.tar.gz`:

- URL: `https://www.openslr.org/resources/104/subtask2_blindtest_wReadme.tar.gz`
- Archive SHA-256: `b134ba007e6a310a2ca2a89619257ea6e7de744e135043c8c517cc05a404889a`
- Archive bytes: `1,008,263,116`

| Packaged name | Source name | Rows | SHA-256 |
|---|---|---:|---|
| `punct.mapping.txt` | `inference/punct.mapping.txt` | 76 | `43927a9aea99289d552bdc509406680688f67e25d52fb6f9debcd27c1e31b1ef` |
| `hin2engwords.txt` | `inference/hin2engwords.txt` | 1,315 | `af49f1a95114021e678a0e86c7a9c4e85c8b91072ebd538d76293e73b8b5968e` |
| `ben2engwords.txt` | `inference/ben2engwords.txt` | 1,177 | `8e53663a8b839e042eb9e430877fc9f783efb056d169de2e4c8d7b1fe28e5b80` |
| `official_compute_wer.py` | `inference/compute_wer.py` | 99 | `fc22eefa193f23dfee0b23d4078233cc9e3cb2c95690ccf48f39c5dfe8d388b6` |

`prepare.py` checks both the packaged checksums and byte equality against the
pinned archive before preparing data. `official_compute_wer.py` is retained as
protocol evidence; runtime scoring uses `../scorer.py`, which produces additive
corpus counts and avoids dependency on a historical `jiwer` version.
