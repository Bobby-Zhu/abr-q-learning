# Capture-disjoint trace splits

These manifests use paths relative to `data/`. Generate them with
`python scripts/create_splits.py` and verify the committed files with
`python scripts/create_splits.py --check`.

| Split | Traces | FCC traces | Captures |
| --- | ---: | ---: | ---: |
| Train | 166 | 39 | 166 |
| Validation | 130 | 44 | 19 |
| Test | 132 | 44 | 20 |

The builder groups FCC files by capture ID and groups other files by their
recording names after stripping clip suffixes. It fingerprints timestamps
relative to the start of each trace and checks that each trace can deliver
data. Duplicate content is retained once, favoring upstream training files,
and links the capture groups of all copies.

Groups touching the upstream training set go to train. Remaining groups are
allocated within FCC and non-FCC strata. The builder places larger groups
first and breaks size ties with a seed-42 hash. Each group goes to the
smaller split in its stratum, choosing validation on a tie. The
[capture audit](capture_audit.json) records each file's group, fingerprint,
assignment and any exclusion. All 428 traces were retained in this run.
