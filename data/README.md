# Imported ABR data

This directory contains the minimal upstream data needed by the project:

- `video_sizes/mao`: six files with 49 encoded chunk sizes each, from MERINA.
- `traces/fcc_hsdpa/train`: the authors' 166 mixed training traces.
- `traces/fcc_hsdpa/holdout`: the authors' 262 mixed holdout traces.

The trace format is `timestamp_seconds bandwidth_Mbps`, separated by
whitespace. Do not modify the imported files in place. Any future split
manifests or processed data should be stored separately so provenance remains
clear.

The [committed manifests](splits/README.md) define a capture-disjoint
166/130/132 train/validation/test partition. Imported files remain in their
original locations.

Sources, exact commits, and licenses are listed in
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
