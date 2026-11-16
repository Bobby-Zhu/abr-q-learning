# Imported ABR data

This directory contains the minimal upstream data needed by the project:

- `video_sizes/mao`: six files with 49 encoded chunk sizes each, from MERINA.
- `traces/fcc_hsdpa/train`: the authors' 166 mixed training traces.
- `traces/fcc_hsdpa/holdout`: the authors' 262 mixed holdout traces.

The trace format is `timestamp_seconds bandwidth_Mbps`, separated by
whitespace. Do not modify the imported files in place. Any future split
manifests or processed data should be stored separately so provenance remains
clear.

The committed manifests in `splits/` now define the 166/131/131
train/validation/test partition without moving or changing imported files.

Sources, exact commits, and licenses are listed in
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
