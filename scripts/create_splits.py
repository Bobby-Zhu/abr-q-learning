#!/usr/bin/env python3
"""Create or verify deterministic train/validation/test trace manifests."""

import argparse
import hashlib
import re
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
TRACE_ROOT = DATA_ROOT / "traces/fcc_hsdpa"
SPLIT_ROOT = DATA_ROOT / "splits"
SEED = 42
PART_SUFFIX = re.compile(r"_part\d+\.log$")


def recording_id(name: str) -> str:
    """Return the source recording of a trace file.

    Upstream holdout files are consecutive slices of longer recordings named
    ``<recording>_part<N>.log``. Slices of one recording must stay in one split,
    otherwise validation and test share neighbouring minutes of the same trip.
    """

    return PART_SUFFIX.sub("", name)


def stable_rank(recording: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{SEED}:{recording}".encode()).hexdigest()
    return digest, recording


def split_by_recording(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    """Assign whole recordings, in stable hash order, to the smaller side."""

    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        groups[recording_id(path.name)].append(path)

    validation: list[Path] = []
    test: list[Path] = []
    for recording in sorted(groups, key=stable_rank):
        target = validation if len(validation) <= len(test) else test
        target.extend(groups[recording])
    return validation, test


def relative_entries(paths: list[Path]) -> tuple[str, ...]:
    return tuple(str(path.relative_to(DATA_ROOT)) for path in sorted(paths))


def build_manifests() -> dict[str, tuple[str, ...]]:
    train = sorted(path for path in (TRACE_ROOT / "train").iterdir() if path.is_file())
    holdout = sorted(
        path for path in (TRACE_ROOT / "holdout").iterdir() if path.is_file()
    )

    fcc = [path for path in holdout if path.name.startswith("trace_")]
    hsdpa = [path for path in holdout if not path.name.startswith("trace_")]
    fcc_validation, fcc_test = split_by_recording(fcc)
    hsdpa_validation, hsdpa_test = split_by_recording(hsdpa)
    validation = fcc_validation + hsdpa_validation
    test = fcc_test + hsdpa_test

    recordings = {
        name: {recording_id(path.name) for path in paths}
        for name, paths in (("train", train), ("validation", validation), ("test", test))
    }
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        shared = recordings[first] & recordings[second]
        if shared:
            raise RuntimeError(
                f"recordings shared by {first} and {second}: {', '.join(sorted(shared))}"
            )

    return {
        "train": relative_entries(train),
        "validation": relative_entries(validation),
        "test": relative_entries(test),
    }


def manifest_text(entries: tuple[str, ...]) -> str:
    return "\n".join(entries) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify committed manifests instead of rewriting them",
    )
    args = parser.parse_args()

    manifests = build_manifests()
    SPLIT_ROOT.mkdir(parents=True, exist_ok=True)
    mismatches = []

    for split_name, entries in manifests.items():
        path = SPLIT_ROOT / f"{split_name}.txt"
        expected = manifest_text(entries)
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != expected:
                mismatches.append(path)
        else:
            path.write_text(expected, encoding="utf-8")
        print(f"{split_name}: {len(entries)} traces")

    if mismatches:
        names = ", ".join(str(path) for path in mismatches)
        raise SystemExit(f"split manifests are stale: {names}")


if __name__ == "__main__":
    main()
