#!/usr/bin/env python3
"""Create or verify deterministic train/validation/test trace manifests."""

import argparse
import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
TRACE_ROOT = DATA_ROOT / "traces/fcc_hsdpa"
SPLIT_ROOT = DATA_ROOT / "splits"
SEED = 42


def stable_rank(path: Path) -> tuple[str, str]:
    digest = hashlib.sha256(f"{SEED}:{path.name}".encode()).hexdigest()
    return digest, path.name


def relative_entries(paths: list[Path]) -> tuple[str, ...]:
    return tuple(str(path.relative_to(DATA_ROOT)) for path in sorted(paths))


def build_manifests() -> dict[str, tuple[str, ...]]:
    train = sorted(path for path in (TRACE_ROOT / "train").iterdir() if path.is_file())
    holdout = sorted(
        path for path in (TRACE_ROOT / "holdout").iterdir() if path.is_file()
    )

    fcc = sorted((path for path in holdout if path.name.startswith("trace_")), key=stable_rank)
    hsdpa = sorted((path for path in holdout if not path.name.startswith("trace_")), key=stable_rank)
    if len(fcc) % 2 or len(hsdpa) % 2:
        raise RuntimeError("each source group must have an even number of holdout traces")

    fcc_midpoint = len(fcc) // 2
    hsdpa_midpoint = len(hsdpa) // 2
    validation = fcc[:fcc_midpoint] + hsdpa[:hsdpa_midpoint]
    test = fcc[fcc_midpoint:] + hsdpa[hsdpa_midpoint:]

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
