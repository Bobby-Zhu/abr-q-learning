"""Compare committed train/validation/test trace manifests before and after the split fix.

Run with Python 3.10+ from the abr-q-learning repo or pass --repo PATH.
The archived validation manifest in evidence/ supports shallow checkouts.
"""

import argparse
from collections import defaultdict
from decimal import Decimal
import hashlib
from itertools import combinations
from pathlib import Path, PurePosixPath
import re
import subprocess


OLD_COMMIT = "4baff74e534bfb7e7c63360781a0071819ee1824"
OLD_BLOBS = {
    "train": "707aea08f33dc4e67093f650fd8364a00e4bdcb4",
    "validation": "8235bd6a6591abc6402ad3bbb0271fc2746fb467",
    "test": "d3d6ac64731b63fe00b3cb2a28fca22fc2bf9d9b",
}
WINDOW = 16


def entries(text):
    paths = [s.strip() for s in text.splitlines() if s.strip() and not s.startswith("#")]
    if len(paths) != len(set(paths)):
        raise ValueError("a manifest contains a repeated path")
    return paths


def read_historical(repo, current):
    try:
        old = {
            split: entries(subprocess.check_output(
                ["git", "-C", str(repo), "show", f"{OLD_COMMIT}:data/splits/{split}.txt"],
                text=True, encoding="utf-8", stderr=subprocess.DEVNULL,
            ))
            for split in OLD_BLOBS
        }
        source = "Git history"
    except (FileNotFoundError, subprocess.CalledProcessError):
        archive = Path(__file__).parent / "evidence" / "validation_4baff74e.txt"
        val = entries(archive.read_text(encoding="utf-8"))
        holdout = set(current["validation"]) | set(current["test"])
        old = {"train": current["train"], "validation": val, "test": sorted(holdout - set(val))}
        source = "archived validation + verified holdout complement"

    for split, paths in old.items():
        data = ("\n".join(paths) + "\n").encode()
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if blob != OLD_BLOBS[split]:
            raise ValueError(f"historical {split} manifest does not match {OLD_COMMIT}")
    return old, source


def recording_id(path):
    # FCC filenames may name a website after the numeric capture ID.
    name = PurePosixPath(path).name.removesuffix(".log")
    stem = re.sub(r"_part\d+$", "", name)
    match = re.match(r"trace_(\d+)_", stem)
    return "fcc_" + match[1] if match else stem


def fingerprints(root, paths):
    # Intern exact Decimal values once. Equal values with different textual
    # precision get the same integer, without rounding or hash collisions.
    values = {}
    whole = defaultdict(set)
    windows = defaultdict(set)
    for path in sorted(paths):
        vector = []
        for line in (root / path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                _, bandwidth = line.split()
                value = Decimal(bandwidth)
                if not value.is_finite():
                    raise ValueError(f"non-finite bandwidth in {path}")
                vector.append(values.setdefault(value, len(values)))
        if not vector:
            raise ValueError(f"empty trace: {path}")
        vector = tuple(vector)
        whole[vector].add(path)
        for pos in range(len(vector) - WINDOW + 1):
            window = vector[pos:pos + WINDOW]
            if any(value != window[0] for value in window[1:]):
                windows[window].add(path)
    # Only fingerprints present in multiple files can match across splits.
    return [[group for group in index.values() if len(group) > 1] for index in (whole, windows)]


def matching_pairs(groups, left, right):
    return {
        (first, second)
        for group in groups
        for first in group & left
        for second in group & right
    }


def report(label, manifests, whole, windows):
    print(f"\n{label}")
    for split, paths in manifests.items():
        print(f"  {split}: {len(paths)} files, {len({recording_id(p) for p in paths})} recording IDs")
    print("  split pair             shared IDs   right files with sibling   full pairs   window pairs")
    for a, b in combinations(("train", "validation", "test"), 2):
        left, right = set(manifests[a]), set(manifests[b])
        shared = left & right
        if shared:
            raise ValueError(f"identical paths present in {a} and {b}: {sorted(shared)}")
        ids = {recording_id(p) for p in left} & {recording_id(p) for p in right}
        siblings = sum(recording_id(p) in ids for p in right)
        exact_pairs = matching_pairs(whole, left, right)
        window_pairs = matching_pairs(windows, left, right)
        print(f"  {a + '/' + b:22} {len(ids):10} {siblings:26} {len(exact_pairs):12} {len(window_pairs):14}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="abr-q-learning checkout")
    args = parser.parse_args()
    data = args.repo / "data"
    current = {
        split: entries((data / "splits" / f"{split}.txt").read_text(encoding="utf-8"))
        for split in OLD_BLOBS
    }
    old, source = read_historical(args.repo, current)
    paths = {p for split in (current, old) for files in split.values() for p in files}
    whole, windows = fingerprints(data, paths)
    print(f"Historical manifests: {source}. All three Git blob IDs verified.")
    print(f"Numeric matches count cross-split file pairs. Windows use {WINDOW} consecutive")
    print("bandwidth samples with at least two distinct values, at any offset, ignoring timestamps.")
    report(f"Before grouping ({OLD_COMMIT[:8]})", old, whole, windows)
    report("Current working-tree manifests", current, whole, windows)


if __name__ == "__main__":
    main()
