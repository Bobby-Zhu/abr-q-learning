"""Paired recording bootstrap of Q-learning minus buffer QoE on the frozen test set."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import random
import re
from statistics import mean


def interval(groups):
    # Preserve the per-slice estimand when a resampled recording has more slices.
    blocks = [(sum(groups[name]), len(groups[name])) for name in sorted(groups)]
    rng = random.Random(0)
    estimates = []
    for _ in range(10_000):
        sample = rng.choices(blocks, k=len(blocks))
        estimates.append(sum(total for total, _ in sample) / sum(n for _, n in sample))
    estimates.sort()
    # Nearest-rank empirical percentiles, indexed from zero.
    return estimates[249], estimates[9749]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    args = parser.parse_args()
    paired = defaultdict(dict)
    with (args.repo / "results/final/test/per_trace.csv").open(newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            if row["policy"] in {"buffer", "q_learning"}:
                key = (row["source"], row["trace_name"])
                run = row["seed"] if row["policy"] == "q_learning" else "buffer"
                if run in paired[key]:
                    raise ValueError(f"Duplicate row for {key}, {run}")
                paired[key][run] = float(row["total_qoe"])
    groups = defaultdict(lambda: defaultdict(list))
    expected_runs = {"buffer", "100", "101", "102", "103", "104"}
    for (source, trace), runs in paired.items():
        if set(runs) != expected_runs:
            raise ValueError(f"Incomplete comparison for {trace}")
        diff = mean(runs[str(s)] for s in range(100, 105)) - runs["buffer"]
        recording = re.sub(r"_part[0-9]+\.log$", "", trace)
        for label in ("All", source):
            groups[label][recording].append(diff)
    for label in ("All", "FCC", "HSDPA"):
        recordings = groups[label]
        low, high = interval(recordings)
        slices = [x for recording in recordings.values() for x in recording]
        print(f"{label:5} {len(slices):3} slices, {len(recordings):2} recordings, "
              f"slice mean {mean(slices):.2f}, equal-recording mean "
              f"{mean(map(mean, recordings.values())):.2f}, 95% CI [{low:.2f}, {high:.2f}]")


if __name__ == "__main__":
    main()
