#!/usr/bin/env python3
"""Tune the buffer rule on validation, then compare it with Q-learning on test.

Example (Python 3.10+, no third-party dependencies):
    python tuned_buffer.py --repo /path/to/abr-q-learning

Searches the reservoir and cushion on the 129 validation slices only, scores
the best setting once on the 133 test slices, and reports the paired
recording bootstrap of the archived five-seed Q-learning mean minus the tuned
rule (same procedure as recording_intervals.py). No training, network access,
or repository writes are performed.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import random
import re
import statistics
import sys

RESERVOIRS = (0, 2, 4, 5, 6, 8, 10, 12, 15)
CUSHIONS = (10, 15, 20, 25, 30, 35, 40, 45, 50)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, required=True, help="Repository root")
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = args.repo.expanduser().resolve()
    if not (root / "src/abr_simulator").is_dir():
        parser.error(f"Not an abr-q-learning repository: {root}")

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "src"))
    from abr_simulator import ABREnvironment, BufferBasedPolicy, SimulatorConfig
    from abr_simulator.evaluation import run_episode
    from abr_simulator.experiments import load_split, load_video

    config = SimulatorConfig()
    video = load_video(root)
    splits = {name: load_split(root, name) for name in ("validation", "test")}

    def per_trace(split, reservoir, cushion):
        rule = BufferBasedPolicy(config, reservoir_s=reservoir, cushion_s=cushion)
        return {trace.name: run_episode(ABREnvironment(trace, video, config),
                                        lambda buffer_s, _: rule.select_action(buffer_s)).total_qoe
                for trace in splits[split]}

    grid = []
    for reservoir in RESERVOIRS:
        for cushion in CUSHIONS:
            if reservoir + cushion <= config.max_buffer_s:
                grid.append((statistics.fmean(per_trace("validation", reservoir, cushion).values()),
                             reservoir, cushion))
    grid.sort(key=lambda item: (-item[0], item[1], item[2]))
    default_val = statistics.fmean(per_trace("validation", 10, 40).values())
    best_val, reservoir, cushion = grid[0]
    print(f"validation: default (10, 40) {default_val:.2f}; "
          f"best ({reservoir}, {cushion}) {best_val:.2f}")

    tuned = per_trace("test", reservoir, cushion)
    print(f"test: tuned rule {statistics.fmean(tuned.values()):.2f}, "
          f"default rule {statistics.fmean(per_trace('test', 10, 40).values()):.2f}")

    q_scores = defaultdict(list)
    with (root / "results/final/test/per_trace.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["policy"] == "q_learning":
                q_scores[row["trace_name"]].append(float(row["total_qoe"]))
    groups = defaultdict(list)
    for name, value in tuned.items():
        groups[re.sub(r"_part[0-9]+\.log$", "", name)].append(statistics.fmean(q_scores[name]) - value)
    recordings = list(groups.values())
    slices = [x for group in recordings for x in group]
    rng = random.Random(args.seed)
    means = sorted(statistics.fmean([x for _ in recordings for x in rng.choice(recordings)])
                   for _ in range(args.draws))
    low, high = means[int(0.025 * args.draws)], means[int(0.975 * args.draws) - 1]
    print(f"Q-learning minus tuned rule: {statistics.fmean(slices):.2f} over {len(slices)} slices, "
          f"{len(recordings)} recordings, 95% CI [{low:.2f}, {high:.2f}]")


if __name__ == "__main__":
    main()
