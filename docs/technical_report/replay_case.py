#!/usr/bin/env python3
"""Replay the report's HSDPA failure using the repository's saved Q-table.

Example (Python 3.10+, no third-party dependencies):
    python replay_case.py --repo /path/to/abr-q-learning

The complete 49-chunk episodes are replayed independently from start_index=1.
Only the requested one-based chunk numbers are printed; totals cover the full
video. Results are checked against the committed per_trace.csv. No training,
network access, or repository writes are performed.
"""

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, required=True, help="Repository root")
    parser.add_argument("--trace", default="norway_car_1_part2.log")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--chunks", type=int, nargs="+", default=[10, 39],
                        help="One-based chunk numbers (default: 10 39)")
    args = parser.parse_args()
    root = args.repo.expanduser().resolve()
    if not (root / "src/abr_simulator").is_dir():
        parser.error(f"Not an abr-q-learning repository: {root}")

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "src"))
    from abr_simulator import ABREnvironment, BufferBasedPolicy, QLearningAgent
    from abr_simulator import SimulatorConfig
    from abr_simulator.evaluation import run_episode
    from abr_simulator.experiments import load_split, load_video

    config = SimulatorConfig()
    video = load_video(root)
    if any(chunk < 1 or chunk > video.chunk_count for chunk in args.chunks):
        parser.error(f"Chunk numbers must be between 1 and {video.chunk_count}")
    matches = [trace for trace in load_split(root, "test")
               if trace.name == args.trace]
    if len(matches) != 1:
        parser.error(f"Expected exactly one test trace named {args.trace!r}")
    trace = matches[0]
    results_dir = root / "results/final/test"
    table = results_dir / f"q_table_seed{args.seed}.json"
    agent = QLearningAgent.load(table, config)
    baseline = BufferBasedPolicy(config)

    class RecordingEnvironment(ABREnvironment):
        """Observe the public step result without changing simulator rules."""

        def step(self, action_index):
            before = self.buffer_s
            result = super().step(action_index)
            previous_rates = [row["measured_throughput_mbps"] for row in self.rows[-3:]]
            prior_estimate = (len(previous_rates) / sum(1 / max(rate, 1e-9)
                              for rate in previous_rates)) if previous_rates else None
            self.rows.append({
                "chunk_one_based": result.chunk_index + 1,
                "bitrate_kbps": result.bitrate_kbps,
                "buffer_before_s": before,
                "prior_three_download_harmonic_mean_mbps": prior_estimate,
                "download_time_s": result.download_time_s,
                "rebuffer_s": result.rebuffer_s,
                "buffer_after_s": result.buffer_s,
                "measured_throughput_mbps": result.measured_throughput_mbps,
                "reward": result.reward,
            })
            return result

    with (results_dir / "per_trace.csv").open(newline="", encoding="utf-8") as f:
        committed = list(csv.DictReader(f))

    report = {
        "trace": trace.name,
        "split": "test",
        "q_learning_seed": args.seed,
        "q_table": table.relative_to(root).as_posix(),
        "start_index": 1,
        "chunk_numbering": "one-based",
        "policies": {},
    }
    agent.begin_episode()
    selectors = {
        "buffer": lambda buffer_s, previous: baseline.select_action(buffer_s),
        "q_learning": agent.select_action,
    }
    for policy, selector in selectors.items():
        environment = RecordingEnvironment(trace, video, config)
        environment.rows = []
        metrics = asdict(run_episode(environment, selector, start_index=1))
        seed = str(args.seed) if policy == "q_learning" else ""
        expected = [row for row in committed if row["policy"] == policy
                    and row["seed"] == seed and row["trace_name"] == trace.name]
        if len(expected) != 1:
            raise RuntimeError(f"Missing/duplicate committed result: {policy}, {seed}")
        for name, value in metrics.items():
            if name != "trace_name" and not math.isclose(
                    value, float(expected[0][name]), rel_tol=1e-9, abs_tol=1e-9):
                raise RuntimeError(f"Replay differs from CSV: {policy}.{name}")
        report["policies"][policy] = {
            "full_episode_metrics": metrics,
            "matches_committed_csv": True,
            "selected_chunks": [row for row in environment.rows
                                if row["chunk_one_based"] in args.chunks],
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
