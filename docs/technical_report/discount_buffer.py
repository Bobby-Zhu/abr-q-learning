#!/usr/bin/env python3
"""Validation-only check of how the discount factor changes steady-state buffering.

Example (Python 3.10+, no third-party dependencies):
    python discount_buffer.py --repo /path/to/abr-q-learning --out results.csv

Retrains the selected configuration with each discount and the five tuning
seeds (0-4), keeping every other hyperparameter fixed, and evaluates the
greedy policies on the 129 validation slices. The test split is never loaded.
Reports QoE, bitrate, rebuffering split into start-up (chunks 1-10) and steady
state (chunks 11-49), and the mean buffer in chunks 20-49.
"""

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--discounts", type=float, nargs="+", default=[0.5, 0.8, 0.9, 0.95])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes", type=int, help="Override the training budget")
    parser.add_argument("--out", type=Path, help="Optional per-seed CSV")
    args = parser.parse_args()
    root = args.repo.expanduser().resolve()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "src"))
    from abr_simulator import ABREnvironment, SimulatorConfig
    from abr_simulator.experiments import load_split, load_video, trace_source
    from abr_simulator.policies.q_learning import QLearningHyperparameters, train_agent

    config = SimulatorConfig()
    video = load_video(root)
    train, validation = load_split(root, "train"), load_split(root, "validation")
    base = QLearningHyperparameters.from_dict(
        json.loads((root / "results/q_learning/best_hyperparameters.json").read_text()))
    if args.episodes:
        base = replace(base, episodes=args.episodes)

    rows = []
    for discount in args.discounts:
        for seed in args.seeds:
            agent, _ = train_agent(train, video, replace(base, discount=discount),
                                   seed=seed, config=config)
            qoe, kbps, early_rb, late_rb, late_buffer = [], [], [], [], []
            by_source = {"FCC": [], "HSDPA": []}
            for trace in validation:
                env = ABREnvironment(trace, video, config)
                env.reset(start_index=1)
                agent.begin_episode()
                previous, total, rates, rb = None, 0.0, [], [0.0, 0.0]
                while True:
                    buffer_s = env.buffer_s
                    result = env.step(agent.select_action(buffer_s, previous))
                    chunk = result.chunk_index + 1
                    if chunk >= 20:
                        late_buffer.append(buffer_s)
                    rb[chunk > 10] += result.rebuffer_s
                    total += result.reward
                    rates.append(result.bitrate_kbps)
                    previous = result
                    if result.done:
                        break
                qoe.append(total)
                by_source[trace_source(trace.name)].append(total)
                kbps.append(statistics.fmean(rates))
                early_rb.append(rb[0])
                late_rb.append(rb[1])
            row = {"discount": discount, "seed": seed, "qoe": statistics.fmean(qoe),
                   "bitrate_kbps": statistics.fmean(kbps),
                   "rebuffer_1_10_s": statistics.fmean(early_rb),
                   "rebuffer_11_49_s": statistics.fmean(late_rb),
                   "buffer_20_49_s": statistics.fmean(late_buffer),
                   "qoe_fcc": statistics.fmean(by_source["FCC"]),
                   "qoe_hsdpa": statistics.fmean(by_source["HSDPA"])}
            rows.append(row)
            print(", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                            for k, v in row.items()), flush=True)
    print(f"\nepisodes per run: {base.episodes}")
    print("discount  QoE            bitrate  rebuf 1-10  rebuf 11-49  buffer 20-49  FCC QoE  HSDPA QoE")
    for discount in args.discounts:
        sel = [r for r in rows if r["discount"] == discount]
        m = lambda key: statistics.fmean(r[key] for r in sel)
        sd = statistics.stdev(r["qoe"] for r in sel) if len(sel) > 1 else 0.0
        print(f"{discount:<9} {m('qoe'):6.2f} ± {sd:4.2f}  {m('bitrate_kbps'):7.0f}  "
              f"{m('rebuffer_1_10_s'):10.2f}  {m('rebuffer_11_49_s'):11.2f}  {m('buffer_20_49_s'):12.1f}  "
              f"{m('qoe_fcc'):7.2f}  {m('qoe_hsdpa'):9.2f}")
    if args.out:
        with args.out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
