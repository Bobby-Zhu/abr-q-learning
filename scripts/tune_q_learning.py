#!/usr/bin/env python3
"""Select Q-learning hyperparameters on the validation split only.

The search was run as two rounds (coarse-to-fine).  Round 1 covered a wide
grid; its best values sat on the grid edge (smallest learning rate, smallest
discount, finest bins), so round 2 extends the grid past that edge, drops
values that were clearly worse, and uses more seeds.  ``--round 1``
reproduces the first round.  Each round has two stages:

1. **Structure and learning rule.**  Every combination of state-bin preset,
   learning rate, discount, and throughput smoothing, trained for a fixed
   number of episodes.
2. **Training schedule.**  For the best stage-1 configuration, vary the
   number of training episodes and how quickly epsilon decays.

Each configuration is trained with several seeds and scored by the mean
validation episode QoE across seeds.  The test split is never loaded here.

Outputs (default ``results/q_learning/``):

* ``tuning.csv`` (``tuning_round1.csv`` for round 1) - one row per
  configuration with mean/std validation metrics
* ``best_hyperparameters.json`` - the selected configuration
"""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import itertools
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from abr_simulator import QLearningHyperparameters, train_agent  # noqa: E402
from abr_simulator.experiments import (  # noqa: E402
    evaluate_selector,
    load_split,
    load_video,
    mean_and_std,
    q_learning_selector,
    summarize,
)


# Throughput edges are roughly the training-trace bandwidth quantiles; buffer
# edges are spaced more finely at low buffer, where rebuffering risk lives.
BIN_PRESETS = {
    "coarse": {
        "throughput_edges_mbps": (0.8, 1.6, 3.0),
        "buffer_edges_s": (5.0, 12.0, 25.0),
    },
    "medium": {
        "throughput_edges_mbps": (0.5, 1.0, 1.5, 2.2, 3.2),
        "buffer_edges_s": (4.0, 8.0, 12.0, 18.0, 26.0),
    },
    "fine": {
        "throughput_edges_mbps": (0.4, 0.7, 1.0, 1.4, 1.9, 2.6, 3.5),
        "buffer_edges_s": (2.0, 4.0, 7.0, 10.0, 14.0, 19.0, 26.0, 35.0),
    },
}
ROUNDS = {
    1: {
        "presets": ("coarse", "medium", "fine"),
        "learning_rates": (0.05, 0.1, 0.2),
        "discounts": (0.8, 0.9, 0.99),
        "histories": (1, 3),
        "stage1_episodes": 30_000,
        "stage2_episodes": (10_000, 30_000, 100_000),
        "seeds": 3,
    },
    2: {
        "presets": ("medium", "fine"),
        "learning_rates": (0.02, 0.05, 0.1),
        "discounts": (0.5, 0.65, 0.8, 0.9),
        "histories": (1, 3),
        "stage1_episodes": 30_000,
        "stage2_episodes": (30_000, 100_000, 200_000),
        "seeds": 5,
    },
}
STAGE2_DECAY_FRACTIONS = (0.5, 0.8)

_VIDEO = None
_TRAIN = None
_VALIDATION = None


def _init_worker() -> None:
    global _VIDEO, _TRAIN, _VALIDATION
    _VIDEO = load_video(PROJECT_ROOT)
    _TRAIN = load_split(PROJECT_ROOT, "train")
    _VALIDATION = load_split(PROJECT_ROOT, "validation")


def _run(job: tuple[str, dict, int]) -> tuple[str, int, dict]:
    name, params_dict, seed = job
    params = QLearningHyperparameters.from_dict(params_dict)
    agent, _ = train_agent(_TRAIN, _VIDEO, params, seed=seed)
    summary = summarize(
        evaluate_selector(_VALIDATION, _VIDEO, q_learning_selector(agent))
    )["all"]
    summary["visited_state_fraction"] = agent.visited_state_fraction()
    return name, seed, summary


def run_configs(
    configs: dict[str, QLearningHyperparameters],
    seeds: list[int],
    workers: int,
) -> dict[str, list[dict]]:
    jobs = [
        (name, params.to_dict(), seed)
        for name, params in configs.items()
        for seed in seeds
    ]
    results: dict[str, list[dict]] = {name: [] for name in configs}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        for done, (name, _seed, summary) in enumerate(pool.map(_run, jobs), start=1):
            results[name].append(summary)
            if done % max(1, len(jobs) // 10) == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} runs finished", flush=True)
    return results


def score(rows: list[dict]) -> float:
    return mean_and_std([row["mean_episode_qoe"] for row in rows])[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--round", type=int, choices=sorted(ROUNDS), default=2)
    parser.add_argument("--seeds", type=int, help="seeds per configuration (default: per round)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "results/q_learning"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tiny search for smoke-testing the script",
    )
    args = parser.parse_args()
    grid = ROUNDS[args.round]
    seeds = list(range(args.seeds or grid["seeds"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    presets = {name: BIN_PRESETS[name] for name in grid["presets"]}
    learning_rates, discounts, histories = (
        grid["learning_rates"], grid["discounts"], grid["histories"]
    )
    stage1_episodes, stage2_episodes = grid["stage1_episodes"], grid["stage2_episodes"]
    if args.quick:
        presets = {"medium": BIN_PRESETS["medium"]}
        learning_rates, discounts, histories = (0.1,), (0.9,), (1,)
        stage1_episodes, stage2_episodes = 500, (500,)

    stage1 = {}
    for (preset, bins), lr, gamma, history in itertools.product(
        presets.items(), learning_rates, discounts, histories
    ):
        name = f"s1 bins={preset} lr={lr} gamma={gamma} hist={history}"
        stage1[name] = QLearningHyperparameters(
            learning_rate=lr,
            discount=gamma,
            throughput_history=history,
            episodes=stage1_episodes,
            **bins,
        )
    print(f"stage 1: {len(stage1)} configurations x {len(seeds)} seeds")
    results = run_configs(stage1, seeds, args.workers)
    best_stage1 = max(stage1, key=lambda name: score(results[name]))
    print(f"best stage 1: {best_stage1}  ({score(results[best_stage1]):.3f})")

    stage2 = {}
    for episodes, decay in itertools.product(stage2_episodes, STAGE2_DECAY_FRACTIONS):
        name = f"s2 {best_stage1[3:]} episodes={episodes} decay={decay}"
        stage2[name] = replace(
            stage1[best_stage1], episodes=episodes, epsilon_decay_fraction=decay
        )
    print(f"stage 2: {len(stage2)} configurations x {len(seeds)} seeds")
    results.update(run_configs(stage2, seeds, args.workers))
    all_configs = {**stage1, **stage2}

    best = max(all_configs, key=lambda name: score(results[name]))
    print(f"selected: {best}  ({score(results[best]):.3f})")

    metric_keys = (
        "mean_episode_qoe",
        "average_bitrate_kbps",
        "mean_rebuffer_s_per_episode",
        "mean_switches_per_episode",
        "visited_state_fraction",
    )
    csv_name = "tuning.csv" if args.round == 2 else f"tuning_round{args.round}.csv"
    with (args.output_dir / csv_name).open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            ["name", "stage", "bins", "learning_rate", "discount", "throughput_history",
             "episodes", "epsilon_decay_fraction"]
            + [f"{key}_{stat}" for key in metric_keys for stat in ("mean", "std")]
        )
        for name in sorted(all_configs, key=lambda n: -score(results[n])):
            params = all_configs[name]
            preset = next(
                key for key, bins in BIN_PRESETS.items()
                if bins["throughput_edges_mbps"] == params.throughput_edges_mbps
            )
            row = [name, name[:2], preset, params.learning_rate, params.discount,
                   params.throughput_history, params.episodes,
                   params.epsilon_decay_fraction]
            for key in metric_keys:
                mean, std = mean_and_std([r[key] for r in results[name]])
                row += [f"{mean:.4f}", f"{std:.4f}"]
            writer.writerow(row)

    print(f"wrote {args.output_dir / csv_name}")
    if args.round == max(ROUNDS) or args.quick:
        selected = all_configs[best].to_dict()
        (args.output_dir / "best_hyperparameters.json").write_text(
            json.dumps(selected, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.output_dir / 'best_hyperparameters.json'}")


if __name__ == "__main__":
    main()
