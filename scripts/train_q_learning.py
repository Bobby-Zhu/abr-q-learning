#!/usr/bin/env python3
"""Train one tabular Q-learning agent on the training split.

Example:

    python3 scripts/train_q_learning.py --episodes 30000 --seed 0 \
        --output results/q_learning/q_table_seed0.json --evaluate validation

Hyperparameters can also be loaded from a JSON file (for example the one
written by ``tune_q_learning.py``) with ``--hyperparameters``; explicit flags
override values from the file.
"""

import argparse
import csv
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from abr_simulator import QLearningHyperparameters, train_agent  # noqa: E402
from abr_simulator.experiments import (  # noqa: E402
    evaluate_selector,
    load_split,
    load_video,
    q_learning_selector,
    summarize,
)


def parse_edges(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split(","))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hyperparameters", type=Path, help="JSON file of hyperparameters")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--discount", type=float)
    parser.add_argument("--epsilon-start", type=float)
    parser.add_argument("--epsilon-end", type=float)
    parser.add_argument("--epsilon-decay-fraction", type=float)
    parser.add_argument("--throughput-history", type=int)
    parser.add_argument("--throughput-edges", type=parse_edges, help="e.g. 0.5,1,2,3")
    parser.add_argument("--buffer-edges", type=parse_edges, help="e.g. 4,8,16,30")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results/q_learning/q_table.json",
        help="Where to write the trained Q-table",
    )
    parser.add_argument(
        "--curve",
        type=Path,
        help="Optional CSV of per-episode training QoE",
    )
    parser.add_argument(
        "--evaluate",
        choices=("none", "train", "validation"),
        default="validation",
        help="Greedy evaluation after training (test is reserved for evaluate_final.py)",
    )
    parser.add_argument("--log-every", type=int, default=5000)
    return parser.parse_args()


def build_hyperparameters(args: argparse.Namespace) -> QLearningHyperparameters:
    values = QLearningHyperparameters().to_dict()
    if args.hyperparameters:
        values.update(json.loads(args.hyperparameters.read_text(encoding="utf-8")))
    overrides = {
        "episodes": args.episodes,
        "learning_rate": args.learning_rate,
        "discount": args.discount,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay_fraction": args.epsilon_decay_fraction,
        "throughput_history": args.throughput_history,
        "throughput_edges_mbps": args.throughput_edges,
        "buffer_edges_s": args.buffer_edges,
    }
    values.update({key: value for key, value in overrides.items() if value is not None})
    return QLearningHyperparameters.from_dict(values)


def main() -> None:
    args = parse_args()
    params = build_hyperparameters(args)
    video = load_video(PROJECT_ROOT)
    train_traces = load_split(PROJECT_ROOT, "train")

    print(f"training on {len(train_traces)} traces, seed {args.seed}")
    print(json.dumps(params.to_dict()))
    agent, returns = train_agent(
        train_traces, video, params, seed=args.seed, log_every=args.log_every
    )
    agent.save(args.output)
    print(f"saved Q-table to {args.output}")
    print(f"visited states: {agent.visited_state_fraction():.1%}")

    if args.curve:
        args.curve.parent.mkdir(parents=True, exist_ok=True)
        with args.curve.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["episode", "training_qoe"])
            writer.writerows(enumerate(returns, start=1))

    if args.evaluate != "none":
        traces = load_split(PROJECT_ROOT, args.evaluate)
        summary = summarize(evaluate_selector(traces, video, q_learning_selector(agent)))
        overall = summary["all"]
        print(f"greedy evaluation on {args.evaluate} ({overall['episode_count']} traces)")
        print(f"mean episode QoE: {overall['mean_episode_qoe']:.3f}")
        print(f"average bitrate: {overall['average_bitrate_kbps']:.1f} Kbps")
        print(f"mean rebuffering per episode: {overall['mean_rebuffer_s_per_episode']:.3f} s")
        print(f"mean switches per episode: {overall['mean_switches_per_episode']:.3f}")


if __name__ == "__main__":
    main()
