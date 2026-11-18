#!/usr/bin/env python3
"""Frozen final comparison of Q-learning against both baselines.

Run this once, after ``tune_q_learning.py`` has written
``results/q_learning/best_hyperparameters.json``.  It retrains the selected
configuration with fresh seeds (different from the tuning seeds) on the
training split, then evaluates every policy greedily on one trace per
held-out recording slice, starting at the beginning of each trace exactly as
the baseline scripts do.

Baselines use their declared defaults: fixed 1200 Kbit/s, and buffer-based
with a 10 s reservoir and 40 s cushion.

Outputs (default ``results/final/<split>/``):

* ``summary.json`` / ``summary.md`` - aggregate metrics overall, FCC, HSDPA
* ``per_trace.csv`` - every policy/seed/trace result
* ``q_table_seed<N>.json`` and ``training_curve_seed<N>.csv``
"""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from abr_simulator import (  # noqa: E402
    BufferBasedPolicy,
    FixedBitratePolicy,
    QLearningHyperparameters,
    SimulatorConfig,
    train_agent,
)
from abr_simulator.experiments import (  # noqa: E402
    buffer_selector,
    evaluate_selector,
    fixed_selector,
    load_split,
    load_video,
    mean_and_std,
    q_learning_selector,
    summarize,
    trace_source,
)


METRICS = (
    ("mean_episode_qoe", "Mean QoE per video", "{:.2f}"),
    ("average_bitrate_kbps", "Avg bitrate (Kbps)", "{:.0f}"),
    ("mean_rebuffer_s_per_episode", "Rebuffer per video (s)", "{:.2f}"),
    ("mean_switches_per_episode", "Switches per video", "{:.2f}"),
)
POLICY_LABELS = {
    "fixed": "Fixed 1200 Kbps",
    "buffer": "Buffer-based (10 s / 40 s)",
    "q_learning": "Q-learning",
}


def format_cell(values: list[float], pattern: str) -> str:
    mean, std = mean_and_std(values)
    if len(values) == 1:
        return pattern.format(mean)
    return f"{pattern.format(mean)} ± {pattern.format(std)}"


def markdown_tables(summaries: dict[str, list[dict]], split: str, seeds: list[int]) -> str:
    lines = [f"# Final comparison on the {split} split", ""]
    lines.append(
        f"Q-learning cells are mean ± standard deviation over {len(seeds)} "
        f"independently trained agents (seeds {', '.join(map(str, seeds))}). "
        "Baselines are deterministic."
    )
    for group in ("all", "FCC", "HSDPA"):
        if group not in summaries["fixed"][0]:
            continue
        count = summaries["fixed"][0][group]["episode_count"]
        title = "All traces" if group == "all" else f"{group} traces"
        lines += ["", f"## {title} ({count})", ""]
        lines.append("| Policy | " + " | ".join(label for _, label, _ in METRICS) + " |")
        lines.append("|---" * (len(METRICS) + 1) + "|")
        for policy, label in POLICY_LABELS.items():
            cells = [
                format_cell([run[group][key] for run in summaries[policy]], pattern)
                for key, _, pattern in METRICS
            ]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument(
        "--hyperparameters",
        type=Path,
        default=PROJECT_ROOT / "results/q_learning/best_hyperparameters.json",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or PROJECT_ROOT / "results/final" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SimulatorConfig()
    video = load_video(PROJECT_ROOT)
    train_traces = load_split(PROJECT_ROOT, "train")
    eval_traces = load_split(PROJECT_ROOT, args.split)
    params = QLearningHyperparameters.from_dict(
        json.loads(args.hyperparameters.read_text(encoding="utf-8"))
    )
    print(f"evaluating on {args.split}: {len(eval_traces)} traces")
    print(f"Q-learning hyperparameters: {json.dumps(params.to_dict())}")

    episodes_by_run: dict[tuple[str, int | None], list] = {}
    episodes_by_run[("fixed", None)] = evaluate_selector(
        eval_traces, video, fixed_selector(FixedBitratePolicy(config)), config
    )
    episodes_by_run[("buffer", None)] = evaluate_selector(
        eval_traces, video, buffer_selector(BufferBasedPolicy(config)), config
    )
    for seed in args.seeds:
        agent, returns = train_agent(train_traces, video, params, seed=seed, config=config)
        agent.save(output_dir / f"q_table_seed{seed}.json")
        with (output_dir / f"training_curve_seed{seed}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as curve:
            writer = csv.writer(curve)
            writer.writerow(["episode", "training_qoe"])
            writer.writerows(enumerate(returns, start=1))
        episodes_by_run[("q_learning", seed)] = evaluate_selector(
            eval_traces, video, q_learning_selector(agent), config
        )
        print(f"  trained and evaluated seed {seed}")

    summaries: dict[str, list[dict]] = {policy: [] for policy in POLICY_LABELS}
    for (policy, _seed), episodes in episodes_by_run.items():
        summaries[policy].append(summarize(episodes))

    with (output_dir / "per_trace.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        fields = list(asdict(next(iter(episodes_by_run.values()))[0]).keys())
        writer.writerow(["policy", "seed", "source"] + fields)
        for (policy, seed), episodes in episodes_by_run.items():
            for episode in episodes:
                row = asdict(episode)
                writer.writerow(
                    [policy, "" if seed is None else seed, trace_source(episode.trace_name)]
                    + [row[field] for field in fields]
                )

    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "hyperparameters": params.to_dict(),
                "seeds": args.seeds,
                "summaries": summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    table = markdown_tables(summaries, args.split, args.seeds)
    (output_dir / "summary.md").write_text(table, encoding="utf-8")
    print()
    print(table)


if __name__ == "__main__":
    main()
