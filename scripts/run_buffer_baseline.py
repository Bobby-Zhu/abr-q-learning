#!/usr/bin/env python3
"""Evaluate the buffer-based baseline on a directory of network traces."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from abr_simulator import (  # noqa: E402
    ABREnvironment,
    BufferBasedPolicy,
    SimulatorConfig,
    aggregate_metrics,
    load_trace_manifest,
    load_video_metadata,
    run_episode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the linear buffer-based ABR baseline."
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="train",
        help="Trace directory to evaluate (default: train)",
    )
    parser.add_argument(
        "--reservoir",
        type=float,
        default=10.0,
        help="Seconds at or below which the lowest bitrate is selected",
    )
    parser.add_argument(
        "--cushion",
        type=float,
        default=40.0,
        help="Seconds over which bitrate rises from lowest to highest",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only the first N traces; 0 evaluates the whole split",
    )
    parser.add_argument(
        "--per-trace",
        action="store_true",
        help="Print one metrics row per trace before the aggregate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative")

    config = SimulatorConfig()
    policy = BufferBasedPolicy(
        config=config,
        reservoir_s=args.reservoir,
        cushion_s=args.cushion,
    )
    video = load_video_metadata(PROJECT_ROOT / "data/video_sizes/mao")
    traces = load_trace_manifest(
        PROJECT_ROOT / "data/splits" / f"{args.split}.txt",
        PROJECT_ROOT / "data",
    )
    if args.limit:
        traces = traces[: args.limit]

    episodes = []
    for trace in traces:
        environment = ABREnvironment(trace, video, config)
        metrics = run_episode(
            environment,
            lambda buffer_s, _previous: policy.select_action(buffer_s),
        )
        episodes.append(metrics)

        if args.per_trace:
            print(
                f"{metrics.trace_name}\t"
                f"QoE={metrics.total_qoe:.3f}\t"
                f"bitrate={metrics.average_bitrate_kbps:.1f} Kbps\t"
                f"rebuffer={metrics.total_rebuffer_s:.3f} s\t"
                f"switches={metrics.switch_count}"
            )

    aggregate = aggregate_metrics(episodes)
    print("buffer-based baseline")
    print(f"split: {args.split}")
    print(
        f"policy: reservoir={policy.reservoir_s:.1f} s, "
        f"cushion={policy.cushion_s:.1f} s, "
        f"upper threshold={policy.upper_threshold_s:.1f} s"
    )
    print(f"episodes: {aggregate.episode_count}")
    print(f"chunks: {aggregate.chunk_count}")
    print(f"mean episode QoE: {aggregate.mean_episode_qoe:.3f}")
    print(f"mean chunk QoE: {aggregate.mean_chunk_qoe:.3f}")
    print(f"average bitrate: {aggregate.average_bitrate_kbps:.1f} Kbps")
    print(
        "mean rebuffering per episode: "
        f"{aggregate.mean_rebuffer_s_per_episode:.3f} s"
    )
    print(f"mean switches per episode: {aggregate.mean_switches_per_episode:.3f}")
    print(
        "average bitrate change per chunk: "
        f"{aggregate.average_bitrate_change_kbps:.1f} Kbps"
    )


if __name__ == "__main__":
    main()
