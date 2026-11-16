#!/usr/bin/env python3
"""Run one complete video with a fixed bitrate as a simulator smoke test."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from abr_simulator import (  # noqa: E402
    ABREnvironment,
    SimulatorConfig,
    load_trace,
    load_video_metadata,
)


def parse_args() -> argparse.Namespace:
    default_trace = sorted(
        (PROJECT_ROOT / "data/traces/fcc_hsdpa/train").iterdir()
    )[0]
    parser = argparse.ArgumentParser(
        description="Simulate one video using a constant bitrate policy."
    )
    parser.add_argument("--trace", type=Path, default=default_trace)
    parser.add_argument(
        "--bitrate-index",
        type=int,
        default=2,
        help="Index in [300, 750, 1200, 1850, 2850, 4300] Kbps (default: 2)",
    )
    parser.add_argument(
        "--show-chunks",
        type=int,
        default=8,
        help="Number of per-chunk rows to print before the summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulatorConfig()
    trace = load_trace(args.trace)
    video = load_video_metadata(PROJECT_ROOT / "data/video_sizes/mao")
    environment = ABREnvironment(trace, video, config)

    results = []
    while True:
        result = environment.step(args.bitrate_index)
        results.append(result)
        if result.done:
            break

    print(f"trace: {trace.name}")
    print(f"fixed bitrate: {config.bitrates_kbps[args.bitrate_index]} Kbps")
    print("chunk  delay_s  buffer_s  rebuffer_s  throughput_Mbps  reward")
    for result in results[: args.show_chunks]:
        print(
            f"{result.chunk_index:>5}  "
            f"{result.download_time_s:>7.3f}  "
            f"{result.buffer_s:>8.3f}  "
            f"{result.rebuffer_s:>10.3f}  "
            f"{result.measured_throughput_mbps:>15.3f}  "
            f"{result.reward:>6.3f}"
        )
    if len(results) > args.show_chunks:
        print(f"... {len(results) - args.show_chunks} more chunks simulated")

    total_rebuffer = sum(result.rebuffer_s for result in results)
    total_reward = sum(result.reward for result in results)
    average_throughput = sum(
        result.measured_throughput_mbps for result in results
    ) / len(results)
    print("\nsummary")
    print(f"chunks: {len(results)}")
    print(f"total rebuffer: {total_rebuffer:.3f} s")
    print(f"average measured throughput: {average_throughput:.3f} Mbps")
    print(f"total QoE reward: {total_reward:.3f}")


if __name__ == "__main__":
    main()
