"""Train tabular Q-learning and compare it with the shared ABR baselines."""

import argparse
from bisect import bisect_right
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from abr_simulator import (
    ABREnvironment, BufferBasedPolicy, FixedBitratePolicy, SimulatorConfig,
    aggregate_metrics, load_trace_manifest, load_video_metadata, run_episode,
)
from create_splits import build_manifests, capture_id


BUFFER_EDGES = (4, 8, 16, 32)
CHECKPOINTS = (1, 3, 6, 12, 24, 40)
SEED = 42
ALPHA = 0.1
GAMMA = 0.95

DATA_ROOT = PROJECT_ROOT / "data"
SPLIT_ROOT = DATA_ROOT / "splits"


class QLearningPolicy:
    def __init__(self, config: SimulatorConfig) -> None:
        self.action_count = len(config.bitrates_kbps)
        self.throughput_edges = tuple(rate / 1000 for rate in config.bitrates_kbps[1:])
        self.initial_state = self.action_count**2 * (len(BUFFER_EDGES) + 1)
        self.values = [[0.0] * self.action_count for _ in range(self.initial_state + 1)]

    def state(self, buffer_s, previous):
        if previous is None:
            return self.initial_state
        throughput_bin = bisect_right(self.throughput_edges, previous.measured_throughput_mbps)
        buffer_bin = bisect_right(BUFFER_EDGES, buffer_s)
        return ((throughput_bin * (len(BUFFER_EDGES) + 1) + buffer_bin) * self.action_count
                + previous.action_index)

    def select_action(self, buffer_s, previous):
        row = self.values[self.state(buffer_s, previous)]
        return max(range(self.action_count), key=row.__getitem__)

    def update(self, state, action, reward, next_state=None):
        target = reward if next_state is None else reward + GAMMA * max(self.values[next_state])
        self.values[state][action] += ALPHA * (target - self.values[state][action])


def train_episode(environment, policy, rng, epsilon):
    environment.reset()
    previous = None
    while True:
        state = policy.state(environment.buffer_s, previous)
        action = (rng.randrange(policy.action_count) if rng.random() < epsilon
                  else policy.select_action(environment.buffer_s, previous))
        result = environment.step(action)
        next_state = None if result.done else policy.state(result.buffer_s, result)
        policy.update(state, action, result.reward, next_state)
        if result.done:
            return
        previous = result


def load_split(name):
    path = SPLIT_ROOT / f"{name}.txt"
    entries = path.read_text(encoding="utf-8").splitlines()
    return list(zip(entries, load_trace_manifest(path, DATA_ROOT), strict=True))


def evaluate(traces, video, config, action):
    return [(entry, run_episode(ABREnvironment(trace, video, config), action))
            for entry, trace in traces]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def historical_overlap(audit):
    manifest = PROJECT_ROOT / "prototype/results/run_01/manifest.csv"
    prototype_groups = {}
    with manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["status"] == "kept":
                _, group = capture_id(Path(row["path"]).name)
                prototype_groups[group] = row["split"]
    test_rows = [row for row in audit["files"]
                 if row["split"] == "test" and row["duplicate_of"] is None]
    return {name: {
        "captures": len({row["capture"] for row in test_rows
                         if prototype_groups.get(row["original_capture"]) == name}),
        "traces": sum(prototype_groups.get(row["original_capture"]) == name for row in test_rows),
    } for name in ("train", "validation", "test")}


def run(out):
    manifests, audit = build_manifests()
    for name, expected in manifests.items():
        actual = (SPLIT_ROOT / f"{name}.txt").read_text(encoding="utf-8").splitlines()
        if actual != expected:
            raise ValueError(f"Stale manifest for {name}; run scripts/create_splits.py")
    if json.loads((SPLIT_ROOT / "capture_audit.json").read_text(encoding="utf-8")) != audit:
        raise ValueError("Stale capture audit; run scripts/create_splits.py")

    out.mkdir(parents=True, exist_ok=False)
    config = SimulatorConfig()
    video = load_video_metadata(DATA_ROOT / "video_sizes/mao")
    training = load_split("train")
    validation = load_split("validation")
    policy = QLearningPolicy(config)
    rng = random.Random(SEED)
    validation_scores = []
    best_qoe = float("-inf")
    for epoch in range(1, CHECKPOINTS[-1] + 1):
        rng.shuffle(training)
        epsilon = max(0.05, 0.5 * 0.92 ** (epoch - 1))
        for _, trace in training:
            train_episode(ABREnvironment(trace, video, config), policy, rng, epsilon)
        if epoch in CHECKPOINTS:
            episodes = evaluate(validation, video, config, policy.select_action)
            score = aggregate_metrics(m for _, m in episodes).mean_chunk_qoe
            validation_scores.append(dict(epoch=epoch, mean_chunk_qoe=score))
            if score > best_qoe:
                best_qoe = score
                best_epoch = epoch
                best_table = [row.copy() for row in policy.values]

    write_csv(out / "validation.csv", validation_scores)
    policy.values = best_table
    frozen = tuple(tuple(row) for row in policy.values)
    save_json(out / "q_table.json", policy.values)

    test = load_split("test")
    fixed = FixedBitratePolicy(config)
    buffer = BufferBasedPolicy(config)
    policies = {
        "fixed": lambda _buffer, _previous: fixed.select_action(),
        "buffer": lambda seconds, _previous: buffer.select_action(seconds),
        "q_learning": policy.select_action,
    }
    groups = {r["path"]: r for r in audit["files"] if r["duplicate_of"] is None}
    expected_ids = [entry for entry, _ in test]
    per_trace = []
    summary = []
    for name, action in policies.items():
        episodes = evaluate(test, video, config, action)
        if [entry for entry, _ in episodes] != expected_ids:
            raise RuntimeError("Policies received different test traces")
        for entry, metric in episodes:
            row = groups[entry]
            per_trace.append(dict(policy=name, path=entry, capture=row["capture"],
                                  source=row["source"], **asdict(metric)))
        for source in ("all", "FCC", "non-FCC"):
            selected = [(entry, m) for entry, m in episodes
                        if source == "all" or groups[entry]["source"] == source]
            by_capture = {}
            for entry, metric in selected:
                by_capture.setdefault(groups[entry]["capture"], []).append(metric.mean_chunk_qoe)
            capture_qoe = sum(sum(qoe) / len(qoe) for qoe in by_capture.values()) / len(by_capture)
            summary.append(dict(policy=name, source=source, captures=len(by_capture),
                                capture_mean_chunk_qoe=capture_qoe,
                                **asdict(aggregate_metrics(m for _, m in selected))))
    if tuple(tuple(row) for row in policy.values) != frozen:
        raise RuntimeError("Test evaluation changed the frozen Q-table")
    write_csv(out / "per_trace.csv", per_trace)
    write_csv(out / "summary.csv", summary)
    save_json(out / "selection.json", dict(
        seed=SEED, learning_rate=ALPHA, discount=GAMMA, checkpoints=CHECKPOINTS,
        selected_epoch=best_epoch, selected_validation_mean_chunk_qoe=best_qoe,
        throughput_edges_mbps=policy.throughput_edges, buffer_edges_s=BUFFER_EDGES,
        initial_state=policy.initial_state, simulator=asdict(config),
        fixed_bitrate_kbps=fixed.bitrate_kbps, buffer_reservoir_s=buffer.reservoir_s,
        buffer_cushion_s=buffer.cushion_s, splits=audit["splits"],
        split_sha256={name: hashlib.sha256((SPLIT_ROOT / name).read_bytes()).hexdigest()
                      for name in ("train.txt", "validation.txt", "test.txt", "capture_audit.json")},
        test_capture_overlap_with_historical_prototype=historical_overlap(audit),
    ))
    print(f"Selected epoch {best_epoch}, mean validation chunk QoE {best_qoe:.6f}")
    for row in summary:
        if row["source"] == "all":
            print(f"{row['policy']}: mean test chunk QoE {row['mean_chunk_qoe']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
