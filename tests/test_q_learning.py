"""Regression tests for capture-disjoint splits and Q-learning."""

import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from create_splits import TraceRecord, assign_splits, build_manifests, capture_id, trace_fingerprint
from run_q_learning import QLearningPolicy, train_episode

from abr_simulator import ABREnvironment, NetworkTrace, SimulatorConfig, VideoMetadata


ROOT = Path(__file__).resolve().parents[1]


class CaptureSplitTests(unittest.TestCase):
    def test_capture_identifiers_keep_full_recordings_and_clips_together(self):
        for name in ("norway_bus_13", "tram.route-report.2011-01-06_0814CET"):
            self.assertEqual(capture_id(name + ".log"), capture_id(name + "_part2.log"))
        self.assertEqual(capture_id("trace_42_full"), capture_id("trace_42_slice"))

    def test_duplicate_connects_alias_to_upstream_training(self):
        records = [
            TraceRecord("train/a", "train", "a", "FCC", "same"),
            TraceRecord("holdout/b", "holdout", "b", "FCC", "same"),
            TraceRecord("holdout/b_part", "holdout", "b", "FCC", "other"),
            TraceRecord("holdout/c", "holdout", "c", "FCC", "c"),
            TraceRecord("holdout/d", "holdout", "d", "FCC", "d"),
        ]
        splits, audit = assign_splits(records)
        self.assertEqual(splits["train"], ["holdout/b_part", "train/a"])
        self.assertEqual(audit["excluded_duplicates"], 1)
        by_path = {row["path"]: row for row in audit["files"]}
        self.assertEqual(by_path["holdout/b"]["duplicate_of"], "train/a")
        self.assertEqual(by_path["holdout/b_part"]["reason"], "capture overlaps upstream training")

    def test_trace_normalization_and_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            trace = Path(temp) / "trace"
            trace.write_text("12.3 0\n12.6 1.0\n", encoding="utf-8")
            expected = trace_fingerprint(trace)
            trace.write_text("0 0.0\n.3 1\n", encoding="utf-8")
            self.assertEqual(trace_fingerprint(trace), expected)
            for invalid in ("0 1\n0 2\n", "0 0\n1 -1\n", "0 0\n1 0\n", "0 0\n1 NaN\n"):
                trace.write_text(invalid, encoding="utf-8")
                with self.assertRaises(ValueError):
                    trace_fingerprint(trace)

    def test_committed_splits_are_deterministic_and_disjoint(self):
        manifests, audit = build_manifests()
        root = ROOT / "data/splits"
        self.assertEqual(audit, json.loads((root / "capture_audit.json").read_text(encoding="utf-8")))
        self.assertEqual(audit["input_traces"], 428)
        for split, entries in manifests.items():
            self.assertEqual(entries, (root / f"{split}.txt").read_text(encoding="utf-8").splitlines())
            self.assertTrue(all("\\" not in entry for entry in entries))
        for field in ("path", "capture", "fingerprint"):
            seen = {}
            for row in audit["files"]:
                if row["duplicate_of"] is None:
                    self.assertEqual(seen.setdefault(row[field], row["split"]), row["split"])
        self.assertEqual(sum(map(len, manifests.values())),
                         audit["input_traces"] - audit["excluded_duplicates"])


class QLearningTests(unittest.TestCase):
    def test_terminal_transition_uses_reward_only(self):
        policy = QLearningPolicy(SimulatorConfig())
        policy.values[policy.initial_state][0] = 10.0
        policy.update(policy.initial_state, 0, -2.0)
        self.assertAlmostEqual(policy.values[policy.initial_state][0], 8.8)
        policy.update(0, 0, -2.0, policy.initial_state)
        self.assertAlmostEqual(policy.values[0][0], 0.636)
        self.assertEqual(len(policy.values), 181)

    def test_future_bandwidth_changes_download_but_not_selected_action(self):
        config = SimulatorConfig()
        video = VideoMetadata(((100_000, 1_000_000),) * 6)
        policy = QLearningPolicy(config)
        snapshots = []
        for future in (0.8, 40.0):
            trace = NetworkTrace("synthetic", (0.0, 1.0, 20.0), (8.0, 8.0, future))
            env = ABREnvironment(trace, video, config)
            first = env.step(policy.select_action(0.0, None))
            state = policy.state(env.buffer_s, first)
            policy.values[state][2] = 1.0
            next_action = policy.select_action(env.buffer_s, first)
            second = env.step(next_action)
            snapshots.append((first, state, next_action, second))
        slow, fast = snapshots
        self.assertEqual(slow[:3], fast[:3])
        self.assertEqual(slow[2], 2)
        self.assertGreater(slow[3].download_time_s, fast[3].download_time_s)

    def test_first_chunk_reward_is_learned_from_shared_environment(self):
        config = SimulatorConfig()
        trace = NetworkTrace("one-chunk", (0.0, 1.0), (8.0, 8.0))
        video = VideoMetadata(((100_000,),) * 6)
        env = ABREnvironment(trace, video, config)
        reward = env.step(0).reward
        policy = QLearningPolicy(config)
        train_episode(env, policy, random.Random(42), epsilon=0.0)
        self.assertAlmostEqual(policy.values[policy.initial_state][0], 0.1 * reward)
        self.assertTrue(all(not any(row) for row in policy.values[:-1]))


if __name__ == "__main__":
    unittest.main()
