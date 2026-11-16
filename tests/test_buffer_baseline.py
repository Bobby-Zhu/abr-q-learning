"""Tests for the linear buffer-based policy and evaluation metrics."""

import math
import unittest

from abr_simulator import (
    ABREnvironment,
    BufferBasedPolicy,
    NetworkTrace,
    SimulatorConfig,
    VideoMetadata,
    aggregate_metrics,
    run_episode,
)


class BufferBasedPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = BufferBasedPolicy()

    def test_threshold_mapping(self) -> None:
        expected_actions = {
            0.0: 0,
            10.0: 0,
            17.999: 0,
            18.0: 1,
            26.0: 2,
            34.0: 3,
            42.0: 4,
            49.999: 4,
            50.0: 5,
            60.0: 5,
        }

        for buffer_s, expected_action in expected_actions.items():
            with self.subTest(buffer_s=buffer_s):
                self.assertEqual(
                    self.policy.select_action(buffer_s), expected_action
                )

    def test_actions_are_monotonic_and_valid(self) -> None:
        actions = [self.policy.select_action(value / 10) for value in range(601)]

        self.assertEqual(actions, sorted(actions))
        self.assertTrue(all(0 <= action <= 5 for action in actions))

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BufferBasedPolicy(reservoir_s=-1)
        with self.assertRaises(ValueError):
            BufferBasedPolicy(cushion_s=0)
        with self.assertRaises(ValueError):
            BufferBasedPolicy(reservoir_s=30, cushion_s=40)

    def test_invalid_buffer_is_rejected(self) -> None:
        for buffer_s in (-1.0, math.inf, math.nan):
            with self.subTest(buffer_s=buffer_s):
                with self.assertRaises(ValueError):
                    self.policy.select_action(buffer_s)


class BufferEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SimulatorConfig(
            payload_fraction=1.0,
            link_rtt_s=0.0,
        )
        self.trace = NetworkTrace(
            "constant-8-mbps",
            tuple(float(value) for value in range(12)),
            (8.0,) * 12,
        )
        self.video = VideoMetadata(
            tuple((1_000_000, 1_000_000) for _ in self.config.bitrates_kbps)
        )

    def test_episode_metrics_match_known_two_chunk_run(self) -> None:
        environment = ABREnvironment(self.trace, self.video, self.config)
        policy = BufferBasedPolicy(self.config)

        metrics = run_episode(
            environment,
            lambda buffer_s, _previous: policy.select_action(buffer_s),
        )

        self.assertEqual(metrics.chunk_count, 2)
        self.assertAlmostEqual(metrics.total_qoe, -4.15)
        self.assertAlmostEqual(metrics.average_bitrate_kbps, 300.0)
        self.assertAlmostEqual(metrics.total_rebuffer_s, 1.0)
        self.assertEqual(metrics.switch_count, 1)
        self.assertEqual(metrics.total_bitrate_change_kbps, 450)
        self.assertAlmostEqual(metrics.average_bitrate_change_kbps, 225.0)

    def test_aggregate_metrics_combines_episodes(self) -> None:
        policy = BufferBasedPolicy(self.config)
        episodes = [
            run_episode(
                ABREnvironment(self.trace, self.video, self.config),
                lambda buffer_s, _previous: policy.select_action(buffer_s),
            )
            for _ in range(2)
        ]

        aggregate = aggregate_metrics(episodes)

        self.assertEqual(aggregate.episode_count, 2)
        self.assertEqual(aggregate.chunk_count, 4)
        self.assertAlmostEqual(aggregate.mean_episode_qoe, -4.15)
        self.assertAlmostEqual(aggregate.mean_chunk_qoe, -2.075)


if __name__ == "__main__":
    unittest.main()
