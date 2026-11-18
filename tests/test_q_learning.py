"""Tests for the tabular Q-learning agent."""

import random
import tempfile
import unittest
from pathlib import Path

from abr_simulator import (
    ABREnvironment,
    NetworkTrace,
    QLearningAgent,
    QLearningHyperparameters,
    SimulatorConfig,
    StateDiscretizer,
    VideoMetadata,
    run_episode,
    train_agent,
)


def tiny_config() -> SimulatorConfig:
    return SimulatorConfig(bitrates_kbps=(300, 750), initial_bitrate_index=1)


def tiny_video() -> VideoMetadata:
    return VideoMetadata(((150_000,) * 4, (375_000,) * 4))


def tiny_trace(bandwidth_mbps: float = 1.0) -> NetworkTrace:
    times = tuple(float(second) for second in range(200))
    return NetworkTrace("flat", times, (bandwidth_mbps,) * len(times))


class StateDiscretizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.discretizer = StateDiscretizer(
            SimulatorConfig(),
            throughput_edges_mbps=(1.0, 2.0),
            buffer_edges_s=(5.0,),
        )

    def test_bins_and_edges(self) -> None:
        cases = {
            (0.5, 0.0, 0): (0, 0, 0),
            (1.0, 4.99, 3): (1, 0, 3),
            (1.99, 5.0, 5): (1, 1, 5),
            (7.0, 60.0, 2): (2, 1, 2),
        }
        for (throughput, buffer_s, last), expected in cases.items():
            with self.subTest(throughput=throughput, buffer_s=buffer_s):
                self.assertEqual(
                    self.discretizer.state(throughput, buffer_s, last), expected
                )

    def test_unmeasured_throughput_has_its_own_bin(self) -> None:
        self.assertEqual(self.discretizer.state(None, 0.0, 1), (3, 0, 1))
        self.assertEqual(self.discretizer.throughput_bins, 4)

    def test_index_is_a_bijection(self) -> None:
        indices = {
            self.discretizer.index((t, b, a))
            for t in range(self.discretizer.throughput_bins)
            for b in range(self.discretizer.buffer_bins)
            for a in range(6)
        }
        self.assertEqual(indices, set(range(self.discretizer.state_count)))

    def test_rejects_bad_edges_and_values(self) -> None:
        with self.assertRaises(ValueError):
            StateDiscretizer(throughput_edges_mbps=(2.0, 1.0))
        with self.assertRaises(ValueError):
            StateDiscretizer(buffer_edges_s=())
        with self.assertRaises(ValueError):
            self.discretizer.state(-1.0, 0.0, 0)
        with self.assertRaises(ValueError):
            self.discretizer.state(1.0, 0.0, 6)


class HyperparameterTests(unittest.TestCase):
    def test_epsilon_decays_linearly_then_holds(self) -> None:
        params = QLearningHyperparameters(
            episodes=100, epsilon_start=1.0, epsilon_end=0.0, epsilon_decay_fraction=0.5
        )
        self.assertAlmostEqual(params.epsilon(0), 1.0)
        self.assertAlmostEqual(params.epsilon(25), 0.5)
        self.assertAlmostEqual(params.epsilon(50), 0.0)
        self.assertAlmostEqual(params.epsilon(99), 0.0)

    def test_round_trips_through_dict(self) -> None:
        params = QLearningHyperparameters(learning_rate=0.3, buffer_edges_s=[3, 9])
        self.assertEqual(QLearningHyperparameters.from_dict(params.to_dict()), params)

    def test_rejects_invalid_values(self) -> None:
        for kwargs in (
            {"learning_rate": 0.0},
            {"discount": 1.5},
            {"epsilon_start": 0.1, "epsilon_end": 0.5},
            {"episodes": 0},
            {"throughput_history": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                QLearningHyperparameters(**kwargs)


class AgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = tiny_config()
        self.params = QLearningHyperparameters(
            learning_rate=0.5,
            discount=0.9,
            throughput_edges_mbps=(1.0,),
            buffer_edges_s=(5.0,),
            episodes=10,
        )
        self.agent = QLearningAgent(self.params, self.config)

    def test_update_moves_value_towards_target(self) -> None:
        self.agent.q_values[1] = [2.0, 4.0]
        td_error = self.agent.update(0, 1, reward=1.0, next_state_index=1)

        # target = 1 + 0.9 * max(2, 4) = 4.6; Q = 0 + 0.5 * 4.6
        self.assertAlmostEqual(td_error, 4.6)
        self.assertAlmostEqual(self.agent.q_values[0][1], 2.3)
        self.assertEqual(self.agent.visit_counts[0][1], 1)

    def test_terminal_update_ignores_next_state(self) -> None:
        self.agent.q_values[1] = [100.0, 100.0]
        self.agent.update(0, 0, reward=-2.0, next_state_index=None)
        self.assertAlmostEqual(self.agent.q_values[0][0], -1.0)

    def test_greedy_prefers_best_and_breaks_ties_low(self) -> None:
        self.assertEqual(self.agent.greedy_action(0), 0)
        self.agent.q_values[0] = [0.1, 0.7]
        self.assertEqual(self.agent.greedy_action(0), 1)

    def test_epsilon_extremes(self) -> None:
        self.agent.q_values[0] = [0.0, 1.0]
        rng = random.Random(0)
        self.assertTrue(
            all(self.agent.epsilon_greedy_action(0, 0.0, rng) == 1 for _ in range(50))
        )
        explored = {self.agent.epsilon_greedy_action(0, 1.0, rng) for _ in range(50)}
        self.assertEqual(explored, {0, 1})

    def test_observe_uses_harmonic_mean_and_resets(self) -> None:
        agent = QLearningAgent(
            QLearningHyperparameters(
                throughput_history=2, throughput_edges_mbps=(1.0,), buffer_edges_s=(5.0,)
            ),
            self.config,
        )
        environment = ABREnvironment(tiny_trace(), tiny_video(), self.config)
        first = environment.step(0)
        slow = first.__class__(**{**first.__dict__, "measured_throughput_mbps": 0.6})
        fast = first.__class__(**{**first.__dict__, "measured_throughput_mbps": 3.0})
        agent.begin_episode()
        agent.observe(0.0, None)
        agent.observe(5.0, slow)
        state = agent.observe(5.0, fast)
        # harmonic mean of 0.6 and 3.0 is 1.0 -> upper throughput bin
        self.assertEqual(
            state, agent.discretizer.index(agent.discretizer.state(1.0, 5.0, 0))
        )

    def test_learns_to_avoid_rebuffering_on_a_slow_link(self) -> None:
        # 0.5 Mbit/s cannot sustain 750 Kbit/s, so the low bitrate is better.
        params = QLearningHyperparameters(
            episodes=400,
            learning_rate=0.2,
            discount=0.9,
            throughput_edges_mbps=(1.0,),
            buffer_edges_s=(5.0,),
        )
        agent, returns = train_agent(
            [tiny_trace(0.5)], tiny_video(), params, seed=1, config=self.config
        )
        self.assertEqual(len(returns), 400)
        agent.begin_episode()
        metrics = run_episode(
            ABREnvironment(tiny_trace(0.5), tiny_video(), self.config),
            agent.select_action,
        )
        self.assertAlmostEqual(metrics.average_bitrate_kbps, 300.0)

    def test_training_is_reproducible_for_a_seed(self) -> None:
        traces = [tiny_trace(0.5), tiny_trace(2.0)]
        first, first_returns = train_agent(
            traces, tiny_video(), self.params, seed=7, config=self.config
        )
        second, second_returns = train_agent(
            traces, tiny_video(), self.params, seed=7, config=self.config
        )
        self.assertEqual(first.q_values, second.q_values)
        self.assertEqual(first_returns, second_returns)

    def test_save_and_load_round_trip(self) -> None:
        self.agent.q_values[3] = [1.5, -2.0]
        self.agent.visit_counts[3] = [4, 1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q.json"
            self.agent.save(path)
            loaded = QLearningAgent.load(path, self.config)
        self.assertEqual(loaded.q_values, self.agent.q_values)
        self.assertEqual(loaded.visit_counts, self.agent.visit_counts)
        self.assertEqual(loaded.hyperparameters, self.params)

    def test_load_rejects_other_ladder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q.json"
            self.agent.save(path)
            with self.assertRaises(ValueError):
                QLearningAgent.load(path, SimulatorConfig())


if __name__ == "__main__":
    unittest.main()
