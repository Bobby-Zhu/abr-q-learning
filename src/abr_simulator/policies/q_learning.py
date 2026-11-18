"""Tabular Q-learning for adaptive-bitrate selection.

The agent observes a discretised state made from three measurements that are
available to a real player before it requests the next chunk:

* the throughput measured while downloading the previous chunk (optionally
  smoothed as a harmonic mean over the last few chunks),
* the playback buffer level, and
* the bitrate index chosen for the previous chunk.

Its action is the bitrate index of the next chunk, and its reward is the
shared linear QoE returned by :class:`ABREnvironment`.  Training uses an
epsilon-greedy behaviour policy; evaluation is purely greedy.  Because
Q-learning bootstraps from ``max_a Q(s', a)`` it learns the greedy policy
even while exploring (off-policy learning).
"""

from __future__ import annotations

from bisect import bisect_right
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import random

from ..config import SimulatorConfig
from ..environment import ABREnvironment, StepResult
from ..trace import NetworkTrace


DEFAULT_THROUGHPUT_EDGES_MBPS = (0.5, 1.0, 1.5, 2.2, 3.2)
DEFAULT_BUFFER_EDGES_S = (4.0, 8.0, 12.0, 18.0, 26.0)


def _check_edges(name: str, edges: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(edge) for edge in edges)
    if not values:
        raise ValueError(f"{name} must contain at least one edge")
    if any(not math.isfinite(edge) or edge <= 0 for edge in values):
        raise ValueError(f"{name} must contain finite positive values")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    return values


@dataclass(frozen=True)
class StateDiscretizer:
    """Map raw observations to a small integer state tuple.

    ``throughput_edges_mbps`` with *k* edges gives *k + 1* measured-throughput
    bins plus one extra bin (index ``k + 1``) for the first chunk, before any
    throughput has been measured.  Buffer edges work the same way without the
    extra bin.  A value equal to an edge falls into the upper bin.
    """

    config: SimulatorConfig = field(default_factory=SimulatorConfig)
    throughput_edges_mbps: tuple[float, ...] = DEFAULT_THROUGHPUT_EDGES_MBPS
    buffer_edges_s: tuple[float, ...] = DEFAULT_BUFFER_EDGES_S

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "throughput_edges_mbps",
            _check_edges("throughput_edges_mbps", self.throughput_edges_mbps),
        )
        object.__setattr__(
            self,
            "buffer_edges_s",
            _check_edges("buffer_edges_s", self.buffer_edges_s),
        )

    @property
    def throughput_bins(self) -> int:
        """Measured bins plus the 'not yet measured' bin."""

        return len(self.throughput_edges_mbps) + 2

    @property
    def buffer_bins(self) -> int:
        return len(self.buffer_edges_s) + 1

    @property
    def state_count(self) -> int:
        return (
            self.throughput_bins
            * self.buffer_bins
            * len(self.config.bitrates_kbps)
        )

    def state(
        self,
        throughput_mbps: float | None,
        buffer_s: float,
        last_bitrate_index: int,
    ) -> tuple[int, int, int]:
        if throughput_mbps is None:
            throughput_bin = len(self.throughput_edges_mbps) + 1
        else:
            if not math.isfinite(throughput_mbps) or throughput_mbps < 0:
                raise ValueError("throughput_mbps must be finite and non-negative")
            throughput_bin = bisect_right(self.throughput_edges_mbps, throughput_mbps)
        if not math.isfinite(buffer_s) or buffer_s < 0:
            raise ValueError("buffer_s must be finite and non-negative")
        if not 0 <= last_bitrate_index < len(self.config.bitrates_kbps):
            raise ValueError("last_bitrate_index is outside the bitrate ladder")
        return (
            throughput_bin,
            bisect_right(self.buffer_edges_s, buffer_s),
            last_bitrate_index,
        )

    def index(self, state: tuple[int, int, int]) -> int:
        throughput_bin, buffer_bin, bitrate_index = state
        return (
            throughput_bin * self.buffer_bins + buffer_bin
        ) * len(self.config.bitrates_kbps) + bitrate_index


@dataclass(frozen=True)
class QLearningHyperparameters:
    """Everything that defines one training run, in one serialisable place."""

    learning_rate: float = 0.1
    discount: float = 0.9
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_fraction: float = 0.8
    episodes: int = 20_000
    throughput_history: int = 1
    throughput_edges_mbps: tuple[float, ...] = DEFAULT_THROUGHPUT_EDGES_MBPS
    buffer_edges_s: tuple[float, ...] = DEFAULT_BUFFER_EDGES_S

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "throughput_edges_mbps", tuple(self.throughput_edges_mbps)
        )
        object.__setattr__(self, "buffer_edges_s", tuple(self.buffer_edges_s))
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0 <= self.discount <= 1:
            raise ValueError("discount must be in [0, 1]")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("require 0 <= epsilon_end <= epsilon_start <= 1")
        if not 0 < self.epsilon_decay_fraction <= 1:
            raise ValueError("epsilon_decay_fraction must be in (0, 1]")
        if self.episodes < 1:
            raise ValueError("episodes must be positive")
        if self.throughput_history < 1:
            raise ValueError("throughput_history must be at least 1")

    def epsilon(self, episode: int) -> float:
        """Linearly decay exploration, then hold it at ``epsilon_end``."""

        decay_episodes = max(1, round(self.episodes * self.epsilon_decay_fraction))
        progress = min(1.0, episode / decay_episodes)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def to_dict(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "discount": self.discount,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay_fraction": self.epsilon_decay_fraction,
            "episodes": self.episodes,
            "throughput_history": self.throughput_history,
            "throughput_edges_mbps": list(self.throughput_edges_mbps),
            "buffer_edges_s": list(self.buffer_edges_s),
        }

    @classmethod
    def from_dict(cls, values: dict) -> "QLearningHyperparameters":
        return cls(**values)


class QLearningAgent:
    """A Q-table plus the per-episode observation memory needed to act."""

    def __init__(
        self,
        hyperparameters: QLearningHyperparameters | None = None,
        config: SimulatorConfig | None = None,
    ) -> None:
        self.config = config or SimulatorConfig()
        self.hyperparameters = hyperparameters or QLearningHyperparameters()
        self.discretizer = StateDiscretizer(
            self.config,
            self.hyperparameters.throughput_edges_mbps,
            self.hyperparameters.buffer_edges_s,
        )
        self.action_count = len(self.config.bitrates_kbps)
        self.q_values = [
            [0.0] * self.action_count for _ in range(self.discretizer.state_count)
        ]
        self.visit_counts = [
            [0] * self.action_count for _ in range(self.discretizer.state_count)
        ]
        self._history: deque[float] = deque(
            maxlen=self.hyperparameters.throughput_history
        )

    # ----- observation -------------------------------------------------
    def begin_episode(self) -> None:
        """Forget throughput measurements from the previous video."""

        self._history.clear()

    def observe(self, buffer_s: float, previous: StepResult | None) -> int:
        """Return the state index for the decision about to be made.

        Call exactly once per chunk, in order, starting after
        :meth:`begin_episode`.
        """

        if previous is None:
            self._history.clear()
            throughput = None
            last_bitrate = self.config.initial_bitrate_index
        else:
            self._history.append(previous.measured_throughput_mbps)
            throughput = self._smoothed_throughput()
            last_bitrate = previous.action_index
        return self.discretizer.index(
            self.discretizer.state(throughput, buffer_s, last_bitrate)
        )

    def _smoothed_throughput(self) -> float:
        # Harmonic mean, as in common rate-based ABR estimators: robust to
        # a single unusually fast download.
        return len(self._history) / sum(1.0 / max(value, 1e-9) for value in self._history)

    # ----- acting ------------------------------------------------------
    def greedy_action(self, state_index: int) -> int:
        """Highest-valued action; ties go to the lower (safer) bitrate."""

        row = self.q_values[state_index]
        best = max(row)
        return row.index(best)

    def epsilon_greedy_action(
        self, state_index: int, epsilon: float, rng: random.Random
    ) -> int:
        if rng.random() < epsilon:
            return rng.randrange(self.action_count)
        return self.greedy_action(state_index)

    def select_action(self, buffer_s: float, previous: StepResult | None) -> int:
        """Greedy action selector compatible with ``run_episode``."""

        return self.greedy_action(self.observe(buffer_s, previous))

    # ----- learning ----------------------------------------------------
    def update(
        self,
        state_index: int,
        action: int,
        reward: float,
        next_state_index: int | None,
    ) -> float:
        """Apply one Q-learning update and return the TD error.

        ``next_state_index`` is ``None`` for the terminal chunk, whose target
        is just the immediate reward.
        """

        target = reward
        if next_state_index is not None:
            target += self.hyperparameters.discount * max(
                self.q_values[next_state_index]
            )
        td_error = target - self.q_values[state_index][action]
        self.q_values[state_index][action] += (
            self.hyperparameters.learning_rate * td_error
        )
        self.visit_counts[state_index][action] += 1
        return td_error

    def train_episode(
        self,
        environment: ABREnvironment,
        epsilon: float,
        rng: random.Random,
        *,
        start_index: int = 1,
    ) -> float:
        """Play one video with exploration, learning online. Returns total QoE."""

        environment.reset(start_index=start_index)
        self.begin_episode()
        state = self.observe(environment.buffer_s, None)
        total_reward = 0.0
        while True:
            action = self.epsilon_greedy_action(state, epsilon, rng)
            result = environment.step(action)
            total_reward += result.reward
            if result.done:
                self.update(state, action, result.reward, None)
                return total_reward
            next_state = self.observe(result.buffer_s, result)
            self.update(state, action, result.reward, next_state)
            state = next_state

    # ----- persistence -------------------------------------------------
    def visited_state_fraction(self) -> float:
        visited = sum(1 for row in self.visit_counts if any(row))
        return visited / len(self.visit_counts)

    def to_dict(self) -> dict:
        return {
            "format": "abr-q-table-v1",
            "bitrates_kbps": list(self.config.bitrates_kbps),
            "hyperparameters": self.hyperparameters.to_dict(),
            "q_values": self.q_values,
            "visit_counts": self.visit_counts,
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w", encoding="utf-8") as output:
            json.dump(self.to_dict(), output)

    @classmethod
    def load(
        cls, path: str | Path, config: SimulatorConfig | None = None
    ) -> "QLearningAgent":
        with Path(path).open(encoding="utf-8") as source:
            data = json.load(source)
        if data.get("format") != "abr-q-table-v1":
            raise ValueError(f"{path} is not an abr-q-table-v1 file")
        agent = cls(QLearningHyperparameters.from_dict(data["hyperparameters"]), config)
        if tuple(data["bitrates_kbps"]) != agent.config.bitrates_kbps:
            raise ValueError("saved Q-table uses a different bitrate ladder")
        if len(data["q_values"]) != len(agent.q_values):
            raise ValueError("saved Q-table has the wrong number of states")
        agent.q_values = [list(map(float, row)) for row in data["q_values"]]
        agent.visit_counts = [list(map(int, row)) for row in data["visit_counts"]]
        return agent


def train_agent(
    traces: Sequence[NetworkTrace],
    video,
    hyperparameters: QLearningHyperparameters,
    *,
    seed: int = 0,
    config: SimulatorConfig | None = None,
    log_every: int = 0,
    log=print,
) -> tuple[QLearningAgent, list[float]]:
    """Train a fresh agent on randomly chosen traces and start positions.

    Every episode samples one training trace and a random start position in
    it, so the agent sees many different network conditions.  Returns the
    agent and the per-episode training QoE (with exploration).
    """

    if not traces:
        raise ValueError("at least one training trace is required")
    config = config or SimulatorConfig()
    rng = random.Random(seed)
    agent = QLearningAgent(hyperparameters, config)
    environments = [ABREnvironment(trace, video, config) for trace in traces]
    returns: list[float] = []

    for episode in range(hyperparameters.episodes):
        environment = environments[rng.randrange(len(environments))]
        start_index = rng.randrange(1, len(environment.trace.times_s))
        epsilon = hyperparameters.epsilon(episode)
        returns.append(
            agent.train_episode(environment, epsilon, rng, start_index=start_index)
        )
        if log_every and (episode + 1) % log_every == 0:
            recent = returns[-log_every:]
            log(
                f"episode {episode + 1}/{hyperparameters.episodes}  "
                f"epsilon={epsilon:.3f}  "
                f"mean training QoE (last {log_every})={sum(recent) / len(recent):.2f}"
            )
    return agent, returns
