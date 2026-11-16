"""Shared episode metrics for baseline and future learning policies."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .environment import ABREnvironment, StepResult


ActionSelector = Callable[[float, StepResult | None], int]


@dataclass(frozen=True)
class EpisodeMetrics:
    trace_name: str
    chunk_count: int
    total_qoe: float
    mean_chunk_qoe: float
    average_bitrate_kbps: float
    total_rebuffer_s: float
    switch_count: int
    total_bitrate_change_kbps: int
    average_bitrate_change_kbps: float


@dataclass(frozen=True)
class AggregateMetrics:
    episode_count: int
    chunk_count: int
    mean_episode_qoe: float
    mean_chunk_qoe: float
    average_bitrate_kbps: float
    mean_rebuffer_s_per_episode: float
    mean_switches_per_episode: float
    average_bitrate_change_kbps: float


def run_episode(
    environment: ABREnvironment,
    select_action: ActionSelector,
    *,
    start_index: int = 1,
) -> EpisodeMetrics:
    """Run one complete video and calculate the promised comparison metrics."""

    environment.reset(start_index=start_index)
    previous_result: StepResult | None = None
    previous_bitrate_kbps = environment.config.bitrates_kbps[
        environment.config.initial_bitrate_index
    ]

    chunk_count = 0
    total_qoe = 0.0
    total_bitrate_kbps = 0
    total_rebuffer_s = 0.0
    switch_count = 0
    total_bitrate_change_kbps = 0

    while True:
        action_index = select_action(environment.buffer_s, previous_result)
        result = environment.step(action_index)

        bitrate_change_kbps = abs(result.bitrate_kbps - previous_bitrate_kbps)
        if bitrate_change_kbps:
            switch_count += 1

        chunk_count += 1
        total_qoe += result.reward
        total_bitrate_kbps += result.bitrate_kbps
        total_rebuffer_s += result.rebuffer_s
        total_bitrate_change_kbps += bitrate_change_kbps

        previous_bitrate_kbps = result.bitrate_kbps
        previous_result = result
        if result.done:
            break

    return EpisodeMetrics(
        trace_name=environment.trace.name,
        chunk_count=chunk_count,
        total_qoe=total_qoe,
        mean_chunk_qoe=total_qoe / chunk_count,
        average_bitrate_kbps=total_bitrate_kbps / chunk_count,
        total_rebuffer_s=total_rebuffer_s,
        switch_count=switch_count,
        total_bitrate_change_kbps=total_bitrate_change_kbps,
        average_bitrate_change_kbps=total_bitrate_change_kbps / chunk_count,
    )


def aggregate_metrics(episodes: Iterable[EpisodeMetrics]) -> AggregateMetrics:
    """Aggregate episode metrics, weighting chunk metrics by chunk count."""

    episode_list = tuple(episodes)
    if not episode_list:
        raise ValueError("at least one episode is required")

    episode_count = len(episode_list)
    chunk_count = sum(episode.chunk_count for episode in episode_list)
    total_qoe = sum(episode.total_qoe for episode in episode_list)
    weighted_bitrate = sum(
        episode.average_bitrate_kbps * episode.chunk_count
        for episode in episode_list
    )
    total_rebuffer = sum(episode.total_rebuffer_s for episode in episode_list)
    total_switches = sum(episode.switch_count for episode in episode_list)
    total_bitrate_change = sum(
        episode.total_bitrate_change_kbps for episode in episode_list
    )

    return AggregateMetrics(
        episode_count=episode_count,
        chunk_count=chunk_count,
        mean_episode_qoe=total_qoe / episode_count,
        mean_chunk_qoe=total_qoe / chunk_count,
        average_bitrate_kbps=weighted_bitrate / chunk_count,
        mean_rebuffer_s_per_episode=total_rebuffer / episode_count,
        mean_switches_per_episode=total_switches / episode_count,
        average_bitrate_change_kbps=total_bitrate_change / chunk_count,
    )
