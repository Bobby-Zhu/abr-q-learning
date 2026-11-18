"""Shared helpers for training, tuning, and final evaluation scripts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict
import math
from pathlib import Path

from .config import SimulatorConfig
from .dataset import load_trace_manifest
from .environment import ABREnvironment, StepResult
from .evaluation import AggregateMetrics, EpisodeMetrics, aggregate_metrics, run_episode
from .policies import BufferBasedPolicy, FixedBitratePolicy, QLearningAgent
from .trace import NetworkTrace
from .video import VideoMetadata, load_video_metadata


SPLITS = ("train", "validation", "test")


def load_split(project_root: Path, split: str) -> tuple[NetworkTrace, ...]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    return load_trace_manifest(
        project_root / "data/splits" / f"{split}.txt", project_root / "data"
    )


def load_video(project_root: Path) -> VideoMetadata:
    return load_video_metadata(project_root / "data/video_sizes/mao")


def trace_source(trace_name: str) -> str:
    """FCC traces are named ``trace_*``; everything else is HSDPA (Norway)."""

    return "FCC" if trace_name.startswith("trace_") else "HSDPA"


def evaluate_selector(
    traces: Sequence[NetworkTrace],
    video: VideoMetadata,
    make_selector: Callable[[], Callable[[float, StepResult | None], int]],
    config: SimulatorConfig | None = None,
) -> list[EpisodeMetrics]:
    """Run one full video per trace from the start of the trace.

    ``make_selector`` is called once per episode so stateful policies start
    each video with a clean memory.
    """

    config = config or SimulatorConfig()
    return [
        run_episode(ABREnvironment(trace, video, config), make_selector())
        for trace in traces
    ]


def q_learning_selector(agent: QLearningAgent):
    def factory():
        agent.begin_episode()
        return agent.select_action

    return factory


def fixed_selector(policy: FixedBitratePolicy):
    return lambda: (lambda _buffer_s, _previous: policy.select_action())


def buffer_selector(policy: BufferBasedPolicy):
    return lambda: (lambda buffer_s, _previous: policy.select_action(buffer_s))


def summarize(episodes: Sequence[EpisodeMetrics]) -> dict:
    """Aggregate metrics overall and per trace source."""

    summary = {"all": asdict(aggregate_metrics(episodes))}
    for source in ("FCC", "HSDPA"):
        subset = [e for e in episodes if trace_source(e.trace_name) == source]
        if subset:
            summary[source] = asdict(aggregate_metrics(subset))
    return summary


def mean_and_std(values: Sequence[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


__all__ = [
    "AggregateMetrics",
    "SPLITS",
    "buffer_selector",
    "evaluate_selector",
    "fixed_selector",
    "load_split",
    "load_video",
    "mean_and_std",
    "q_learning_selector",
    "summarize",
    "trace_source",
]
