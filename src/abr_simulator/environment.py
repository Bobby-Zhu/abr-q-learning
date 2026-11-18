"""A small, deterministic adaptation of MERINA's ABR virtual player.

The download, buffer, trace-wrap, and sleep mechanics are derived from
``envs/env_log.py`` in https://github.com/confiwent/merina (MIT licensed).
The local API is deliberately policy-agnostic: an agent supplies only a
bitrate index and receives the measurements caused by that choice.
"""

from dataclasses import dataclass
import math

from .config import SimulatorConfig
from .qoe import linear_qoe_reward
from .trace import NetworkTrace
from .video import VideoMetadata


BITS_PER_BYTE = 8.0
BITS_PER_MEGABIT = 1_000_000.0


@dataclass(frozen=True)
class StepResult:
    """Measurements produced after downloading one video chunk."""

    chunk_index: int
    action_index: int
    bitrate_kbps: int
    chunk_size_bytes: int
    download_time_s: float
    sleep_time_s: float
    buffer_s: float
    rebuffer_s: float
    measured_throughput_mbps: float
    reward: float
    done: bool
    remaining_chunks: int
    next_chunk_sizes_bytes: tuple[int, ...] | None


class ABREnvironment:
    """Simulate sequential video-chunk downloads over one bandwidth trace."""

    def __init__(
        self,
        trace: NetworkTrace,
        video: VideoMetadata,
        config: SimulatorConfig | None = None,
    ) -> None:
        self.config = config or SimulatorConfig()
        if video.bitrate_levels != len(self.config.bitrates_kbps):
            raise ValueError(
                "video metadata bitrate count does not match the configured ladder"
            )
        self.video = video
        self.trace = trace
        self.reset()

    def reset(
        self,
        trace: NetworkTrace | None = None,
        *,
        start_index: int = 1,
    ) -> None:
        """Start a fresh video, optionally on another trace or trace position.

        Index 0 is not used as a starting interval because each bandwidth
        sample describes the interval ending at its corresponding timestamp,
        following the upstream MERINA/Pensieve convention.
        """

        if trace is not None:
            self.trace = trace
        if not 1 <= start_index < len(self.trace.times_s):
            raise ValueError("start_index must be between 1 and trace length - 1")

        self._trace_index = start_index
        self._last_trace_time_s = self.trace.times_s[start_index - 1]
        self._chunk_index = 0
        self._buffer_s = 0.0
        self._previous_bitrate_index = self.config.initial_bitrate_index
        self._done = False

    @property
    def buffer_s(self) -> float:
        return self._buffer_s

    @property
    def chunk_index(self) -> int:
        return self._chunk_index

    def step(self, action_index: int) -> StepResult:
        """Download the next chunk at the selected bitrate index."""

        if self._done:
            raise RuntimeError("episode is complete; call reset() before step()")
        if not 0 <= action_index < len(self.config.bitrates_kbps):
            raise ValueError("action_index is outside the bitrate ladder")

        chunk_index = self._chunk_index
        bitrate_kbps = self.config.bitrates_kbps[action_index]
        chunk_size_bytes = self.video.sizes_bytes[action_index][chunk_index]

        transfer_time_s = self._download_bytes(chunk_size_bytes)
        download_time_s = transfer_time_s + self.config.link_rtt_s
        rebuffer_s = max(download_time_s - self._buffer_s, 0.0)

        self._buffer_s = max(self._buffer_s - download_time_s, 0.0)
        self._buffer_s += self.config.chunk_duration_s

        sleep_time_s = 0.0
        if self._buffer_s > self.config.max_buffer_s:
            excess_s = self._buffer_s - self.config.max_buffer_s
            sleep_time_s = (
                math.ceil(excess_s / self.config.drain_quantum_s)
                * self.config.drain_quantum_s
            )
            self._buffer_s -= sleep_time_s
            self._advance_without_downloading(sleep_time_s)

        previous_bitrate_kbps = self.config.bitrates_kbps[
            self._previous_bitrate_index
        ]
        reward = linear_qoe_reward(
            bitrate_kbps,
            previous_bitrate_kbps,
            rebuffer_s,
            self.config,
        )

        self._previous_bitrate_index = action_index
        self._chunk_index += 1
        self._done = self._chunk_index >= self.video.chunk_count
        remaining_chunks = self.video.chunk_count - self._chunk_index
        next_sizes = (
            None
            if self._done
            else self.video.sizes_for_chunk(self._chunk_index)
        )
        measured_throughput_mbps = (
            chunk_size_bytes * BITS_PER_BYTE / download_time_s / BITS_PER_MEGABIT
        )

        return StepResult(
            chunk_index=chunk_index,
            action_index=action_index,
            bitrate_kbps=bitrate_kbps,
            chunk_size_bytes=chunk_size_bytes,
            download_time_s=download_time_s,
            sleep_time_s=sleep_time_s,
            buffer_s=self._buffer_s,
            rebuffer_s=rebuffer_s,
            measured_throughput_mbps=measured_throughput_mbps,
            reward=reward,
            done=self._done,
            remaining_chunks=remaining_chunks,
            next_chunk_sizes_bytes=next_sizes,
        )

    def _download_bytes(self, byte_count: int) -> float:
        sent_bytes = 0.0
        elapsed_s = 0.0

        while sent_bytes < byte_count:
            interval_s = self._current_interval_s()
            bandwidth_bytes_s = (
                self.trace.bandwidth_mbps[self._trace_index]
                * BITS_PER_MEGABIT
                / BITS_PER_BYTE
            )
            deliverable_bytes = (
                bandwidth_bytes_s * interval_s * self.config.payload_fraction
            )

            if bandwidth_bytes_s > 0 and sent_bytes + deliverable_bytes > byte_count:
                fractional_s = (
                    (byte_count - sent_bytes)
                    / bandwidth_bytes_s
                    / self.config.payload_fraction
                )
                elapsed_s += fractional_s
                self._last_trace_time_s += fractional_s
                # Floating-point rounding can land exactly on the interval
                # boundary; move to the next interval instead of leaving a
                # zero-length one behind.
                if self._last_trace_time_s >= self.trace.times_s[self._trace_index]:
                    self._advance_trace_interval()
                sent_bytes = float(byte_count)
            else:
                sent_bytes += deliverable_bytes
                elapsed_s += interval_s
                self._advance_trace_interval()

        return elapsed_s

    def _advance_without_downloading(self, duration_s: float) -> None:
        remaining_s = duration_s
        while remaining_s > 0:
            interval_s = self._current_interval_s()
            if interval_s > remaining_s:
                self._last_trace_time_s += remaining_s
                if self._last_trace_time_s >= self.trace.times_s[self._trace_index]:
                    self._advance_trace_interval()
                return
            remaining_s -= interval_s
            self._advance_trace_interval()

    def _current_interval_s(self) -> float:
        interval_s = self.trace.times_s[self._trace_index] - self._last_trace_time_s
        if interval_s <= 0:
            raise RuntimeError("trace interval did not advance")
        return interval_s

    def _advance_trace_interval(self) -> None:
        self._last_trace_time_s = self.trace.times_s[self._trace_index]
        self._trace_index += 1
        if self._trace_index >= len(self.trace.times_s):
            self._trace_index = 1
            self._last_trace_time_s = self.trace.times_s[0]
