"""A transparent buffer-based adaptive-bitrate baseline."""

from dataclasses import dataclass, field
import math

from ..config import SimulatorConfig


@dataclass(frozen=True)
class BufferBasedPolicy:
    """Map playback buffer occupancy linearly onto the bitrate ladder.

    The reservoir is the low-buffer safety region.  The cushion is the region
    over which actions rise from the lowest to the highest bitrate.  This is a
    deliberately simple baseline: it has no training and never reads network
    throughput or future trace values.
    """

    config: SimulatorConfig = field(default_factory=SimulatorConfig)
    reservoir_s: float = 10.0
    cushion_s: float = 40.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.reservoir_s) or self.reservoir_s < 0:
            raise ValueError("reservoir_s must be a finite non-negative value")
        if not math.isfinite(self.cushion_s) or self.cushion_s <= 0:
            raise ValueError("cushion_s must be a finite positive value")
        if self.upper_threshold_s > self.config.max_buffer_s:
            raise ValueError(
                "reservoir_s + cushion_s must not exceed the maximum buffer"
            )
        if len(self.config.bitrates_kbps) < 2:
            raise ValueError("buffer-based adaptation requires at least two bitrates")

    @property
    def upper_threshold_s(self) -> float:
        return self.reservoir_s + self.cushion_s

    def select_action(self, buffer_s: float) -> int:
        """Return a bitrate index using only the current buffer level."""

        if not math.isfinite(buffer_s) or buffer_s < 0:
            raise ValueError("buffer_s must be a finite non-negative value")

        highest_index = len(self.config.bitrates_kbps) - 1
        if buffer_s <= self.reservoir_s:
            return 0
        if buffer_s >= self.upper_threshold_s:
            return highest_index

        fraction = (buffer_s - self.reservoir_s) / self.cushion_s
        return min(highest_index - 1, int(fraction * highest_index))
