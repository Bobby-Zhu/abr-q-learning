"""Configuration shared by the ABR simulator and every policy."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulatorConfig:
    """Physical and QoE constants for one simulation run.

    Defaults match the linear-QoE setup in the MERINA repository.  Keeping
    them in one immutable object prevents policies from silently using
    different experiment rules.
    """

    bitrates_kbps: tuple[int, ...] = (300, 750, 1200, 1850, 2850, 4300)
    chunk_duration_s: float = 4.0
    max_buffer_s: float = 60.0
    drain_quantum_s: float = 0.5
    payload_fraction: float = 0.95
    link_rtt_s: float = 0.080

    # Linear QoE: bitrate_Mbps - 4.3 * rebuffer_s - bitrate_change_Mbps
    quality_weight: float = 1.0
    rebuffer_penalty: float = 4.3
    smoothness_penalty: float = 1.0
    initial_bitrate_index: int = 1

    def __post_init__(self) -> None:
        if not self.bitrates_kbps or any(rate <= 0 for rate in self.bitrates_kbps):
            raise ValueError("bitrates_kbps must contain positive values")
        if tuple(sorted(self.bitrates_kbps)) != self.bitrates_kbps:
            raise ValueError("bitrates_kbps must be strictly increasing")
        if len(set(self.bitrates_kbps)) != len(self.bitrates_kbps):
            raise ValueError("bitrates_kbps must not contain duplicates")
        if self.chunk_duration_s <= 0:
            raise ValueError("chunk_duration_s must be positive")
        if self.max_buffer_s < self.chunk_duration_s:
            raise ValueError("max_buffer_s must hold at least one chunk")
        if self.drain_quantum_s <= 0:
            raise ValueError("drain_quantum_s must be positive")
        if not 0 < self.payload_fraction <= 1:
            raise ValueError("payload_fraction must be in (0, 1]")
        if self.link_rtt_s < 0:
            raise ValueError("link_rtt_s must be non-negative")
        if not 0 <= self.initial_bitrate_index < len(self.bitrates_kbps):
            raise ValueError("initial_bitrate_index is outside the bitrate ladder")
