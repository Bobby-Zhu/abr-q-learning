"""A constant-bitrate reference policy."""

from dataclasses import dataclass, field

from ..config import SimulatorConfig


@dataclass(frozen=True)
class FixedBitratePolicy:
    """Select the same configured bitrate for every video chunk."""

    config: SimulatorConfig = field(default_factory=SimulatorConfig)
    bitrate_index: int = 2

    def __post_init__(self) -> None:
        if not 0 <= self.bitrate_index < len(self.config.bitrates_kbps):
            raise ValueError("bitrate_index is outside the configured ladder")

    @property
    def bitrate_kbps(self) -> int:
        return self.config.bitrates_kbps[self.bitrate_index]

    def select_action(self) -> int:
        return self.bitrate_index
