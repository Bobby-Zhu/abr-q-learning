"""Quality-of-Experience reward shared by all future policies."""

from .config import SimulatorConfig


def linear_qoe_reward(
    bitrate_kbps: int,
    previous_bitrate_kbps: int,
    rebuffer_s: float,
    config: SimulatorConfig,
) -> float:
    """Compute MERINA/Pensieve-style linear QoE for one downloaded chunk."""

    quality_mbps = bitrate_kbps / 1000.0
    switch_mbps = abs(bitrate_kbps - previous_bitrate_kbps) / 1000.0
    return (
        config.quality_weight * quality_mbps
        - config.rebuffer_penalty * rebuffer_s
        - config.smoothness_penalty * switch_mbps
    )
