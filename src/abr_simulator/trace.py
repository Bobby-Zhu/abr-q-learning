"""Load the two-column bandwidth traces used by the simulator."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NetworkTrace:
    """A sequence of timestamps (seconds) and bandwidths (Mbit/s)."""

    name: str
    times_s: tuple[float, ...]
    bandwidth_mbps: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.times_s) != len(self.bandwidth_mbps):
            raise ValueError("trace timestamps and bandwidths have different lengths")
        if len(self.times_s) < 2:
            raise ValueError("a trace must contain at least two samples")
        if any(current <= previous for previous, current in zip(self.times_s, self.times_s[1:])):
            raise ValueError("trace timestamps must be strictly increasing")
        if any(value < 0 for value in self.bandwidth_mbps):
            raise ValueError("trace bandwidths must be non-negative")
        if not any(value > 0 for value in self.bandwidth_mbps):
            raise ValueError("a trace must contain at least one positive bandwidth")


def load_trace(path: str | Path) -> NetworkTrace:
    """Load one whitespace-separated ``timestamp bandwidth`` trace file."""

    trace_path = Path(path)
    times: list[float] = []
    bandwidths: list[float] = []

    with trace_path.open(encoding="utf-8") as trace_file:
        for line_number, raw_line in enumerate(trace_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"{trace_path}:{line_number}: expected two columns")
            try:
                timestamp, bandwidth = float(fields[0]), float(fields[1])
            except ValueError as error:
                raise ValueError(
                    f"{trace_path}:{line_number}: timestamp and bandwidth must be numbers"
                ) from error
            times.append(timestamp)
            bandwidths.append(bandwidth)

    return NetworkTrace(trace_path.name, tuple(times), tuple(bandwidths))


def load_traces(directory: str | Path) -> tuple[NetworkTrace, ...]:
    """Load every regular file in a directory in deterministic name order."""

    trace_directory = Path(directory)
    paths = sorted(path for path in trace_directory.iterdir() if path.is_file())
    if not paths:
        raise ValueError(f"no trace files found in {trace_directory}")
    return tuple(load_trace(path) for path in paths)
