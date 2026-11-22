"""Descriptive bandwidth statistics on a common time grid.

These functions analyze recorded link capacity. They neither train a policy
nor change the simulator, and their forecasts use only earlier observations.
"""

from collections.abc import Sequence
import math
from statistics import fmean, median, pstdev

from .trace import NetworkTrace


def pearson_correlation(
    first: Sequence[float], second: Sequence[float]
) -> float | None:
    """Return Pearson r; constant inputs or fewer than 3 pairs give None."""

    if len(first) != len(second):
        raise ValueError("correlation inputs must have equal lengths")
    if len(first) < 3:
        return None
    first_mean, second_mean = fmean(first), fmean(second)
    first_ss = sum((value - first_mean) ** 2 for value in first)
    second_ss = sum((value - second_mean) ** 2 for value in second)
    if first_ss == 0 or second_ss == 0:
        return None
    cross = sum(
        (x - first_mean) * (y - second_mean) for x, y in zip(first, second)
    )
    return max(-1.0, min(1.0, cross / math.sqrt(first_ss * second_ss)))


def resample_bandwidth(trace: NetworkTrace, window_s: float = 5.0) -> tuple[float, ...]:
    """Time-average capacity in full, nonoverlapping windows from trace start.

    As in ABREnvironment, bandwidth[i] describes (time[i-1], time[i]];
    bandwidth[0] is not a usable interval. Integrate that piecewise-constant
    function exactly, including zero-capacity outages. Do not interpolate,
    concatenate recordings, repeat the trace, or keep a partial final window.
    """

    if not math.isfinite(window_s) or window_s <= 0:
        raise ValueError("window_s must be finite and positive")
    if any(not math.isfinite(value) for value in trace.times_s + trace.bandwidth_mbps):
        raise ValueError("trace samples must be finite")

    duration_s = trace.times_s[-1] - trace.times_s[0]
    window_count = math.floor(duration_s / window_s)
    interval_index = 1
    cursor_s = trace.times_s[0]
    windows: list[float] = []
    for index in range(window_count):
        end_s = trace.times_s[0] + (index + 1) * window_s
        integral = 0.0
        while cursor_s < end_s:
            interval_end_s = min(end_s, trace.times_s[interval_index])
            integral += (
                interval_end_s - cursor_s
            ) * trace.bandwidth_mbps[interval_index]
            cursor_s = interval_end_s
            if cursor_s >= trace.times_s[interval_index]:
                interval_index += 1
        windows.append(integral / window_s)
    return tuple(windows)


def harmonic_prediction(history: Sequence[float]) -> float:
    """Use the Q-learning agent's harmonic-mean rule, including its 1e-9 floor."""

    if not history or any(not math.isfinite(value) or value < 0 for value in history):
        raise ValueError("history must contain finite non-negative values")
    return len(history) / sum(1.0 / max(value, 1e-9) for value in history)


def forecast_errors(values: Sequence[float]) -> dict[str, int | float | None]:
    """Compare history lengths 1 and 3 on identical next-window targets.

    Both predictors start at index 3 so neither benefits from an easier set of
    target windows. NMAE divides MAE by the full trace's mean window capacity;
    it does not divide by individual (possibly zero) targets as MAPE would.
    """

    result: dict[str, int | float | None] = {"forecast_target_count": max(0, len(values) - 3)}
    mean_capacity = fmean(values) if values else 0.0
    for length in (1, 3):
        errors = [
            harmonic_prediction(values[index - length : index]) - values[index]
            for index in range(3, len(values))
        ]
        mae = fmean(abs(error) for error in errors) if errors else None
        result[f"forecast_h{length}_mae_mbps"] = mae
        result[f"forecast_h{length}_nmae"] = (
            mae / mean_capacity if mae is not None and mean_capacity > 0 else None
        )
        result[f"forecast_h{length}_bias_mbps"] = fmean(errors) if errors else None
        result[f"forecast_h{length}_overestimate_fraction"] = (
            fmean(error > 0 for error in errors) if errors else None
        )
    return result


def describe_trace(trace: NetworkTrace, window_s: float = 5.0) -> dict:
    """Summarize native sampling, aligned variation, and causal forecasts."""

    windows = resample_bandwidth(trace, window_s)
    intervals = tuple(b - a for a, b in zip(trace.times_s, trace.times_s[1:]))
    native_values = trace.bandwidth_mbps[1:]
    duration_s = sum(intervals)
    duration_mean = sum(
        duration * bandwidth for duration, bandwidth in zip(intervals, native_values)
    ) / duration_s
    mean_capacity = fmean(windows) if windows else None
    absolute_change = (
        fmean(abs(b - a) for a, b in zip(windows, windows[1:]))
        if len(windows) >= 2 else None
    )
    result = {
        "native_interval_count": len(intervals),
        "duration_s": duration_s,
        "native_median_interval_s": median(intervals),
        "native_lag1_correlation": pearson_correlation(native_values[:-1], native_values[1:]),
        "native_mean_absolute_change_mbps": (
            fmean(abs(b - a) for a, b in zip(native_values, native_values[1:]))
            if len(native_values) >= 2 else None
        ),
        "duration_weighted_bandwidth_mbps": duration_mean,
        "zero_bandwidth_time_fraction": sum(
            duration for duration, bandwidth in zip(intervals, native_values) if bandwidth == 0
        ) / duration_s,
        "window_s": window_s,
        "window_count": len(windows),
        "discarded_tail_s": duration_s - len(windows) * window_s,
        "mean_bandwidth_mbps": mean_capacity,
        "coefficient_of_variation": (
            pstdev(windows) / mean_capacity if mean_capacity and mean_capacity > 0 else None
        ),
        "lag1_correlation": pearson_correlation(windows[:-1], windows[1:]),
        "mean_absolute_change_mbps": absolute_change,
        "normalized_mean_absolute_change": (
            absolute_change / mean_capacity
            if absolute_change is not None and mean_capacity and mean_capacity > 0
            else None
        ),
    }
    result.update(forecast_errors(windows))
    return result
