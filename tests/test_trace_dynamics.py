"""Check the scientific definitions against hand-calculable traces."""

import math
import unittest

from abr_simulator import NetworkTrace
from abr_simulator.trace_dynamics import (
    describe_trace,
    forecast_errors,
    harmonic_prediction,
    pearson_correlation,
    resample_bandwidth,
)


class TraceDynamicsTests(unittest.TestCase):
    def test_irregular_intervals_are_time_weighted_and_first_row_is_unused(self) -> None:
        trace = NetworkTrace(
            "offset-irregular",
            (100.0, 101.0, 104.0, 105.0, 110.0, 112.0),
            (999.0, 2.0, 8.0, 0.0, 4.0, 9.0),
        )
        self.assertEqual(resample_bandwidth(trace), (5.2, 4.0))
        description = describe_trace(trace)
        self.assertEqual(description["discarded_tail_s"], 2.0)
        self.assertAlmostEqual(description["duration_weighted_bandwidth_mbps"], 64 / 12)
        self.assertAlmostEqual(description["zero_bandwidth_time_fraction"], 1 / 12)

    def test_equal_time_grid_removes_native_sampling_frequency_difference(self) -> None:
        levels = (1.0, 4.0, 2.0, 8.0, 3.0)
        coarse = NetworkTrace(
            "five-second-sampling",
            tuple(float(index * 5) for index in range(6)),
            (99.0,) + levels,
        )
        fine = NetworkTrace(
            "one-second-sampling",
            tuple(float(index) for index in range(26)),
            (99.0,) + tuple(level for level in levels for _ in range(5)),
        )
        self.assertEqual(resample_bandwidth(coarse), resample_bandwidth(fine))
        for metric in ("lag1_correlation", "coefficient_of_variation", "forecast_h3_nmae"):
            self.assertEqual(describe_trace(coarse)[metric], describe_trace(fine)[metric])

    def test_constant_trace_has_zero_forecast_error_and_undefined_correlation(self) -> None:
        trace = NetworkTrace("constant", (0.0, 25.0), (100.0, 2.0))
        stats = describe_trace(trace)
        self.assertEqual(stats["mean_bandwidth_mbps"], 2.0)
        self.assertEqual(stats["coefficient_of_variation"], 0.0)
        self.assertEqual(stats["mean_absolute_change_mbps"], 0.0)
        self.assertEqual(stats["forecast_h3_nmae"], 0.0)
        self.assertEqual(stats["forecast_target_count"], 2)
        self.assertIsNone(stats["lag1_correlation"])

    def test_forecasters_share_targets_and_use_only_past_windows(self) -> None:
        errors = forecast_errors((1.0, 2.0, 4.0, 8.0, 16.0))
        self.assertEqual(errors["forecast_target_count"], 2)
        self.assertAlmostEqual(errors["forecast_h1_mae_mbps"], 6.0)
        self.assertAlmostEqual(errors["forecast_h1_nmae"], 30 / 31)
        self.assertAlmostEqual(errors["forecast_h3_mae_mbps"], 66 / 7)
        self.assertLess(errors["forecast_h3_bias_mbps"], 0.0)

    def test_zero_capacity_uses_finite_harmonic_mean_and_safe_normalization(self) -> None:
        self.assertAlmostEqual(harmonic_prediction((0.0, 1.0, 2.0)), 3e-9, places=15)
        stats = forecast_errors((0.0,) * 5)
        self.assertIsNone(stats["forecast_h3_nmae"])
        self.assertTrue(math.isfinite(stats["forecast_h3_mae_mbps"]))

    def test_correlation_and_short_trace_missing_values(self) -> None:
        self.assertEqual(pearson_correlation((1, 3, 1, 3), (3, 1, 3, 1)), -1.0)
        self.assertIsNone(pearson_correlation((1, 2), (2, 1)))
        short = NetworkTrace("too-short", (0.0, 4.0), (1.0, 2.0))
        stats = describe_trace(short)
        self.assertEqual(stats["window_count"], 0)
        self.assertIsNone(stats["forecast_h1_mae_mbps"])
        self.assertIsNone(stats["lag1_correlation"])

    def test_invalid_windows_and_nonfinite_samples_are_rejected(self) -> None:
        trace = NetworkTrace("valid", (0.0, 5.0), (1.0, 2.0))
        for window_s in (0, -1, math.inf, math.nan):
            with self.subTest(window_s=window_s), self.assertRaises(ValueError):
                resample_bandwidth(trace, window_s)
        with self.assertRaises(ValueError):
            resample_bandwidth(NetworkTrace("nonfinite", (0.0, 5.0), (1.0, math.inf)))


if __name__ == "__main__":
    unittest.main()
