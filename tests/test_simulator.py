"""Unit and integration tests for Stage 1 of the simulator."""

from pathlib import Path
import unittest

from abr_simulator import (
    ABREnvironment,
    NetworkTrace,
    SimulatorConfig,
    VideoMetadata,
    load_trace,
    load_traces,
    load_video_metadata,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SimulatorMechanicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trace = NetworkTrace(
            "constant-8-mbps",
            tuple(float(value) for value in range(12)),
            (8.0,) * 12,
        )

    def test_download_updates_rebuffer_and_buffer(self) -> None:
        config = SimulatorConfig(
            bitrates_kbps=(300,),
            payload_fraction=1.0,
            link_rtt_s=0.0,
            initial_bitrate_index=0,
        )
        video = VideoMetadata(((1_000_000, 1_000_000),))
        environment = ABREnvironment(self.trace, video, config)

        first = environment.step(0)
        second = environment.step(0)

        self.assertAlmostEqual(first.download_time_s, 1.0)
        self.assertAlmostEqual(first.rebuffer_s, 1.0)
        self.assertAlmostEqual(first.buffer_s, 4.0)
        self.assertAlmostEqual(first.reward, -4.0)
        self.assertAlmostEqual(second.rebuffer_s, 0.0)
        self.assertAlmostEqual(second.buffer_s, 7.0)
        self.assertTrue(second.done)

    def test_zero_bandwidth_interval_represents_an_outage(self) -> None:
        trace = NetworkTrace(
            "one-second-outage",
            (0.0, 1.0, 2.0, 3.0),
            (8.0, 0.0, 8.0, 8.0),
        )
        config = SimulatorConfig(
            bitrates_kbps=(300,),
            payload_fraction=1.0,
            link_rtt_s=0.0,
            initial_bitrate_index=0,
        )
        environment = ABREnvironment(trace, VideoMetadata(((1_000_000,),)), config)

        result = environment.step(0)

        self.assertAlmostEqual(result.download_time_s, 2.0)
        self.assertAlmostEqual(result.rebuffer_s, 2.0)

    def test_buffer_cap_advances_trace_during_sleep(self) -> None:
        config = SimulatorConfig(
            bitrates_kbps=(300,),
            payload_fraction=1.0,
            link_rtt_s=0.0,
            max_buffer_s=4.0,
            drain_quantum_s=0.5,
            initial_bitrate_index=0,
        )
        video = VideoMetadata(((500_000, 500_000),))
        environment = ABREnvironment(self.trace, video, config)

        environment.step(0)
        second = environment.step(0)

        self.assertAlmostEqual(second.sleep_time_s, 3.5)
        self.assertAlmostEqual(second.buffer_s, 4.0)

    def test_step_after_episode_requires_reset(self) -> None:
        config = SimulatorConfig(bitrates_kbps=(300,), initial_bitrate_index=0)
        environment = ABREnvironment(
            self.trace,
            VideoMetadata(((100_000,),)),
            config,
        )
        environment.step(0)

        with self.assertRaises(RuntimeError):
            environment.step(0)


class ImportedDataTests(unittest.TestCase):
    def test_mao_video_metadata_has_six_levels_and_49_chunks(self) -> None:
        video = load_video_metadata(PROJECT_ROOT / "data/video_sizes/mao")

        self.assertEqual(video.bitrate_levels, 6)
        self.assertEqual(video.chunk_count, 49)

    def test_expected_trace_counts_are_present(self) -> None:
        train = load_traces(PROJECT_ROOT / "data/traces/fcc_hsdpa/train")
        holdout = load_traces(PROJECT_ROOT / "data/traces/fcc_hsdpa/holdout")

        self.assertEqual(len(train), 166)
        self.assertEqual(len(holdout), 262)

    def test_real_trace_completes_a_video(self) -> None:
        trace_path = sorted(
            (PROJECT_ROOT / "data/traces/fcc_hsdpa/train").iterdir()
        )[0]
        trace = load_trace(trace_path)
        video = load_video_metadata(PROJECT_ROOT / "data/video_sizes/mao")
        environment = ABREnvironment(trace, video)

        results = []
        while not results or not results[-1].done:
            results.append(environment.step(2))

        self.assertEqual(len(results), 49)
        self.assertTrue(all(result.download_time_s > 0 for result in results))
        self.assertTrue(all(result.buffer_s >= 0 for result in results))


if __name__ == "__main__":
    unittest.main()
