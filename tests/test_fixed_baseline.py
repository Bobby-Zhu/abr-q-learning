"""Tests for the fixed-bitrate policy and deterministic data splits."""

from pathlib import Path
import re
import unittest

from abr_simulator import (
    ABREnvironment,
    FixedBitratePolicy,
    NetworkTrace,
    SimulatorConfig,
    VideoMetadata,
    load_trace_manifest,
    run_episode,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
SPLIT_ROOT = DATA_ROOT / "splits"


def recording_id(entry: str) -> str:
    return re.sub(r"_part\d+\.log$", "", Path(entry).name)


def read_entries(split_name: str) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in (SPLIT_ROOT / f"{split_name}.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    )


class FixedBitratePolicyTests(unittest.TestCase):
    def test_default_policy_always_selects_1200_kbps(self) -> None:
        policy = FixedBitratePolicy()

        self.assertEqual(policy.bitrate_index, 2)
        self.assertEqual(policy.bitrate_kbps, 1200)
        self.assertTrue(all(policy.select_action() == 2 for _ in range(100)))

    def test_every_configured_index_is_supported(self) -> None:
        config = SimulatorConfig()

        for index, bitrate in enumerate(config.bitrates_kbps):
            with self.subTest(index=index):
                policy = FixedBitratePolicy(config, index)
                self.assertEqual(policy.select_action(), index)
                self.assertEqual(policy.bitrate_kbps, bitrate)

    def test_invalid_index_is_rejected(self) -> None:
        for index in (-1, 6):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    FixedBitratePolicy(bitrate_index=index)

    def test_fixed_episode_metrics_match_known_run(self) -> None:
        config = SimulatorConfig(payload_fraction=1.0, link_rtt_s=0.0)
        trace = NetworkTrace(
            "constant-8-mbps",
            tuple(float(value) for value in range(12)),
            (8.0,) * 12,
        )
        video = VideoMetadata(
            tuple((1_000_000, 1_000_000) for _ in config.bitrates_kbps)
        )
        policy = FixedBitratePolicy(config)

        metrics = run_episode(
            ABREnvironment(trace, video, config),
            lambda _buffer_s, _previous: policy.select_action(),
        )

        self.assertEqual(metrics.chunk_count, 2)
        self.assertAlmostEqual(metrics.total_qoe, -2.35)
        self.assertAlmostEqual(metrics.average_bitrate_kbps, 1200.0)
        self.assertAlmostEqual(metrics.total_rebuffer_s, 1.0)
        self.assertEqual(metrics.switch_count, 1)
        self.assertEqual(metrics.total_bitrate_change_kbps, 450)


class DataSplitTests(unittest.TestCase):
    def test_split_sizes_and_source_balance(self) -> None:
        train = read_entries("train")
        validation = read_entries("validation")
        test = read_entries("test")

        self.assertEqual(len(train), 166)
        self.assertEqual(len(validation), 129)
        self.assertEqual(len(test), 133)
        for entries, expected_fcc, expected_hsdpa in ((validation, 43, 86), (test, 45, 88)):
            fcc_count = sum(Path(entry).name.startswith("trace_") for entry in entries)
            self.assertEqual(fcc_count, expected_fcc)
            self.assertEqual(len(entries) - fcc_count, expected_hsdpa)

    def test_no_recording_spans_two_splits(self) -> None:
        recordings = {
            split_name: {recording_id(entry) for entry in read_entries(split_name)}
            for split_name in ("train", "validation", "test")
        }

        self.assertTrue(recordings["train"].isdisjoint(recordings["validation"]))
        self.assertTrue(recordings["train"].isdisjoint(recordings["test"]))
        self.assertTrue(recordings["validation"].isdisjoint(recordings["test"]))

    def test_splits_are_disjoint_and_cover_imported_data(self) -> None:
        train = set(read_entries("train"))
        validation = set(read_entries("validation"))
        test = set(read_entries("test"))

        self.assertTrue(train.isdisjoint(validation))
        self.assertTrue(train.isdisjoint(test))
        self.assertTrue(validation.isdisjoint(test))

        imported_train = {
            str(path.relative_to(DATA_ROOT))
            for path in (DATA_ROOT / "traces/fcc_hsdpa/train").iterdir()
            if path.is_file()
        }
        imported_holdout = {
            str(path.relative_to(DATA_ROOT))
            for path in (DATA_ROOT / "traces/fcc_hsdpa/holdout").iterdir()
            if path.is_file()
        }
        self.assertEqual(train, imported_train)
        self.assertEqual(validation | test, imported_holdout)

    def test_manifest_loader_reads_declared_splits(self) -> None:
        for split_name, expected_count in (
            ("train", 166),
            ("validation", 129),
            ("test", 133),
        ):
            with self.subTest(split=split_name):
                traces = load_trace_manifest(
                    SPLIT_ROOT / f"{split_name}.txt",
                    DATA_ROOT,
                )
                self.assertEqual(len(traces), expected_count)


if __name__ == "__main__":
    unittest.main()
