"""Tests for the fixed-bitrate policy and deterministic data splits."""

from pathlib import Path
import unittest
import json

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
    def test_splits_are_disjoint_and_cover_imported_data(self) -> None:
        train = set(read_entries("train"))
        validation = set(read_entries("validation"))
        test = set(read_entries("test"))

        self.assertTrue(train.isdisjoint(validation))
        self.assertTrue(train.isdisjoint(test))
        self.assertTrue(validation.isdisjoint(test))

        imported_train = {
            path.relative_to(DATA_ROOT).as_posix()
            for path in (DATA_ROOT / "traces/fcc_hsdpa/train").iterdir()
            if path.is_file()
        }
        imported_holdout = {
            path.relative_to(DATA_ROOT).as_posix()
            for path in (DATA_ROOT / "traces/fcc_hsdpa/holdout").iterdir()
            if path.is_file()
        }
        audit = json.loads((SPLIT_ROOT / "capture_audit.json").read_text(encoding="utf-8"))
        excluded = {row["path"] for row in audit["files"] if row["duplicate_of"] is not None}
        self.assertEqual(train | validation | test | excluded, imported_train | imported_holdout)
        moved = {row["path"] for row in audit["files"]
                 if row["split"] == "train" and row["upstream_split"] == "holdout"}
        self.assertEqual(train, (imported_train | moved) - excluded)
        self.assertEqual(validation | test, imported_holdout - moved - excluded)

    def test_manifest_loader_reads_declared_splits(self) -> None:
        audit = json.loads((SPLIT_ROOT / "capture_audit.json").read_text(encoding="utf-8"))
        for split_name, counts in audit["splits"].items():
            with self.subTest(split=split_name):
                traces = load_trace_manifest(
                    SPLIT_ROOT / f"{split_name}.txt",
                    DATA_ROOT,
                )
                self.assertEqual(len(traces), counts["traces"])


if __name__ == "__main__":
    unittest.main()
