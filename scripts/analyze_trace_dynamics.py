#!/usr/bin/env python3
"""Describe FCC/HSDPA trace dynamics and link them to saved QoE differences."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
from statistics import fmean, median, stdev
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from abr_simulator.experiments import SPLITS, load_split, trace_source  # noqa: E402
from abr_simulator.trace_dynamics import describe_trace, pearson_correlation  # noqa: E402


METRICS = (
    "native_median_interval_s",
    "native_lag1_correlation",
    "duration_weighted_bandwidth_mbps",
    "zero_bandwidth_time_fraction",
    "mean_bandwidth_mbps",
    "lag1_correlation",
    "mean_absolute_change_mbps",
    "normalized_mean_absolute_change",
    "coefficient_of_variation",
    "forecast_h1_mae_mbps",
    "forecast_h3_mae_mbps",
    "forecast_h1_nmae",
    "forecast_h3_nmae",
    "forecast_h3_bias_mbps",
    "q_learning_minus_buffer_qoe",
)


def recording_id(name: str) -> str:
    return re.sub(r"_part\d+\.log$", "", name)


def quantile(values: list[float], fraction: float) -> float:
    """Linear interpolation between sorted values, including n=1."""

    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def write_csv(output_path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_saved_qoe(split: str) -> tuple[dict[str, float], dict | None]:
    """Read existing evaluation artifacts; never rerun or train an agent."""

    source = PROJECT_ROOT / "results/final" / split / "per_trace.csv"
    if not source.exists():
        return {}, None

    buffers: dict[str, float] = {}
    agents: dict[str, dict[str, float]] = defaultdict(dict)
    with source.open(newline="", encoding="utf-8") as source_file:
        for row in csv.DictReader(source_file):
            name = row["trace_name"]
            if row["policy"] == "buffer":
                if name in buffers:
                    raise ValueError(f"duplicate saved buffer result for {name}")
                buffers[name] = float(row["total_qoe"])
            elif row["policy"] == "q_learning":
                if row["seed"] in agents[name]:
                    raise ValueError(f"duplicate saved agent result for {name}")
                agents[name][row["seed"]] = float(row["total_qoe"])

    if set(buffers) != set(agents):
        raise ValueError(f"saved Q-learning and buffer trace lists differ in {source}")
    seed_sets = {tuple(sorted(runs)) for runs in agents.values()}
    if len(seed_sets) != 1:
        raise ValueError(f"inconsistent saved agent seed sets in {source}")
    deltas = {
        name: fmean(runs.values()) - buffers[name] for name, runs in agents.items()
    }
    return deltas, {
        "file": str(source.relative_to(PROJECT_ROOT)),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "seeds": [int(seed) for seed in next(iter(seed_sets))],
    }


def recording_means(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["source"], row["recording_id"])].append(row)

    recordings = []
    for (split, source, recording), slices in sorted(groups.items()):
        entry = {
            "split": split,
            "source": source,
            "recording_id": recording,
            "trace_count": len(slices),
        }
        for metric in METRICS:
            values = [row[metric] for row in slices if row[metric] is not None]
            entry[metric] = fmean(values) if values else None
        recordings.append(entry)
    return recordings


def summarize(rows: list[dict], unit: str) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["source"])].append(row)

    summaries = []
    for (split, source), members in sorted(groups.items()):
        for metric in METRICS:
            values = [row[metric] for row in members if row[metric] is not None]
            summaries.append({
                "split": split,
                "source": source,
                "unit": unit,
                "group_count": len(members),
                "metric": metric,
                "valid_count": len(values),
                "mean": fmean(values) if values else None,
                "median": median(values) if values else None,
                "p25": quantile(values, 0.25) if values else None,
                "p75": quantile(values, 0.75) if values else None,
                "std": stdev(values) if len(values) > 1 else None,
            })
    return summaries


def qoe_associations(rows: list[dict], unit: str) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["q_learning_minus_buffer_qoe"] is not None:
            groups[(row["split"], row["source"])].append(row)
            groups[(row["split"], "ALL")].append(row)

    correlations = []
    for (split, source), members in sorted(groups.items()):
        for metric in (
            "mean_bandwidth_mbps", "lag1_correlation",
            "coefficient_of_variation", "forecast_h3_nmae",
        ):
            paired = [row for row in members if row[metric] is not None]
            correlations.append({
                "split": split, "source": source, "unit": unit,
                "metric": metric, "pair_count": len(paired),
                "pearson_r_with_qoe_gain": pearson_correlation(
                    [row[metric] for row in paired],
                    [row["q_learning_minus_buffer_qoe"] for row in paired],
                ),
            })
    return correlations


def short(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.3f}"


def markdown_summary(rows: list[dict], summaries: list[dict], window_s: float) -> str:
    lines = [
        "# Trace dynamics: generated results", "",
        f"Common window: {window_s:g} seconds. Forecasts use only past windows.",
        "Full definitions, caveats, and interpretation: [TRACE_DYNAMICS.md](../../docs/TRACE_DYNAMICS.md).",
        "This is a descriptive follow-up using frozen splits and saved policy results.",
        "No agent is trained, tuned, or reevaluated by this script.", "",
    ]
    for split in SPLITS:
        subset = [row for row in rows if row["split"] == split]
        if not subset:
            continue
        lines += [f"## {split.capitalize()} split", "",
                  "These entries are medians across trace files. "
                  "NMAE is normalized MAE; lower is better.", "",
                  "| Source | Traces / recordings | Native interval (s) | Lag-1 r "
                  "| CV | Normalized change | h=1 NMAE | h=3 NMAE |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for source in ("FCC", "HSDPA"):
            members = [row for row in subset if row["source"] == source]
            if not members:
                continue
            def get(metric: str) -> float | None:
                return next(item["median"] for item in summaries if
                            item["split"] == split and item["source"] == source
                            and item["unit"] == "trace" and item["metric"] == metric)
            cells = [short(get(metric)) for metric in (
                "native_median_interval_s", "lag1_correlation",
                "coefficient_of_variation", "normalized_mean_absolute_change",
                "forecast_h1_nmae", "forecast_h3_nmae",
            )]
            count = len({row["recording_id"] for row in members})
            lines.append(f"| {source} | {len(members)} / {count} | " + " | ".join(cells) + " |")
        lines += ["", "Counts with missing/undefined statistics:", ""]
        for source in ("FCC", "HSDPA"):
            members = [row for row in subset if row["source"] == source]
            if members:
                acf = sum(row["lag1_correlation"] is None for row in members)
                forecast = sum(row["forecast_h3_nmae"] is None for row in members)
                lines.append(
                    f"- {source}: {acf} undefined correlations; "
                    f"{forecast} undefined normalized forecast errors."
                )
        lines.append("")
    lines += [
        "## Files and interpretation", "",
        "- `per_trace.csv`: all trace statistics, forecast target counts, and saved QoE differences.",
        "- `per_recording.csv`: means of trace metrics within each original recording.",
        "- `summary.csv` / `summary.json`: mean, median, IQR, and valid counts, "
        "both by trace and by recording.",
        "- `qoe_associations.csv`: descriptive correlations with saved Q-learning minus buffer QoE.",
        "- `metadata.json`: settings and SHA-256 fingerprints of input data/results.", "",
        "Adjacent slices from one recording are dependent; "
        "no independent-trace significance test is implied.",
        "Lag-one correlation describes persistence, while forecast error measures a particular predictor.",
        "The two can disagree. These trace-level forecasts are proxies for the agent's chunk measurements.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/trace_dynamics")
    args = parser.parse_args()
    splits = list(dict.fromkeys(args.splits))
    rows = []
    metadata = {
        "analysis_version": 1,
        "window_s": args.window_seconds,
        "predictor_history_windows": [1, 3],
        "common_forecast_warmup_windows": 3,
        "minimum_correlation_pairs": 3,
        "partial_windows": "discarded",
        "trace_wrap": False,
        "aggregation": "equal trace; also equal recording after averaging its slice metrics",
        "purpose": "descriptive analysis after frozen evaluation; no model changes",
        "inputs": [],
    }
    for split in splits:
        traces = load_split(PROJECT_ROOT, split)
        deltas, qoe_input = load_saved_qoe(split)
        if qoe_input and set(deltas) != {trace.name for trace in traces}:
            raise ValueError(f"saved evaluation trace names disagree with {split} manifest")
        manifest = PROJECT_ROOT / "data/splits" / f"{split}.txt"
        digest = hashlib.sha256()
        for entry in manifest.read_text(encoding="utf-8").splitlines():
            if entry.strip() and not entry.startswith("#"):
                digest.update(entry.encode() + b"\0")
                digest.update((PROJECT_ROOT / "data" / entry).read_bytes())
                digest.update(b"\0")
        metadata["inputs"].append({
            "split": split, "trace_count": len(traces),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "trace_paths_and_bytes_sha256": digest.hexdigest(),
            "saved_qoe": qoe_input,
        })
        for trace in traces:
            rows.append({
                "split": split, "source": trace_source(trace.name),
                "trace_name": trace.name, "recording_id": recording_id(trace.name),
                **describe_trace(trace, args.window_seconds),
                "q_learning_minus_buffer_qoe": deltas.get(trace.name),
            })
        print(f"{split}: analyzed {len(traces)} traces on {args.window_seconds:g}-second windows")

    recordings = recording_means(rows)
    summaries = summarize(rows, "trace") + summarize(recordings, "recording")
    associations = qoe_associations(rows, "trace") + qoe_associations(recordings, "recording")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, items in (
        ("per_trace.csv", rows), ("per_recording.csv", recordings),
        ("summary.csv", summaries),
    ):
        write_csv(args.output_dir / name, items)
    write_csv(args.output_dir / "qoe_associations.csv", associations, fields=[
        "split", "source", "unit", "metric", "pair_count", "pearson_r_with_qoe_gain",
    ])
    for name, data in (("summary.json", summaries), ("metadata.json", metadata)):
        (args.output_dir / name).write_text(
            json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    (args.output_dir / "summary.md").write_text(
        markdown_summary(rows, summaries, args.window_seconds), encoding="utf-8"
    )
    output_label = (
        args.output_dir.relative_to(PROJECT_ROOT)
        if args.output_dir.is_relative_to(PROJECT_ROOT) else args.output_dir
    )
    print(f"Tables saved to {output_label}")


if __name__ == "__main__":
    main()
