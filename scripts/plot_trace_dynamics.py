#!/usr/bin/env python3
"""Create report figures from analyze_trace_dynamics.py's saved CSVs.

Requires the optional plotting dependency in requirements-analysis.txt.
Numerical analysis, training, and evaluation continue to use the stdlib.
"""

import argparse
import csv
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from abr_simulator.trace_dynamics import pearson_correlation  # noqa: E402


COLORS = {"FCC": "#2878b5", "HSDPA": "#d66c32"}
PANELS = (
    ("native_median_interval_s", "Native sampling interval", "Seconds"),
    ("lag1_correlation", "Persistence at the common lag", "Lag-one Pearson r"),
    ("coefficient_of_variation", "Relative variability", "Population std / mean"),
    ("normalized_mean_absolute_change", "Relative step-to-step change", "Mean absolute change / mean"),
    ("forecast_h1_nmae", "Last-window forecast error", "MAE / mean capacity"),
    ("forecast_h3_nmae", "Three-window harmonic forecast error", "MAE / mean capacity"),
)


def load_rows(filename: Path, split: str) -> list[dict]:
    with filename.open(newline="", encoding="utf-8") as source:
        return [row for row in csv.DictReader(source) if row["split"] == split]


def numeric(rows: list[dict], source: str, metric: str) -> list[float]:
    return [float(row[metric]) for row in rows if row["source"] == source and row[metric] != ""]


def save_figure(figure, directory: Path, stem: str) -> None:
    for suffix in ("svg", "pdf", "png"):
        figure.savefig(
            directory / f"{stem}.{suffix}", dpi=160,
            metadata={"Creator": "scripts/plot_trace_dynamics.py"},
        )
        if suffix == "svg":
            svg_path = directory / f"{stem}.{suffix}"
            svg_path.write_text(
                "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines())
                + "\n", encoding="utf-8",
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "results/trace_dynamics")
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("Install plotting dependencies: python3 -m pip install -r requirements-analysis.txt")

    metadata = json.loads((args.input_dir / "metadata.json").read_text(encoding="utf-8"))
    window_s = metadata["window_s"]
    traces = load_rows(args.input_dir / "per_trace.csv", args.split)
    recordings = load_rows(args.input_dir / "per_recording.csv", args.split)
    if not traces:
        raise SystemExit(f"no {args.split} traces in {args.input_dir}")
    directory = args.input_dir / "figures"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })

    figure, axes = plt.subplots(2, 3, figsize=(12, 7.8))
    for axis, (metric, title, ylabel) in zip(axes.flat, PANELS):
        for position, source in enumerate(COLORS, start=1):
            values = numeric(traces, source, metric)
            if not values:
                continue
            box = axis.boxplot(
                [values], positions=[position], widths=0.42,
                whis=(10, 90), patch_artist=True, showfliers=True,
                medianprops={"color": "#1d1d1d", "linewidth": 1.6},
                flierprops={"marker": ".", "markersize": 3, "alpha": 0.35},
            )
            box["boxes"][0].set_facecolor(COLORS[source])
            box["boxes"][0].set_alpha(0.3)
            recording_values = numeric(recordings, source, metric)
            offsets = [((index * 0.61803398875) % 1 - 0.5) * 0.26
                       for index in range(len(recording_values))]
            axis.scatter(
                [position + offset for offset in offsets], recording_values,
                s=17, facecolors="none", edgecolors=COLORS[source], linewidths=0.8,
                zorder=3,
            )
        axis.set_xticks([1, 2], list(COLORS))
        axis.set_xlim(0.5, 2.5)
        axis.set_title(title, fontsize=10)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
        if metric == "lag1_correlation":
            axis.set_ylim(-1.05, 1.05)
            axis.axhline(0, color="#777777", linewidth=0.7)
    counts = "; ".join(
        f"{source}: {sum(row['source'] == source for row in traces)} traces / "
        f"{sum(row['source'] == source for row in recordings)} recordings"
        for source in COLORS
    )
    figure.suptitle(f"Trace dynamics on {window_s:g}-second windows — {args.split} split", y=0.98)
    figure.text(
        0.5, 0.035,
        f"{counts}\nBoxes: trace median/IQR; whiskers: 10th–90th percentiles. "
        "Open dots: recording means.\n"
        "All adjacent windows stay inside their trace. "
        "These distributions are descriptive; slices of a recording are dependent.",
        ha="center", fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.115, 1, 0.95))
    save_figure(figure, directory, f"{args.split}_source_comparison")
    plt.close(figure)

    if any(row["q_learning_minus_buffer_qoe"] != "" for row in traces):
        figure, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
        for axis, source in zip(axes, COLORS):
            paired = [row for row in traces if row["source"] == source
                      and row["forecast_h3_nmae"] != ""
                      and row["q_learning_minus_buffer_qoe"] != ""]
            xs = [float(row["forecast_h3_nmae"]) for row in paired]
            ys = [float(row["q_learning_minus_buffer_qoe"]) for row in paired]
            axis.scatter(xs, ys, color=COLORS[source], s=24, alpha=0.6, linewidths=0)
            r = pearson_correlation(xs, ys)
            label = "undefined" if r is None else f"{r:.2f}"
            axis.set_title(f"{source}: {len(paired)} traces, Pearson r = {label}")
            axis.axhline(0, color="#777777", linestyle="--", linewidth=0.9)
            axis.set_xlabel("Three-window forecast NMAE (lower = more accurate)")
            axis.grid(alpha=0.18)
        axes[0].set_ylabel("Saved Q-learning minus buffer QoE per video")
        figure.suptitle(f"Forecast error and policy advantage — {args.split} split")
        figure.text(
            0.5, 0.035,
            "Each point is a trace; Q-learning QoE is averaged over the saved seeds.\n"
            "Exploratory association only: a proxy forecast error "
            "does not establish the cause of a policy gain.",
            ha="center", fontsize=9,
        )
        figure.tight_layout(rect=(0, 0.12, 1, 0.95))
        save_figure(figure, directory, f"{args.split}_qoe_relationship")
        plt.close(figure)
    print(f"Figures saved to {directory}")


if __name__ == "__main__":
    main()
