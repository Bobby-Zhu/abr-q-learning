#!/usr/bin/env python3
"""Draw report figures (SVG, no third-party packages) from evaluate_final.py output.

    python3 scripts/make_figures.py --split test

Writes into ``results/final/<split>/figures/``:

* ``training_curve.svg`` - smoothed training QoE for each seed
* ``metrics.svg`` - the four comparison metrics, one panel each
* ``qoe_difference.svg`` - per-trace QoE of Q-learning minus buffer-based
"""

import argparse
import csv
from html import escape
import json
import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from abr_simulator.experiments import mean_and_std  # noqa: E402


# Reference categorical palette, slots 1-3 (validated as a set).
COLORS = {"fixed": "#2a78d6", "buffer": "#eb6834", "q_learning": "#1baf7a"}
LABELS = {"fixed": "Fixed 1200 Kbps", "buffer": "Buffer-based", "q_learning": "Q-learning"}
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
FONT = "font-family='system-ui, -apple-system, Segoe UI, sans-serif'"


class Svg:
    def __init__(self, width: int, height: int, title: str) -> None:
        self.width, self.height = width, height
        self.parts = [
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
            f"viewBox='0 0 {width} {height}' {FONT}>",
            f"<title>{escape(title)}</title>",
            f"<rect width='{width}' height='{height}' fill='{SURFACE}'/>",
        ]

    def add(self, element: str) -> None:
        self.parts.append(element)

    def text(self, x, y, content, size=12, fill=TEXT_2, anchor="start", weight="normal"):
        self.add(
            f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' fill='{fill}' "
            f"text-anchor='{anchor}' font-weight='{weight}'>{escape(str(content))}</text>"
        )

    def line(self, x1, y1, x2, y2, stroke=GRID, width=1, extra=""):
        self.add(
            f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' "
            f"stroke='{stroke}' stroke-width='{width}' {extra}/>"
        )

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.parts + ["</svg>"]) + "\n", encoding="utf-8")


def nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    if high == low:
        high = low + 1
    raw = (high - low) / count
    magnitude = 10 ** math.floor(math.log10(raw))
    step = min((s * magnitude for s in (1, 2, 2.5, 5, 10) if s * magnitude >= raw))
    start = math.floor(low / step) * step
    ticks = []
    value = start
    while value <= high + step * 0.001:
        ticks.append(round(value, 10))
        value += step
    if ticks[-1] < high:
        ticks.append(ticks[-1] + step)
    return ticks


def fmt(value: float) -> str:
    return f"{value:,.0f}" if abs(value) >= 100 or value == int(value) else f"{value:.1f}"


def bar_path(x, width, y_value, y_zero, radius=4):
    """A bar whose data end (away from zero) has rounded corners."""

    top, bottom = sorted((y_value, y_zero))
    height = bottom - top
    r = min(radius, width / 2, height)
    if y_value <= y_zero:  # positive bar, round the top
        return (
            f"M{x:.1f},{bottom:.1f} V{top + r:.1f} Q{x:.1f},{top:.1f} {x + r:.1f},{top:.1f} "
            f"H{x + width - r:.1f} Q{x + width:.1f},{top:.1f} {x + width:.1f},{top + r:.1f} "
            f"V{bottom:.1f} Z"
        )
    return (  # negative bar, round the bottom
        f"M{x:.1f},{top:.1f} V{bottom - r:.1f} Q{x:.1f},{bottom:.1f} {x + r:.1f},{bottom:.1f} "
        f"H{x + width - r:.1f} Q{x + width:.1f},{bottom:.1f} {x + width:.1f},{bottom - r:.1f} "
        f"V{top:.1f} Z"
    )


def legend(svg: Svg, x: float, y: float, policies) -> None:
    for policy in policies:
        svg.add(
            f"<rect x='{x:.1f}' y='{y - 9:.1f}' width='10' height='10' rx='2' "
            f"fill='{COLORS[policy]}'/>"
        )
        svg.text(x + 15, y, LABELS[policy], size=12, fill=TEXT)
        x += 30 + 7.2 * len(LABELS[policy])


def metrics_figure(summary: dict, path: Path, split: str) -> None:
    panels = [
        ("mean_episode_qoe", "Mean QoE per video (higher is better)"),
        ("average_bitrate_kbps", "Average bitrate, Kbps"),
        ("mean_rebuffer_s_per_episode", "Rebuffering per video, s (lower is better)"),
        ("mean_switches_per_episode", "Bitrate switches per video (lower is better)"),
    ]
    policies = ("fixed", "buffer", "q_learning")
    panel_w, panel_h = 330, 230
    svg = Svg(2 * panel_w + 40, 2 * panel_h + 90, f"Policy comparison on {split}")
    svg.text(20, 28, f"Policy comparison, {split} split, all traces", size=16, fill=TEXT, weight="600")
    legend(svg, 20, 52, policies)

    for index, (key, title) in enumerate(panels):
        ox = 20 + (index % 2) * (panel_w + 10)
        oy = 75 + (index // 2) * (panel_h + 10)
        values = {}
        for policy in policies:
            runs = [run["all"][key] for run in summary["summaries"][policy]]
            values[policy] = mean_and_std(runs)
        low = min(0.0, min(m - s for m, s in values.values()))
        high = max(0.0, max(m + s for m, s in values.values()))
        ticks = nice_ticks(low, high, 4)
        plot_x, plot_y = ox + 50, oy + 28
        plot_w, plot_h = panel_w - 70, panel_h - 60

        def y_of(v):
            return plot_y + plot_h - (v - ticks[0]) / (ticks[-1] - ticks[0]) * plot_h

        svg.text(ox, oy + 12, title, size=12, fill=TEXT, weight="600")
        for tick in ticks:
            svg.line(plot_x, y_of(tick), plot_x + plot_w, y_of(tick),
                     stroke=TEXT_2 if tick == 0 else GRID)
            svg.text(plot_x - 6, y_of(tick) + 4, fmt(tick), size=10, anchor="end")
        slot = plot_w / len(policies)
        bar_w = min(48, slot * 0.55)
        for i, policy in enumerate(policies):
            mean, std = values[policy]
            x = plot_x + slot * i + (slot - bar_w) / 2
            svg.add(f"<path d='{bar_path(x, bar_w, y_of(mean), y_of(0))}' fill='{COLORS[policy]}'>"
                    f"<title>{LABELS[policy]}: {mean:.2f}</title></path>")
            if std > 0:
                cx = x + bar_w / 2
                svg.line(cx, y_of(mean - std), cx, y_of(mean + std), stroke=TEXT, width=1.5)
                for v in (mean - std, mean + std):
                    svg.line(cx - 5, y_of(v), cx + 5, y_of(v), stroke=TEXT, width=1.5)
            label_y = y_of(max(mean + std, 0)) - 6 if mean >= 0 else y_of(mean - std) + 14
            svg.text(x + bar_w / 2, label_y, fmt(mean), size=11, fill=TEXT, anchor="middle")
    svg.text(20, svg.height - 10,
             "Q-learning bars are the mean over trained seeds; whiskers show ±1 standard deviation.",
             size=11)
    svg.save(path)


def training_curve_figure(directory: Path, seeds: list[int], path: Path) -> None:
    curves = []
    for seed in seeds:
        with (directory / f"training_curve_seed{seed}.csv").open(encoding="utf-8") as source:
            values = [float(row["training_qoe"]) for row in csv.DictReader(source)]
        window = max(1, min(500, len(values) // 20))
        smoothed, total = [], 0.0
        for i, value in enumerate(values):
            total += value
            if i >= window:
                total -= values[i - window]
            if i >= window - 1:
                smoothed.append((i + 1, total / window))
        curves.append(smoothed)

    width, height = 680, 360
    svg = Svg(width, height, "Q-learning training curve")
    svg.text(20, 28, "Training QoE while learning (exploring, random trace starts)",
             size=16, fill=TEXT, weight="600")
    svg.text(20, 48, f"Moving average over {window} episodes; one line per seed", size=12)
    plot_x, plot_y, plot_w, plot_h = 70, 70, width - 100, height - 120
    all_y = [y for curve in curves for _, y in curve]
    episodes = max(x for curve in curves for x, _ in curve)
    y_ticks = nice_ticks(min(all_y), max(all_y), 5)
    x_ticks = nice_ticks(0, episodes, 5)

    def px(x):
        return plot_x + x / x_ticks[-1] * plot_w

    def py(y):
        return plot_y + plot_h - (y - y_ticks[0]) / (y_ticks[-1] - y_ticks[0]) * plot_h

    for tick in y_ticks:
        svg.line(plot_x, py(tick), plot_x + plot_w, py(tick), stroke=TEXT_2 if tick == 0 else GRID)
        svg.text(plot_x - 6, py(tick) + 4, fmt(tick), size=10, anchor="end")
    for tick in x_ticks:
        svg.text(px(tick), plot_y + plot_h + 16, fmt(tick), size=10, anchor="middle")
    svg.text(plot_x + plot_w / 2, height - 14, "Training episode", size=11, anchor="middle")
    svg.text(16, plot_y - 8, "QoE per video", size=11)
    step = max(1, len(curves[0]) // 400)
    for curve in curves:
        points = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in curve[::step])
        svg.add(f"<polyline points='{points}' fill='none' stroke='{COLORS['q_learning']}' "
                f"stroke-width='2' stroke-opacity='0.6' stroke-linejoin='round'/>")
    svg.save(path)


def difference_figure(directory: Path, path: Path, split: str) -> None:
    by_trace: dict[str, dict] = {}
    with (directory / "per_trace.csv").open(encoding="utf-8") as source:
        for row in csv.DictReader(source):
            entry = by_trace.setdefault(row["trace_name"], {"q": [], "source": row["source"]})
            if row["policy"] == "q_learning":
                entry["q"].append(float(row["total_qoe"]))
            elif row["policy"] == "buffer":
                entry["buffer"] = float(row["total_qoe"])
    diffs = sorted(sum(e["q"]) / len(e["q"]) - e["buffer"] for e in by_trace.values())
    better = sum(d > 0 for d in diffs)

    width, height = 680, 360
    svg = Svg(width, height, "Per-trace QoE difference")
    svg.text(20, 28, "Q-learning minus buffer-based, QoE per video, one bar per trace",
             size=16, fill=TEXT, weight="600")
    svg.text(20, 48, f"{split} split: Q-learning is higher on {better} of {len(diffs)} traces "
             "(Q-learning averaged over seeds)", size=12)
    plot_x, plot_y, plot_w, plot_h = 70, 70, width - 100, height - 120
    ticks = nice_ticks(min(0, diffs[0]), max(0, diffs[-1]), 5)

    def py(v):
        return plot_y + plot_h - (v - ticks[0]) / (ticks[-1] - ticks[0]) * plot_h

    for tick in ticks:
        svg.line(plot_x, py(tick), plot_x + plot_w, py(tick), stroke=TEXT_2 if tick == 0 else GRID)
        svg.text(plot_x - 6, py(tick) + 4, fmt(tick), size=10, anchor="end")
    slot = plot_w / len(diffs)
    for i, diff in enumerate(diffs):
        top, bottom = sorted((py(diff), py(0)))
        svg.add(f"<rect x='{plot_x + i * slot:.2f}' y='{top:.1f}' width='{max(slot - 1, 0.8):.2f}' "
                f"height='{max(bottom - top, 0.5):.1f}' fill='{COLORS['q_learning'] if diff > 0 else COLORS['buffer']}'/>")
    svg.text(plot_x + plot_w / 2, height - 14, "Traces, sorted by difference", size=11, anchor="middle")
    svg.text(plot_x + plot_w - 4, plot_y + 12, "Q-learning better", size=11, fill=TEXT, anchor="end")
    svg.text(plot_x + 4, plot_y + plot_h - 6, "Buffer-based better", size=11, fill=TEXT)
    svg.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--results-dir", type=Path)
    args = parser.parse_args()
    directory = args.results_dir or PROJECT_ROOT / "results/final" / args.split
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    metrics_figure(summary, figures / "metrics.svg", summary["split"])
    training_curve_figure(directory, summary["seeds"], figures / "training_curve.svg")
    difference_figure(directory, figures / "qoe_difference.svg", summary["split"])
    print(f"wrote figures to {figures}")


if __name__ == "__main__":
    main()
