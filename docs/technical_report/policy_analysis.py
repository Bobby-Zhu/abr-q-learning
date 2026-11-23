#!/usr/bin/env python3
"""Decision-level comparison of the archived Q-learning agents and the buffer rule.

Example (Python 3.10+, no third-party dependencies unless --figure is given):
    python policy_analysis.py --repo /path/to/abr-q-learning
    python policy_analysis.py --repo /path/to/abr-q-learning --figure out.pdf

Replays the five frozen test Q-tables and the default buffer rule from the
start of every test slice, checks the totals against the committed
per_trace.csv, and prints the statistics quoted in the report subsection
"What Did the Agent Learn?". No training, network access, or repository
writes are performed. --figure additionally needs matplotlib.
"""

import argparse
import csv
from collections import defaultdict
import math
from pathlib import Path
import statistics
import sys

SEEDS = range(100, 105)
LADDER = (300, 750, 1200, 1850, 2850, 4300)


def harmonic(values):
    return len(values) / sum(1 / max(v, 1e-9) for v in values) if values else None


def replay(root):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "src"))
    from abr_simulator import ABREnvironment, BufferBasedPolicy, QLearningAgent
    from abr_simulator import SimulatorConfig
    from abr_simulator.experiments import load_split, load_video, trace_source

    config = SimulatorConfig()
    video = load_video(root)
    rule = BufferBasedPolicy(config)
    agents = {seed: QLearningAgent.load(
        root / f"results/final/test/q_table_seed{seed}.json", config) for seed in SEEDS}
    rows = []
    for trace in load_split(root, "test"):
        for policy in ["buffer", *SEEDS]:
            agent = None if policy == "buffer" else agents[policy]
            env = ABREnvironment(trace, video, config)
            env.reset(start_index=1)
            if agent:
                agent.begin_episode()
            previous, history, last_kbps = None, [], LADDER[config.initial_bitrate_index]
            while True:
                buffer_s = env.buffer_s
                estimate = harmonic(history[-3:])
                if agent:
                    state = agent.observe(buffer_s, previous)
                    action = agent.greedy_action(state)
                    visits = sum(agent.visit_counts[state])
                else:
                    action, visits = rule.select_action(buffer_s), None
                result = env.step(action)
                rows.append({
                    "trace": trace.name, "source": trace_source(trace.name),
                    "policy": "buffer" if policy == "buffer" else "q", "seed": policy,
                    "chunk": result.chunk_index + 1, "buffer": buffer_s,
                    "estimate": estimate, "kbps": result.bitrate_kbps,
                    "rule_kbps": LADDER[rule.select_action(buffer_s)],
                    "measured": result.measured_throughput_mbps,
                    "rebuffer": result.rebuffer_s, "reward": result.reward,
                    "jump": abs(result.bitrate_kbps - last_kbps), "visits": visits,
                })
                history.append(result.measured_throughput_mbps)
                previous, last_kbps = result, result.bitrate_kbps
                if result.done:
                    break
    return rows


def check_against_committed(root, rows):
    totals = defaultdict(float)
    for row in rows:
        key = ("buffer", "") if row["policy"] == "buffer" else ("q_learning", str(row["seed"]))
        totals[(*key, row["trace"])] += row["reward"]
    worst = 0.0
    with (root / "results/final/test/per_trace.csv").open(newline="", encoding="utf-8") as f:
        for line in csv.DictReader(f):
            key = (line["policy"], line["seed"], line["trace_name"])
            if key in totals:
                worst = max(worst, abs(totals[key] - float(line["total_qoe"])))
    if worst > 1e-6:
        raise SystemExit(f"Replay disagrees with per_trace.csv by {worst}")
    print(f"replay matches per_trace.csv ({len(totals)} episodes, max diff {worst:.1e})")


def share(rows, predicate):
    return sum(1 for r in rows if predicate(r)) / len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, required=True, help="Repository root")
    parser.add_argument("--figure", type=Path, help="Optional output path for Figure 2")
    args = parser.parse_args()
    root = args.repo.expanduser().resolve()
    if not (root / "src/abr_simulator").is_dir():
        parser.error(f"Not an abr-q-learning repository: {root}")

    rows = replay(root)
    check_against_committed(root, rows)
    q = [r for r in rows if r["policy"] == "q"]
    b = [r for r in rows if r["policy"] == "buffer"]
    episodes = {"q": len(SEEDS) * len({r["trace"] for r in q}), "buffer": len({r["trace"] for r in b})}

    print(f"Q higher/same/lower than buffer rule in Q's states: "
          f"{share(q, lambda r: r['kbps'] > r['rule_kbps']):.3f} / "
          f"{share(q, lambda r: r['kbps'] == r['rule_kbps']):.3f} / "
          f"{share(q, lambda r: r['kbps'] < r['rule_kbps']):.3f}")
    print(f"median buffer: Q {statistics.median(r['buffer'] for r in q):.1f} s, "
          f"buffer rule {statistics.median(r['buffer'] for r in b):.1f} s; "
          f"decisions above 35 s: Q {sum(r['buffer'] >= 35 for r in q)} of {len(q)}, "
          f"rule {share(b, lambda r: r['buffer'] >= 35):.3f}")

    per_chunk = []
    cumulative = 0.0
    for chunk in range(1, 50):
        entry = {"chunk": chunk}
        for name, subset in (("q", q), ("buffer", b)):
            selected = [r for r in subset if r["chunk"] == chunk]
            entry[f"{name}_buffer"] = statistics.fmean(r["buffer"] for r in selected)
            entry[f"{name}_reward"] = sum(r["reward"] for r in selected) / episodes[name]
        cumulative += entry["q_reward"] - entry["buffer_reward"]
        entry["cumulative"] = cumulative
        per_chunk.append(entry)
    peak = max(per_chunk, key=lambda e: e["cumulative"])
    print(f"cumulative advantage peaks at {peak['cumulative']:.2f} (chunk {peak['chunk']}), "
          f"final {cumulative:.2f}")
    for cutoff in (5, 10, 15, 20):
        early = per_chunk[cutoff - 1]["cumulative"]
        print(f"  cutoff {cutoff:2}: before {early:+.2f}, after {cumulative - early:+.2f}")

    def phase_gain(source, early):
        out = {}
        for name, subset in (("q", q), ("buffer", b)):
            sel = [r for r in subset if r["source"] == source and (r["chunk"] <= 10) == early]
            n = episodes[name] * len({r["trace"] for r in subset if r["source"] == source}) \
                / len({r["trace"] for r in subset})
            out[name] = sum(r["reward"] for r in sel) / n
        return out["q"] - out["buffer"]
    for source in ("FCC", "HSDPA"):
        print(f"{source}: chunks 1-10 {phase_gain(source, True):+.2f}, "
              f"chunks 11-49 {phase_gain(source, False):+.2f}")
    for name, subset in (("Q", q), ("buffer rule", b)):
        early = [r["kbps"] for r in subset if r["chunk"] <= 10]
        jumps = [r["jump"] for r in subset if r["jump"] > 0]
        print(f"{name}: mean bitrate in chunks 1-10 {statistics.fmean(early):.0f} kbps; "
              f"switches > 1000 kbps {sum(j > 1000 for j in jumps) / len(jumps):.3f}")

    print(f"Q decisions in states visited > 1000 times: {share(q, lambda r: r['visits'] > 1000):.3f}; "
          f"never-visited: {sum(r['visits'] == 0 for r in q)}")
    rare = [r for r in q if r["visits"] <= 1000]
    total_rebuffer = sum(r["rebuffer"] for r in q)
    print(f"  states <= 1000 visits: {len(rare) / len(q):.3f} of decisions, "
          f"{sum(r['rebuffer'] for r in rare) / total_rebuffer:.3f} of rebuffering")

    later = [r for r in q if r["chunk"] > 1]
    later_rebuffer = sum(r["rebuffer"] for r in later)
    for cut in (0.5, 0.25):
        hit = [r for r in later if r["measured"] / r["estimate"] < cut]
        print(f"delivered < {cut:.0%} of estimate: {len(hit) / len(later):.3f} of decisions, "
              f"{sum(r['rebuffer'] for r in hit) / later_rebuffer:.2f} of post-start rebuffering")
    stalls = [r for r in later if r["rebuffer"] > 0]
    print(f"post-start stall time beginning with >= 10 s buffered: "
          f"{sum(r['rebuffer'] for r in stalls if r['buffer'] >= 10) / later_rebuffer:.2f}")
    for name, subset in (("Q", later), ("buffer rule", [r for r in b if r["chunk"] > 1])):
        ratios = [r["kbps"] / 1000 / r["estimate"] for r in subset]
        print(f"{name}: requests above estimate {sum(x > 1 for x in ratios) / len(ratios):.3f}, "
              f"median chosen/estimate {statistics.median(ratios):.2f}")

    if args.figure:
        draw_figure(args.figure, per_chunk, q)


def draw_figure(path, per_chunk, q):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, orange, ink, muted = "#2a78d6", "#eb6834", "#222222", "#777777"
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": muted, "xtick.color": muted, "ytick.color": muted,
                         "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6})
    chunks = [e["chunk"] for e in per_chunk]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.1))
    cum = [e["cumulative"] for e in per_chunk]
    peak = max(range(len(cum)), key=cum.__getitem__)
    ax[0].plot(chunks, cum, color=blue, lw=2)
    ax[0].axhline(0, color=muted, lw=0.8)
    for i in (peak, len(cum) - 1):
        ax[0].plot(chunks[i], cum[i], "o", color=blue, ms=6, mec="white", mew=1.5)
    ax[0].annotate(f"peak {cum[peak]:.2f} at chunk {chunks[peak]}", (chunks[peak], cum[peak]),
                   xytext=(21, 3.75), fontsize=8, color=ink)
    ax[0].annotate(f"final {cum[-1]:.2f}", (49, cum[-1]), xytext=(40, 2.4), fontsize=8, color=ink)
    ax[0].set(ylim=(-0.3, 4.2), xlabel="Chunk", ylabel="Cumulative QoE advantage")
    ax[0].set_title("(a) Where the gain accrues", fontsize=9, loc="left")
    qb, bb = [e["q_buffer"] for e in per_chunk], [e["buffer_buffer"] for e in per_chunk]
    ax[1].plot(chunks, qb, color=blue, lw=2)
    ax[1].plot(chunks, bb, color=orange, lw=2)
    ax[1].text(28, qb[27] - 4, "Q-learning", color=ink, fontsize=8)
    ax[1].text(28, bb[27] + 2.6, "Buffer-based", color=ink, fontsize=8)
    ax[1].axhline(10, color=muted, lw=0.8, ls="--")
    ax[1].text(33, 7.5, "reservoir 10 s", fontsize=7, color=muted)
    ax[1].set(ylim=(0, 34), xlabel="Chunk", ylabel="Buffer before request (s)")
    ax[1].set_title("(b) Mean buffer level", fontsize=9, loc="left")
    edges = (0.4, 0.7, 1.0, 1.4, 1.9, 2.6, 3.5)
    labels = ["<0.4", "0.4–0.7", "0.7–1.0", "1.0–1.4", "1.4–1.9", "1.9–2.6", "2.6–3.5", ">3.5"]
    groups = defaultdict(list)
    for r in q:
        if r["estimate"] is not None and 14 <= r["buffer"] < 19:
            groups[sum(r["estimate"] >= e for e in edges)].append(r)
    xs = sorted(groups)
    ax[2].plot(xs, [statistics.fmean(r["kbps"] for r in groups[x]) for x in xs], "-o",
               color=blue, lw=2, ms=5, mec="white", mew=1, label="Q-learning")
    ax[2].plot(xs, [statistics.fmean(r["rule_kbps"] for r in groups[x]) for x in xs], "-s",
               color=orange, lw=2, ms=5, mec="white", mew=1, label="Buffer rule, same states")
    ax[2].set_xticks(xs)
    ax[2].set_xticklabels([labels[x] for x in xs], rotation=45, ha="right", fontsize=7)
    ax[2].set(xlabel="Throughput estimate (Mbit/s)", ylabel="Mean chosen bitrate (kbps)")
    ax[2].set_title("(c) States with buffer 14–19 s", fontsize=9, loc="left")
    ax[2].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"figure written to {path}")


if __name__ == "__main__":
    main()
