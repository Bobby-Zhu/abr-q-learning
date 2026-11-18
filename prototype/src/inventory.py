"""Inventory the two selected upstream directories and assign whole captures to splits."""
import csv
import hashlib
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def prepare(out, data):
    rows, series, seen, parents = [], {}, {}, {}
    for directory in ("fcc_ori", "traces_3gp"):
        for p in sorted((data / directory).rglob("*")):
            if not p.is_file():
                continue
            key = p.relative_to(data).as_posix()
            match = re.match(r"trace_(\d+)_", p.name)
            source = "HSDPA" if directory == "traces_3gp" else ("FCC" if match else "non-FCC in fcc_ori")
            group = ("fcc_" + match[1]) if match else re.sub(r"_part\d+\.log$", "", p.name)
            row = dict(source=source, path=key, capture=group, samples="", duration_s="", zero_intervals="", fingerprint="", status="excluded", reason="", split="")
            rows.append(row)
            try:
                a = np.loadtxt(p, ndmin=2)
                if a.shape[1] != 2 or len(a) < 2 or not np.isfinite(a).all():
                    raise ValueError("invalid shape or nonfinite values")
                t, b = a.T
                row.update(samples=len(t), duration_s=float(t[-1]-t[0]), zero_intervals=int((b[1:]==0).sum()))
                if np.any(np.diff(t) <= 0) or np.any(b < 0) or not np.any(b[1:] > 0):
                    raise ValueError("invalid interval, negative speed or no delivery")
                if source.startswith("non-FCC"):
                    raise ValueError("out-of-scope commute capture in FCC directory")
                norm = np.column_stack((t-t[0], b))
                digest = hashlib.sha256(norm.astype("<f8").tobytes()).hexdigest()
                row["fingerprint"] = digest
                if digest in seen:
                    raise ValueError("duplicate of " + seen[digest])
                seen[digest] = key
                if source == "FCC" and group in parents:
                    full, path = parents[group]
                    start = np.searchsorted(full[:, 0], t[0])
                    if np.array_equal(full[start:start+len(a)], a):
                        raise ValueError("subsequence of " + path)
                if source == "FCC" and p.parent.name == "cooked_traces":
                    parents[group] = a, key
                series[key] = (norm[:,0].tolist(), b.tolist())
                row["status"] = "kept"
            except ValueError as e:
                row["reason"] = str(e)
    for source in ("FCC","HSDPA"):
        groups = defaultdict(list)
        for row in rows:
            if row["source"] == source and row["status"] == "kept":
                groups[row["capture"]].append(row)
        ids = sorted(groups)
        random.Random(SEED).shuffle(ids)
        # Greedy assignment by remaining trace quota, with whole captures indivisible.
        n = sum(map(len,groups.values()))
        gap = dict(train=.70*n, validation=.15*n, test=.15*n)
        for g in sorted(ids, key=lambda k: -len(groups[k])):
            split = max(gap, key=gap.get)
            for row in groups[g]:
                row["split"] = split
            gap[split] -= len(groups[g])
    write_csv(out / "manifest.csv", rows)
    counts = []
    for src in dict.fromkeys(r["source"] for r in rows):
        subset = [r for r in rows if r["source"] == src]
        c = Counter(r["split"] or "excluded" for r in subset)
        counts.append(dict(source=src,files=len(subset),excluded=c["excluded"],train=c["train"],validation=c["validation"],test=c["test"]))
    write_csv(out / "counts.csv",counts)
    kept = [r for r in rows if r["status"]=="kept"]
    assert len({r["fingerprint"] for r in kept}) == len(kept)
    assert len({(r["capture"],r["split"]) for r in kept}) == len({r["capture"] for r in kept})
    return kept, series
