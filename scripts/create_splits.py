#!/usr/bin/env python3
"""Create or verify deterministic, capture-disjoint trace manifests."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
SPLIT_ROOT = DATA_ROOT / "splits"
SEED = 42


@dataclass(frozen=True)
class TraceRecord:
    path: str
    upstream_split: str
    capture: str
    source: str
    fingerprint: str


def capture_id(name: str) -> tuple[str, str]:
    stem = re.sub(r"_part\d+$", "", name.removesuffix(".log"))
    fcc = re.match(r"trace_(\d+)_", stem)
    return ("FCC", f"fcc_{fcc[1]}") if fcc else ("non-FCC", stem)


def trace_fingerprint(path: Path) -> str:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"{path}:{line_number}: expected timestamp and bandwidth")
            time, bandwidth = map(Decimal, fields)
            if (not time.is_finite() or not bandwidth.is_finite() or bandwidth < 0
                    or (rows and time <= rows[-1][0])):
                raise ValueError(f"{path}:{line_number}: invalid timestamp or bandwidth")
            rows.append((time, bandwidth))
    if len(rows) < 2 or not any(bandwidth > 0 for _, bandwidth in rows[1:]):
        raise ValueError(f"{path}: no usable delivery interval")
    start = rows[0][0]
    normalized = "\n".join(
        f"{(time - start).normalize():f} {bandwidth.normalize() if bandwidth else 0:f}"
        for time, bandwidth in rows
    )
    return hashlib.sha256(normalized.encode()).hexdigest()


def read_records(data_root: Path) -> list[TraceRecord]:
    trace_root = data_root / "traces/fcc_hsdpa"
    records = []
    for upstream_split in ("train", "holdout"):
        for path in sorted((trace_root / upstream_split).iterdir()):
            if path.is_file():
                source, capture = capture_id(path.name)
                records.append(TraceRecord(path.relative_to(data_root).as_posix(),
                                           upstream_split, capture, source,
                                           trace_fingerprint(path)))
    return records


def assign_splits(records: list[TraceRecord]) -> tuple[dict, dict]:
    parent = {record.capture: record.capture for record in records}

    def representative(capture: str) -> str:
        while parent[capture] != capture:
            capture = parent[capture]
        return capture

    retained = {}
    duplicates = {}
    for record in sorted(records, key=lambda r: (r.upstream_split != "train", r.path)):
        original = retained.setdefault(record.fingerprint, record)
        if original is not record:
            first, second = sorted((representative(original.capture),
                                    representative(record.capture)))
            parent[second] = first
            duplicates[record.path] = original.path

    training_groups = {representative(r.capture) for r in records if r.upstream_split == "train"}
    groups = defaultdict(list)
    for record in retained.values():
        groups[representative(record.capture)].append(record)

    assignments = {capture: "train" for capture in training_groups}
    for source in ("FCC", "non-FCC"):
        holdout_groups = [
            capture for capture, members in groups.items()
            if capture not in training_groups
            and ("FCC" if any(r.source == "FCC" for r in members) else "non-FCC") == source
        ]
        counts = {"validation": 0, "test": 0}
        for capture in sorted(holdout_groups, key=lambda c: (
                -len(groups[c]), hashlib.sha256(f"{SEED}:{c}".encode()).hexdigest(), c)):
            split = min(counts, key=counts.__getitem__)
            assignments[capture] = split
            counts[split] += len(groups[capture])

    manifests = defaultdict(list)
    audit_rows = []
    for record in sorted(records, key=lambda r: r.path):
        capture = representative(record.capture)
        split = assignments[capture]
        duplicate_of = duplicates.get(record.path)
        if duplicate_of is None:
            manifests[split].append(record.path)
        reason = ("exact duplicate" if duplicate_of else
                  "capture overlaps upstream training" if split == "train" and
                  record.upstream_split == "holdout" else None)
        audit_rows.append(dict(path=record.path, upstream_split=record.upstream_split,
                               source=record.source, original_capture=record.capture,
                               capture=capture, fingerprint=record.fingerprint, split=split,
                               duplicate_of=duplicate_of, reason=reason))

    summary = {}
    for split in ("train", "validation", "test"):
        rows = [r for r in audit_rows if r["split"] == split and r["duplicate_of"] is None]
        if not rows:
            raise ValueError(f"{split} split is empty")
        summary[split] = dict(traces=len(rows), captures=len({r["capture"] for r in rows}),
                              fcc_traces=sum(r["source"] == "FCC" for r in rows))
    return dict(manifests), dict(seed=SEED, input_traces=len(records),
                                excluded_duplicates=len(duplicates), splits=summary, files=audit_rows)


def build_manifests(data_root: Path = DATA_ROOT) -> tuple[dict, dict]:
    return assign_splits(read_records(data_root))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the committed files")
    args = parser.parse_args()
    manifests, audit = build_manifests()
    files = {f"{split}.txt": "\n".join(entries) + "\n" for split, entries in manifests.items()}
    files["capture_audit.json"] = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    for name, content in files.items():
        path = SPLIT_ROOT / name
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"split artifact is stale: {path}")
        else:
            path.write_text(content, encoding="utf-8", newline="\n")
    for split, counts in audit["splits"].items():
        print(f"{split}: {counts['traces']} traces in {counts['captures']} captures")
    print(f"Excluded {audit['excluded_duplicates']} exact duplicates")


if __name__ == "__main__":
    main()
