"""Load reproducible trace sets from project split manifests."""

from pathlib import Path

from .trace import NetworkTrace, load_trace


def load_trace_manifest(
    manifest_path: str | Path,
    data_root: str | Path,
) -> tuple[NetworkTrace, ...]:
    """Load traces listed as paths relative to ``data_root``.

    Empty lines and lines beginning with ``#`` are ignored. Duplicate entries,
    absolute paths, missing files, and paths escaping the data root are rejected.
    """

    manifest = Path(manifest_path)
    root = Path(data_root).resolve()
    trace_paths: list[Path] = []
    seen: set[Path] = set()

    with manifest.open(encoding="utf-8") as manifest_file:
        for line_number, raw_line in enumerate(manifest_file, start=1):
            entry = raw_line.strip()
            if not entry or entry.startswith("#"):
                continue

            relative_path = Path(entry)
            if relative_path.is_absolute():
                raise ValueError(
                    f"{manifest}:{line_number}: expected a relative data path"
                )
            trace_path = (root / relative_path).resolve()
            if not trace_path.is_relative_to(root):
                raise ValueError(
                    f"{manifest}:{line_number}: path escapes the data directory"
                )
            if trace_path in seen:
                raise ValueError(
                    f"{manifest}:{line_number}: duplicate trace path {entry}"
                )
            if not trace_path.is_file():
                raise ValueError(
                    f"{manifest}:{line_number}: trace file does not exist: {entry}"
                )

            seen.add(trace_path)
            trace_paths.append(trace_path)

    if not trace_paths:
        raise ValueError(f"no trace paths found in {manifest}")
    return tuple(load_trace(path) for path in trace_paths)
