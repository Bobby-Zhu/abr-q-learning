"""Load per-chunk byte sizes for each bitrate level."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoMetadata:
    """Chunk sizes indexed as ``sizes_bytes[bitrate_index][chunk_index]``."""

    sizes_bytes: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not self.sizes_bytes:
            raise ValueError("video metadata must contain at least one bitrate")
        chunk_counts = {len(sizes) for sizes in self.sizes_bytes}
        if len(chunk_counts) != 1 or not next(iter(chunk_counts)):
            raise ValueError("all bitrate levels must have the same non-zero chunk count")
        if any(size <= 0 for sizes in self.sizes_bytes for size in sizes):
            raise ValueError("video chunk sizes must be positive")

    @property
    def bitrate_levels(self) -> int:
        return len(self.sizes_bytes)

    @property
    def chunk_count(self) -> int:
        return len(self.sizes_bytes[0])

    def sizes_for_chunk(self, chunk_index: int) -> tuple[int, ...]:
        return tuple(sizes[chunk_index] for sizes in self.sizes_bytes)


def load_video_metadata(directory: str | Path) -> VideoMetadata:
    """Load ``video_size_0``, ``video_size_1``, ... from a directory."""

    video_directory = Path(directory)
    sizes_by_bitrate: list[tuple[int, ...]] = []
    bitrate_index = 0

    while True:
        size_path = video_directory / f"video_size_{bitrate_index}"
        if not size_path.exists():
            break
        sizes: list[int] = []
        with size_path.open(encoding="utf-8") as size_file:
            for line_number, raw_line in enumerate(size_file, start=1):
                fields = raw_line.split()
                if not fields:
                    continue
                try:
                    sizes.append(int(fields[0]))
                except ValueError as error:
                    raise ValueError(
                        f"{size_path}:{line_number}: chunk size must be an integer"
                    ) from error
        sizes_by_bitrate.append(tuple(sizes))
        bitrate_index += 1

    if not sizes_by_bitrate:
        raise ValueError(f"no video_size_N files found in {video_directory}")
    return VideoMetadata(tuple(sizes_by_bitrate))
