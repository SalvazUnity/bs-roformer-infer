"""Chunking-plan primitive shared by chunked Torch inference.

`ChunkingPlan` is the one owner of how a mixture is cut into overlapping
chunks (chunk size, step, fade, border): `utils.demix_track` derives its
windowing purely from this, so the numbers live in exactly one place rather
than being recomputed ad hoc.

Reads: (none)
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_CHUNK_SIZE = 588800


@dataclass(frozen=True)
class ChunkingPlan:
    """How one mixture is cut into overlapping chunks."""

    chunk_size: int
    num_overlap: int
    step: int
    fade_size: int
    border: int

    @classmethod
    def from_config(cls, config) -> ChunkingPlan:
        # chunk_size moved between config sections across upstream config versions.
        if hasattr(config.inference, "chunk_size"):
            chunk_size = config.inference.chunk_size
        elif hasattr(config, "audio") and hasattr(config.audio, "chunk_size"):
            chunk_size = config.audio.chunk_size
        else:
            chunk_size = DEFAULT_CHUNK_SIZE
        num_overlap = config.inference.num_overlap
        step = chunk_size // num_overlap
        return cls(
            chunk_size=chunk_size,
            num_overlap=num_overlap,
            step=step,
            fade_size=chunk_size // 10,
            border=chunk_size - step,
        )
