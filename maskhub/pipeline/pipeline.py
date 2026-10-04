"""Bounded, resumable batch pipeline.

Rows are pulled lazily from ``source.rows()`` and grouped into numbered
batches. Batching and committing happen in a single generator loop, which
gives deterministic backpressure: the source iterator is not advanced
while a batch is being committed, so a source that "pushes millions of
rows" can never materialise more than one batch in memory.

Group commit per batch (the batch stays invisible until the last step)::

    sink.stage(seq, rows)     # data lands outside the visible namespace
    manifest.stage(seq, ...)  # durable: data is staged
    sink.commit(seq)          # atomic promotion (os.replace) -> visible
    manifest.commit(seq)      # durable: commit point

Resume: on start, OPEN/STAGED leftovers from a previous crash are marked
ABANDONED (and their staging files discarded). Sequence numbers are
monotonic attempt numbers, so retries get fresh, higher numbers and batches
that already committed are never rerun or overwritten; committed batches
may therefore have holes (abandoned attempts) between them.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from maskhub.pipeline.manifest import Manifest
from maskhub.pipeline.sink import BatchSink
from maskhub.pipeline.source import BatchSource

LOGGER = logging.getLogger("maskhub.pipeline")

DEFAULT_BATCH_ROWS = 10_000
# Rough ceiling for the serialised size of one buffered batch.
DEFAULT_BATCH_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class BatchConfig:
    batch_rows: int = DEFAULT_BATCH_ROWS
    batch_bytes: int = DEFAULT_BATCH_BYTES

    def __post_init__(self) -> None:
        if self.batch_rows <= 0:
            raise ValueError("batch_rows must be positive")
        if self.batch_bytes <= 0:
            raise ValueError("batch_bytes must be positive")


@dataclass(frozen=True)
class PipelineResult:
    started_at_sequence: int
    last_committed_sequence: int
    committed_batches: int
    committed_rows: int
    abandoned_sequences: tuple[int, ...]


def estimate_row_bytes(row: dict[str, Any]) -> int:
    """Size estimate used for the byte-based memory budget."""
    try:
        return len(json.dumps(row, ensure_ascii=False).encode("utf-8"))
    except (TypeError, ValueError):
        return len(repr(row).encode("utf-8"))


class BatchPipeline:
    def __init__(
        self,
        source: BatchSource,
        sink: BatchSink,
        manifest: Manifest,
        config: BatchConfig | None = None,
    ) -> None:
        self._source = source
        self._sink = sink
        self._manifest = manifest
        self._config = config or BatchConfig()

    def run(self) -> PipelineResult:
        """Drain the source to the sink, resuming after the last commit."""
        abandoned = tuple(self._recover())
        first_sequence = self._manifest.next_sequence()
        self._guard_sink_alignment(first_sequence)

        sequence = first_sequence
        for rows, source_start, source_end in self._iter_batches(sequence):
            self._commit_group(sequence, rows, source_start, source_end)
            sequence += 1

        total_committed_rows = sum(
            record.row_count for record in self._manifest.committed()
        )
        return PipelineResult(
            started_at_sequence=first_sequence,
            last_committed_sequence=self._manifest.last_committed_sequence(),
            committed_batches=sequence - first_sequence,
            committed_rows=total_committed_rows,
            abandoned_sequences=abandoned,
        )

    def _recover(self) -> list[int]:
        """Mark crash leftovers abandoned and drop their staging files."""
        pending = self._manifest.recover_pending()
        if pending:
            self._sink.clean_stale_staging(pending)
            LOGGER.warning(
                "abandoned incomplete batches after recovery: %s", pending
            )
        return pending

    def _guard_sink_alignment(self, first_sequence: int) -> None:
        """Refuse to start if the physical sink holds unknown committed data.

        New attempt numbers are strictly higher than every number the
        manifest knows about, so they can never collide with prior work.
        A committed file the manifest has no record of indicates a lost or
        swapped manifest — running would eventually overwrite it, so fail.
        """
        known = {record.sequence for record in self._manifest.committed()}
        orphans = sorted(set(self._sink.committed_sequences()) - known)
        if orphans:
            raise RuntimeError(
                f"sink contains committed batches unknown to manifest: {orphans}"
            )

    def _iter_batches(
        self, first_sequence: int
    ) -> Iterator[tuple[list[dict[str, Any]], int, int]]:
        """Group rows into bounded batches.

        The source iterator is pulled only here, one row at a time, and a
        batch is yielded the moment either bound is reached; nothing beyond
        one batch is ever buffered. Each batch carries its half-open source
        offsets ``[start, end)`` for exact resume.
        """
        rows: list[dict[str, Any]] = []
        byte_size = 0
        sequence = first_sequence
        start_offset = self._manifest.committed_source_offset()
        offset = start_offset
        batch_start = offset

        for row in self._source.rows(start_offset):
            if not rows:
                self._manifest.open(sequence)
                batch_start = offset
            rows.append(row)
            byte_size += estimate_row_bytes(row)
            offset += 1
            if (
                len(rows) >= self._config.batch_rows
                or byte_size >= self._config.batch_bytes
            ):
                yield rows, batch_start, offset
                rows = []
                byte_size = 0
                sequence += 1

        if rows:
            yield rows, batch_start, offset

    def _commit_group(
        self,
        sequence: int,
        rows: list[dict[str, Any]],
        source_start: int,
        source_end: int,
    ) -> None:
        """Stage data, then flip visibility exactly once at the end.

        If anything fails the batch is deliberately left in its OPEN/STAGED
        state with its staging file on disk, exactly as a hard crash would.
        The next run finds it via :meth:`Manifest.recover_pending`, marks it
        ABANDONED and redoes it under a fresh sequence number — readers
        never observe it in the meantime.
        """
        staged_bytes = self._sink.stage(sequence, rows)
        self._manifest.stage(
            sequence,
            row_count=len(rows),
            byte_count=staged_bytes,
            source_start=source_start,
            source_end=source_end,
        )
        self._sink.commit(sequence)
        self._manifest.commit(sequence)
