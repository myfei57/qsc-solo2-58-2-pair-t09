"""Read committed batches back, in commit order.

Only batches the manifest records as ``committed`` are yielded, so a
crashed/staged batch is never seen even if a stray file exists. Sequences
are yielded ascending; gaps mark abandoned attempts.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from maskhub.pipeline.manifest import BatchRecord, BatchState, Manifest
from maskhub.pipeline.sink import BatchSink


class CommittedReader:
    def __init__(self, sink: BatchSink, manifest: Manifest) -> None:
        self._sink = sink
        self._manifest = manifest

    def sequences(self) -> list[int]:
        return [
            record.sequence
            for record in self._manifest.committed()
            if self._sink.is_committed(record.sequence)
        ]

    def batches(self) -> Iterator[tuple[BatchRecord, list[dict[str, Any]]]]:
        for record in self._manifest.committed():
            if not self._sink.is_committed(record.sequence):
                continue
            yield record, self._sink.read_committed(record.sequence)

    def rows(self) -> Iterator[dict[str, Any]]:
        for _, batch_rows in self.batches():
            yield from batch_rows

    def abandoned(self) -> list[BatchRecord]:
        return [
            record
            for record in self._manifest.records()
            if record.state == BatchState.ABANDONED.value
        ]
