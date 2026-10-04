"""Source protocol: where rows come from.

Implementations expose rows lazily through an iterator. The pipeline never
materialises the whole iterable — it pulls one row at a time and the
bounded batching in :mod:`maskhub.pipeline.pipeline` applies backpressure,
so a source with millions of rows cannot inflate process memory.

Resume: when restarting, the pipeline tells the source how many source rows
are already committed. A source that supports seeking (database cursor,
file offset) should reposition directly; a plain iterable falls back to
skipping the prefix.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BatchSource(Protocol):
    def rows(self, start_offset: int = 0) -> Iterator[dict[str, Any]]:
        """Yield rows with source-row offsets >= ``start_offset``.

        ``start_offset`` is the number of source rows already covered by
        committed batches. Sources unable to seek may simply skip that many
        rows of a fresh iteration.
        """
        ...


class IterableSource:
    """Adapter for an in-memory iterable / generator factory.

    Pass either an iterable (re-iterable, e.g. a list) or a zero-argument
    callable returning a fresh iterator (needed for resume with generators).
    """

    def __init__(
        self,
        rows: Iterable[dict[str, Any]]
        | "callable[[], Iterator[dict[str, Any]]]",
    ) -> None:
        self._factory = rows if callable(rows) else lambda: iter(rows)

    def rows(self, start_offset: int = 0) -> Iterator[dict[str, Any]]:
        iterator = self._factory()
        # Fallback resume for non-seekable sources: discard the committed
        # prefix. Ending before the offset simply means there is no new
        # data (never an error, and never a reason to reprocess old rows).
        for _ in range(start_offset):
            try:
                next(iterator)
            except StopIteration:
                return
        yield from iterator
