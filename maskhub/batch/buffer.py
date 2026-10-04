"""Bounded in-memory batching with backpressure for fast producers."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterable, Iterator
from typing import Generic, TypeVar

T = TypeVar("T")
_STOP = object()


class BoundedBatchBuffer(Generic[T]):
    """Group an iterable into lists without allowing the producer to run ahead.

    At most one completed batch is queued and one batch is being assembled.  A
    source that yields millions of rows therefore cannot force the whole input
    set into memory.
    """

    def __init__(
        self,
        source: Iterable[T],
        batch_size: int,
        max_pending_batches: int = 1,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_pending_batches <= 0:
            raise ValueError("max_pending_batches must be positive")
        self._source = source
        self._batch_size = batch_size
        self._queue: queue.Queue[object] = queue.Queue(max_pending_batches)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._producer_error: BaseException | None = None

    def __enter__(self) -> "BoundedBatchBuffer[T]":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("batch buffer already started")
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1)
            self._thread = None

    def batches(self) -> Iterator[list[T]]:
        if self._thread is None:
            raise RuntimeError("batch buffer has not started")
        while True:
            item = self._queue.get()
            if item is _STOP:
                if self._producer_error is not None:
                    raise self._producer_error
                return
            try:
                yield item  # type: ignore[misc]
            except BaseException:
                self.close()
                raise

    def _produce(self) -> None:
        batch: list[T] = []
        try:
            for item in self._source:
                if self._stop.is_set():
                    return
                batch.append(item)
                if len(batch) == self._batch_size:
                    self._put(batch)
                    if self._stop.is_set():
                        return
                    batch = []
            if batch and not self._stop.is_set():
                self._put(batch)
        except BaseException as exc:  # Propagate source/transform failures.
            self._producer_error = exc
        finally:
            self._put_stop()

    def _put(self, batch: list[T]) -> None:
        while not self._stop.is_set():
            try:
                self._queue.put(batch, timeout=0.05)
                return
            except queue.Full:
                continue

    def _put_stop(self) -> None:
        while True:
            try:
                self._queue.put(_STOP, timeout=0.05)
                return
            except queue.Full:
                if self._stop.is_set():
                    # Drain a batch the consumer no longer wants, then publish
                    # the sentinel so a future batches() call exits promptly.
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
