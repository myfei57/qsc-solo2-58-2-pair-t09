"""Durable record of which batches have been committed.

The manifest is the single source of truth for recovery. State transitions::

    OPEN -> STAGED -> COMMITTED      (happy path)
    OPEN | STAGED -> ABANDONED       (crash leftovers, on resume)

Only ``COMMITTED`` batches are visible to readers.

Durability model: the manifest is an append-only JSON-lines *event log*.
Each transition appends exactly one line (O(1), no full-document rewrite,
which keeps millions of batches cheap). On start the log is replayed and
folded into current batch state. A crash mid-append can only tear the very
last line, which fails to parse and is ignored. The log is compacted
periodically (snapshot via temp file + ``os.replace``) so it cannot grow
without bound.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

MANIFEST_FILENAME = "manifest.jsonl"
# Compact once the log holds more than this many event lines. Each batch
# emits at most a few events, so this bounds the log to a small multiple of
# the number of batches.
COMPACTION_THRESHOLD = 10_000


class BatchState(str, Enum):
    OPEN = "open"
    STAGED = "staged"
    COMMITTED = "committed"
    ABANDONED = "abandoned"


@dataclass(frozen=True)
class BatchRecord:
    sequence: int
    state: str
    row_count: int
    byte_count: int
    opened_at: str
    updated_at: str
    error: str = ""
    # Half-open source-row offsets covered by this batch: [source_start, source_end).
    source_start: int = 0
    source_end: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: object) -> "BatchRecord | None":
        if not isinstance(payload, dict):
            return None
        try:
            record = cls(
                sequence=int(payload["sequence"]),
                state=str(payload["state"]),
                row_count=int(payload["row_count"]),
                byte_count=int(payload["byte_count"]),
                opened_at=str(payload["opened_at"]),
                updated_at=str(payload["updated_at"]),
                error=str(payload.get("error", "")),
                source_start=int(payload.get("source_start", 0)),
                source_end=int(payload.get("source_end", 0)),
            )
        except (KeyError, TypeError, ValueError):
            return None
        if record.state not in {state.value for state in BatchState}:
            return None
        return record


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class ManifestStore(Protocol):
    """Append-event persistence backend for the manifest."""

    def events(self) -> Iterator[dict[str, Any]]:
        """Yield all logged events, oldest first."""
        ...

    def append(self, event: dict[str, Any]) -> None:
        """Durably append one event."""
        ...

    def replace_all(self, events: list[dict[str, Any]]) -> None:
        """Atomically replace the log with the given events (compaction)."""
        ...


class JsonlManifestStore:
    """Append-only JSON-lines file with atomic snapshot compaction."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)

    def events(self) -> Iterator[dict[str, Any]]:
        try:
            raw = self._path.read_bytes().decode("utf-8")
        except FileNotFoundError:
            return
        lines = raw.splitlines()
        for index, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                # Only the final line may be torn by a crash mid-append.
                if index == len(lines) - 1:
                    return
                raise
            if isinstance(event, dict):
                yield event

    def append(self, event: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n"
        with open(self._path, "ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def replace_all(self, events: list[dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(self._path.name + ".tmp")
        payload = b"".join(
            json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n"
            for event in events
        )
        with open(tmp, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self._path)


class Manifest:
    """In-memory batch table guarded by a lock, backed by the event log."""

    def __init__(
        self,
        store: ManifestStore,
        compaction_threshold: int = COMPACTION_THRESHOLD,
    ) -> None:
        self._store = store
        self._threshold = compaction_threshold
        self._lock = threading.RLock()
        self._batches: dict[int, BatchRecord] = {}
        self._event_count = 0
        self._replay()

    def _replay(self) -> None:
        for event in self._store.events():
            self._event_count += 1
            record = BatchRecord.from_dict(event)
            if record is not None:
                self._batches[record.sequence] = record

    def _append_locked(self, record: BatchRecord) -> None:
        self._store.append(record.as_dict())
        self._event_count += 1
        self._batches[record.sequence] = record
        if self._event_count >= self._threshold:
            self._compact_locked()

    def _compact_locked(self) -> None:
        # One folding event per batch, then the log starts small again.
        events = [
            self._batches[sequence].as_dict()
            for sequence in sorted(self._batches)
        ]
        self._store.replace_all(events)
        self._event_count = len(events)

    def records(self) -> list[BatchRecord]:
        with self._lock:
            return [self._batches[key] for key in sorted(self._batches)]

    def get(self, sequence: int) -> BatchRecord | None:
        with self._lock:
            return self._batches.get(sequence)

    def committed(self) -> list[BatchRecord]:
        with self._lock:
            return [
                self._batches[key]
                for key in sorted(self._batches)
                if self._batches[key].state == BatchState.COMMITTED.value
            ]

    def last_committed_sequence(self) -> int:
        """Highest sequence ever committed; 0 when nothing committed.

        Sequence numbers are monotonic *attempt* numbers: a crashed attempt
        stays ABANDONED forever and its retry gets a fresh higher number, so
        committed batches may contain holes. Readers iterate committed
        batches in ascending sequence order.
        """
        with self._lock:
            return max(
                (
                    record.sequence
                    for record in self._batches.values()
                    if record.state == BatchState.COMMITTED.value
                ),
                default=0,
            )

    def next_sequence(self) -> int:
        """Next free attempt number across every state."""
        with self._lock:
            return (
                max(
                    (record.sequence for record in self._batches.values()),
                    default=0,
                )
                + 1
            )

    def committed_source_offset(self) -> int:
        """Rows of the source already covered by committed batches.

        Batches cover ``[source_start, source_end)``; committed batches form
        a prefix of the source (failed attempts are always redone before
        further batches), so this is the maximum ``source_end`` committed.
        """
        with self._lock:
            return max(
                (
                    record.source_end
                    for record in self._batches.values()
                    if record.state == BatchState.COMMITTED.value
                ),
                default=0,
            )

    def open(self, sequence: int) -> BatchRecord:
        with self._lock:
            if sequence in self._batches:
                raise ValueError(f"batch {sequence} already registered")
            now = _utc_now()
            record = BatchRecord(
                sequence=sequence,
                state=BatchState.OPEN.value,
                row_count=0,
                byte_count=0,
                opened_at=now,
                updated_at=now,
            )
            self._append_locked(record)
            return record

    def stage(
        self,
        sequence: int,
        row_count: int,
        byte_count: int,
        source_start: int,
        source_end: int,
    ) -> BatchRecord:
        return self._transition(
            sequence,
            {BatchState.OPEN.value},
            BatchState.STAGED,
            row_count=row_count,
            byte_count=byte_count,
            source_start=source_start,
            source_end=source_end,
        )

    def commit(self, sequence: int) -> BatchRecord:
        return self._transition(
            sequence,
            {BatchState.STAGED.value},
            BatchState.COMMITTED,
        )

    def abandon(self, sequence: int, error: str = "") -> BatchRecord | None:
        with self._lock:
            record = self._batches.get(sequence)
            if record is None or record.state == BatchState.COMMITTED.value:
                return None
            return self._replace_locked(
                record,
                state=BatchState.ABANDONED.value,
                error=error or record.error,
            )

    def recover_pending(self) -> list[int]:
        """Mark crash leftovers abandoned. Returns their sequences."""
        with self._lock:
            pending = [
                key
                for key in sorted(self._batches)
                if self._batches[key].state
                in {BatchState.OPEN.value, BatchState.STAGED.value}
            ]
            for key in pending:
                self._replace_locked(
                    self._batches[key],
                    state=BatchState.ABANDONED.value,
                    error="interrupted before commit",
                )
            return pending

    def _transition(
        self,
        sequence: int,
        allowed: set[str],
        target: BatchState,
        **updates: Any,
    ) -> BatchRecord:
        with self._lock:
            record = self._batches.get(sequence)
            if record is None:
                raise KeyError(f"unknown batch {sequence}")
            if record.state not in allowed:
                raise ValueError(
                    f"batch {sequence} cannot move {record.state} -> {target.value}"
                )
            return self._replace_locked(record, state=target.value, **updates)

    def _replace_locked(
        self, record: BatchRecord, **updates: Any
    ) -> BatchRecord:
        data = record.as_dict()
        data.update(updates)
        data["updated_at"] = _utc_now()
        updated = BatchRecord(
            sequence=data["sequence"],
            state=data["state"],
            row_count=data["row_count"],
            byte_count=data["byte_count"],
            opened_at=data["opened_at"],
            updated_at=data["updated_at"],
            error=data["error"],
            source_start=data["source_start"],
            source_end=data["source_end"],
        )
        self._append_locked(updated)
        return updated
