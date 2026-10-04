"""Sink protocol: where committed batches land.

A sink stages batch data out of the visible namespace first, then promotes
it as part of the commit. Re-committing an already committed sequence is
refused so a rerun can never overwrite finished work.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from maskhub.pipeline.manifest import BatchState

SUFFIX = ".jsonl"


@runtime_checkable
class BatchSink(Protocol):
    def stage(self, sequence: int, rows: list[dict[str, Any]]) -> int:
        """Persist one batch invisibly. Return its byte size."""
        ...

    def commit(self, sequence: int) -> None:
        """Promote a staged batch to visible storage, atomically."""
        ...

    def discard(self, sequence: int) -> None:
        """Remove staged data for an abandoned/failed batch."""
        ...

    def is_committed(self, sequence: int) -> bool: ...

    def read_committed(self, sequence: int) -> list[dict[str, Any]]: ...


class BatchAlreadyCommitted(RuntimeError):
    """Raised when a finished batch would be overwritten by a rerun."""


class DirectorySink:
    """JSON-lines-per-batch sink in a directory.

    Visible layout::

        <dir>/committed/000001.jsonl
        <dir>/staging/000001.tmp

    Readers only ever look at ``committed``; ``staging`` content is
    invisible and a crash leaves it there to be discarded on recovery.
    """

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self._root = Path(directory)
        self._committed = self._root / "committed"
        self._staging = self._root / "staging"
        self._committed.mkdir(parents=True, exist_ok=True)
        self._staging.mkdir(parents=True, exist_ok=True)

    def _name(self, sequence: int) -> str:
        return f"{sequence:06d}"

    def _staging_path(self, sequence: int) -> Path:
        return self._staging / f"{self._name(sequence)}.tmp"

    def _committed_path(self, sequence: int) -> Path:
        return self._committed / f"{self._name(sequence)}{SUFFIX}"

    def stage(self, sequence: int, rows: list[dict[str, Any]]) -> int:
        if self.is_committed(sequence):
            raise BatchAlreadyCommitted(f"batch {sequence} already committed")
        payload = b"".join(
            json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n"
            for row in rows
        )
        path = self._staging_path(sequence)
        tmp = path.with_name(path.name + ".part")
        with open(tmp, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        return len(payload)

    def commit(self, sequence: int) -> None:
        if self.is_committed(sequence):
            return
        staged = self._staging_path(sequence)
        if not staged.exists():
            raise FileNotFoundError(f"batch {sequence} was never staged")
        target = self._committed_path(sequence)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged, target)

    def discard(self, sequence: int) -> None:
        for path in (self._staging_path(sequence),):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def is_committed(self, sequence: int) -> bool:
        return self._committed_path(sequence).exists()

    def read_committed(self, sequence: int) -> list[dict[str, Any]]:
        path = self._committed_path(sequence)
        if not path.exists():
            raise FileNotFoundError(f"batch {sequence} is not committed")
        rows: list[dict[str, Any]] = []
        for line in path.read_bytes().decode("utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def committed_sequences(self) -> list[int]:
        result = []
        for path in self._committed.glob(f"*{SUFFIX}"):
            try:
                result.append(int(path.name[: -len(SUFFIX)]))
            except ValueError:
                continue
        return sorted(result)

    def clean_stale_staging(self, sequences: list[int]) -> None:
        """Discard staging files left by batches that never committed."""
        for sequence in sequences:
            self.discard(sequence)
