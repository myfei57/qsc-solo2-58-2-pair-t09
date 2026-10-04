"""Atomic batch destinations.

The file destination exposes a batch only by writing its manifest after the
batch's immutable data file has been put in place.  Consumers discover batches
through manifests; data files live under ``.objects`` and a partial file left
behind by a crash is ignored until the batch is either completed or discarded.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol

from maskhub.errors import Conflict


class BatchDestination(Protocol):
    def stage(self, sequence: int, rows: Iterable[Any]) -> Any: ...

    def commit(self, sequence: int) -> Any: ...

    def discard(self, sequence: int) -> None: ...

    def is_committed(self, sequence: int) -> bool: ...


def default_serialize(row: Any) -> str:
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"))


class FileBatchDestination:
    """Line-oriented, manifest-indexed batch files."""

    def __init__(
        self,
        directory: str | os.PathLike[str],
        serialize: Callable[[Any], str] = default_serialize,
    ) -> None:
        self._directory = Path(directory)
        self._staging = self._directory / ".staging"
        self._objects = self._directory / ".objects"
        self._serialize = serialize

    def stage(self, sequence: int, rows: Iterable[Any]) -> Path:
        if self.is_committed(sequence):
            raise Conflict(f"batch {sequence} is already committed")
        self._directory.mkdir(parents=True, exist_ok=True)
        self._staging.mkdir(parents=True, exist_ok=True)
        final = self._data_path(sequence)
        staged = self._staging / f"{sequence:020d}.jsonl"
        self._unlink(final)
        self._unlink(staged)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=self._staging,
            delete=False,
        ) as tmp:
            temporary = Path(tmp.name)
            try:
                for row in rows:
                    tmp.write(self._serialize(row))
                    tmp.write("\n")
                tmp.flush()
                os.replace(tmp.name, staged)
                temporary = None
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return staged

    def commit(self, sequence: int) -> Path:
        manifest_path = self._manifest_path(sequence)
        if manifest_path.exists():
            return manifest_path

        staged = self._staging / f"{sequence:020d}.jsonl"
        if not staged.is_file():
            raise FileNotFoundError(f"staged batch {sequence} is missing")
        data_path = self._data_path(sequence)
        self._objects.mkdir(parents=True, exist_ok=True)
        os.replace(staged, data_path)

        manifest = {
            "sequence": sequence,
            "data": str(data_path.relative_to(self._directory)),
            "visible": True,
        }
        payload = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=self._directory, delete=False) as tmp:
                temporary = Path(tmp.name)
                tmp.write(payload)
            os.replace(tmp.name, manifest_path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return manifest_path

    def discard(self, sequence: int) -> None:
        if self.is_committed(sequence):
            raise Conflict(f"batch {sequence} is already committed")
        self._unlink(self._staging / f"{sequence:020d}.jsonl")
        self._unlink(self._data_path(sequence))

    def is_committed(self, sequence: int) -> bool:
        return self._manifest_path(sequence).is_file()

    def committed_batches(self) -> list[int]:
        if not self._directory.is_dir():
            return []
        sequences: list[int] = []
        for path in self._directory.glob("*.manifest.json"):
            try:
                sequences.append(int(path.name.removesuffix(".manifest.json")))
            except ValueError:
                continue
        sequences.sort()
        return sequences

    def visible_files(self) -> list[Path]:
        files: list[Path] = []
        for sequence in self.committed_batches():
            path = self._data_path(sequence)
            if path.is_file():
                files.append(path)
        return files

    def _data_path(self, sequence: int) -> Path:
        return self._objects / f"{sequence:020d}.jsonl"

    def _manifest_path(self, sequence: int) -> Path:
        return self._directory / f"{sequence:020d}.manifest.json"

    @staticmethod
    def _unlink(path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
