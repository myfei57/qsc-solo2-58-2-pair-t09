"""Run a source through bounded, durable batch commits."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from maskhub.batch.buffer import BoundedBatchBuffer
from maskhub.batch.destination import BatchDestination
from maskhub.batch.state import (
    BatchJournal,
    BatchState,
    BatchStatus,
    IncompleteReason,
    JobState,
    JobStatus,
)
from maskhub.errors import Conflict

Position = Any
PositionedRow = tuple[Position, Any]
BatchSource = Callable[[Position], Iterable[Any]]
Transform = Callable[[Any], Any]


@dataclass(frozen=True)
class BatchResult:
    committed: int
    next_sequence: int
    resume_position: Position
    completed: bool
    incomplete_sequences: tuple[int, ...] = ()
    detail: str = ""


class BatchRunner:
    """Commit one bounded batch at a time and resume after crashes.

    A callable source receives the checkpoint recorded after the last complete
    batch.  Integer checkpoints are interpreted as source offsets; sources with
    opaque checkpoints must yield ``(checkpoint, row)`` pairs.
    """

    def __init__(
        self,
        journal: BatchJournal,
        destination: BatchDestination,
        batch_size: int = 1000,
        max_pending_batches: int = 1,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._journal = journal
        self._destination = destination
        self._batch_size = batch_size
        self._max_pending_batches = max_pending_batches

    def resume(self, job_id: str) -> JobState:
        """Mark records left behind by a previous crashed process."""
        existed = self._journal.has_job(job_id)
        job = self._journal.get_or_create_job(job_id)
        if not existed or job.status != JobStatus.RUNNING.value:
            return job

        incomplete: list[int] = []
        for state in self._journal.list_batches(job_id):
            if not state.is_committed and self._destination.is_committed(state.sequence):
                state = self._journal.mark_committed(state)
            if not state.is_committed:
                self._journal.mark_incomplete(
                    state,
                    IncompleteReason.INTERRUPTED,
                    "batch was not committed before the process exited",
                )
                incomplete.append(state.sequence)
                break

        detail = (
            f"interrupted before batch(es): {incomplete}"
            if incomplete
            else "process interrupted before any batch completed"
        )
        return self._journal.mark_job_interrupted(job, detail)

    def run(
        self,
        job_id: str,
        source: BatchSource | Iterable[Any],
        transform: Transform | None = None,
    ) -> BatchResult:
        """Recover the job, then process all source rows until it ends."""
        recovered = self.resume(job_id)
        if recovered.status == JobStatus.COMPLETED.value:
            return BatchResult(
                committed=recovered.next_sequence,
                next_sequence=recovered.next_sequence,
                resume_position=recovered.resume_position,
                completed=True,
                detail="job already completed; committed batches were not rerun",
            )

        next_sequence, resume_position = self._checkpoint(recovered)
        self._journal.mark_job_running(job_id, next_sequence, resume_position)
        job = self._journal.get_job(job_id)
        records: Iterable[Any]
        if callable(source):
            records = source(resume_position)
        elif next_sequence == 0:
            records = source
        else:
            raise ValueError(
                "source must be a callable accepting a resume position to continue this job"
            )

        committed = next_sequence
        incomplete: list[int] = []
        completed = False
        detail = ""
        try:
            positioned = _with_positions(records, resume_position)
            buffer = BoundedBatchBuffer[PositionedRow](
                positioned,
                self._batch_size,
                self._max_pending_batches,
            )
            with buffer:
                for batch in buffer.batches():
                    start_position, _ = batch[0]
                    last_position, _ = batch[-1]
                    end_position = _advance_position(last_position)
                    sequence = next_sequence
                    state = self._journal.mark_processing(
                        job_id,
                        sequence,
                        start_position,
                        end_position,
                        len(batch),
                    )
                    try:
                        transformed = _transform_rows(batch, transform)
                        self._destination.stage(sequence, transformed)
                        state = self._journal.mark_staged(state)
                        state = self._commit_batch(state)
                    except Exception as exc:
                        if state.is_committed:
                            raise
                        state = self._journal.mark_incomplete(
                            state,
                            IncompleteReason.FAILED,
                            str(exc) or exc.__class__.__name__,
                        )
                        try:
                            self._destination.discard(sequence)
                        except Exception:
                            pass
                        incomplete.append(sequence)
                        detail = f"batch {sequence} failed: {state.detail}"
                        job = self._journal.get_job(job_id)
                        self._journal.mark_job_failed(job, detail)
                        return BatchResult(
                            committed,
                            next_sequence,
                            resume_position,
                            False,
                            tuple(incomplete),
                            detail,
                        )
                    next_sequence += 1
                    committed = next_sequence
                    resume_position = state.end_position
                    job = self._journal.get_job(job_id)
                    self._journal.mark_job_running(job_id, next_sequence, resume_position)

            unfinished = [
                state.sequence
                for state in self._journal.list_batches(job_id)
                if not state.is_committed and state.sequence >= committed
            ]
            if unfinished:
                detail = f"source ended without replaying unfinished batch(es): {unfinished}"
                job = self._journal.get_job(job_id)
                self._journal.mark_job_failed(job, detail)
                return BatchResult(
                    committed,
                    next_sequence,
                    resume_position,
                    False,
                    tuple(unfinished),
                    detail,
                )

            completed = True
            job = self._journal.get_job(job_id)
            self._journal.mark_job_completed(job)
        except Exception as exc:
            detail = str(exc) or exc.__class__.__name__
            job = self._journal.get_job(job_id)
            self._journal.mark_job_failed(job, detail)
            return BatchResult(
                committed,
                next_sequence,
                resume_position,
                False,
                tuple(incomplete),
                detail,
            )

        return BatchResult(
            committed,
            next_sequence,
            resume_position,
            completed,
            tuple(incomplete),
            detail,
        )

    def _checkpoint(self, job: JobState) -> tuple[int, Position]:
        states = self._journal.list_batches(job.job_id)
        committed = 0
        resume_position: Position = job.resume_position

        for state in states:
            if state.is_committed:
                if state.sequence != committed:
                    raise Conflict(
                        f"batch {state.sequence} is committed before batch {committed}"
                    )
                if not self._destination.is_committed(state.sequence):
                    raise Conflict(
                        f"batch {state.sequence} is marked committed but is not visible"
                    )
                resume_position = state.end_position
                committed += 1
                continue

            if state.sequence != committed:
                raise Conflict(
                    f"batch {state.sequence} exists before next batch {committed}"
                )
            break

        return committed, resume_position

    def _commit_batch(self, state: BatchState) -> BatchState:
        sequence = state.sequence
        if self._destination.is_committed(sequence):
            # The journal had not advanced when the previous process exited.
            return self._journal.mark_committed(state)

        try:
            self._destination.commit(sequence)
        except Exception:
            if self._destination.is_committed(sequence):
                return self._journal.mark_committed(state)
            raise
        return self._journal.mark_committed(state)


def _advance_position(position: Position) -> Position:
    return position + 1 if isinstance(position, int) else position


def _with_positions(
    rows: Iterable[Any], start: Position
) -> Iterator[PositionedRow]:
    position = 0 if start is None else start
    if isinstance(position, int):
        for row in rows:
            yield position, row
            position += 1
        return

    # Sources with opaque positions must emit ``(position, row)`` pairs.
    for item in rows:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("opaque-position sources must yield (position, row) pairs")
        position, row = item
        yield position, row


def _transform_rows(
    batch: list[PositionedRow], transform: Transform | None
) -> Iterator[Any]:
    if transform is None:
        return (row for _, row in batch)
    return (transform(row) for _, row in batch)
