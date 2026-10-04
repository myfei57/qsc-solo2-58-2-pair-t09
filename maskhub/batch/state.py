"""Persistent state for recoverable batch jobs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from maskhub.errors import Conflict, NotFound
from maskhub.store.kvstore import RecordStore
from maskhub.store.query import keys_with_prefix

JOB_PREFIX = "batch.job."
STATE_PREFIX = "batch.state."


class JobStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class BatchStatus(str, Enum):
    PROCESSING = "processing"
    STAGED = "staged"
    COMMITTED = "committed"
    INCOMPLETE = "incomplete"


class IncompleteReason(str, Enum):
    INTERRUPTED = "interrupted"
    FAILED = "failed"


@dataclass(frozen=True)
class JobState:
    job_id: str
    status: str
    next_sequence: int = 0
    resume_position: Any | None = None
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: object) -> "JobState | None":
        if not isinstance(payload, dict):
            return None
        job_id = payload.get("job_id")
        status = payload.get("status")
        next_sequence = payload.get("next_sequence", 0)
        resume_position = payload.get("resume_position")
        detail = payload.get("detail", "")
        if not isinstance(job_id, str) or not isinstance(status, str):
            return None
        if not isinstance(next_sequence, int) or isinstance(next_sequence, bool):
            return None
        if next_sequence < 0 or not isinstance(detail, str):
            return None
        return cls(
            job_id=job_id,
            status=status,
            next_sequence=next_sequence,
            resume_position=resume_position,
            detail=detail,
        )


@dataclass(frozen=True)
class BatchState:
    job_id: str
    sequence: int
    stage: str
    start_position: Any | None = None
    end_position: Any | None = None
    row_count: int = 0
    reason: str = ""
    detail: str = ""

    @property
    def is_committed(self) -> bool:
        return self.stage == BatchStatus.COMMITTED.value

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: object) -> "BatchState | None":
        if not isinstance(payload, dict):
            return None
        job_id = payload.get("job_id")
        sequence = payload.get("sequence")
        stage = payload.get("stage")
        row_count = payload.get("row_count", 0)
        reason = payload.get("reason", "")
        detail = payload.get("detail", "")
        if not isinstance(job_id, str) or not isinstance(stage, str):
            return None
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
            return None
        if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count < 0:
            return None
        if not isinstance(reason, str) or not isinstance(detail, str):
            return None
        return cls(
            job_id=job_id,
            sequence=sequence,
            stage=stage,
            start_position=payload.get("start_position"),
            end_position=payload.get("end_position"),
            row_count=row_count,
            reason=reason,
            detail=detail,
        )


def job_key(job_id: str) -> str:
    return JOB_PREFIX + job_id


def batch_key(job_id: str, sequence: int) -> str:
    return f"{STATE_PREFIX}{job_id}.{sequence}"


class BatchJournal:
    """Stores one job record and one immutable-at-commit record per batch."""

    def __init__(self, store: RecordStore) -> None:
        self._store = store

    def create_job(self, job_id: str) -> JobState:
        if self._store.has(job_key(job_id)):
            raise Conflict(f"batch job {job_id} already exists")
        job = JobState(job_id=job_id, status=JobStatus.RUNNING.value)
        self._store.put_json(job_key(job_id), job.as_dict())
        return job

    def get_job(self, job_id: str) -> JobState:
        state = JobState.from_dict(self._store.get_json(job_key(job_id)))
        if state is None:
            raise NotFound(f"unknown batch job {job_id}")
        return state

    def has_job(self, job_id: str) -> bool:
        return self._store.has(job_key(job_id))

    def save_job(self, job: JobState) -> None:
        self._store.put_json(job_key(job.job_id), job.as_dict())

    def get_or_create_job(self, job_id: str) -> JobState:
        if self._store.has(job_key(job_id)):
            return self.get_job(job_id)
        return self.create_job(job_id)

    def get_batch(self, job_id: str, sequence: int) -> BatchState:
        state = BatchState.from_dict(self._store.get_json(batch_key(job_id, sequence)))
        if state is None:
            raise NotFound(f"unknown batch {sequence} for job {job_id}")
        return state

    def save_batch(self, state: BatchState) -> None:
        self._store.put_json(batch_key(state.job_id, state.sequence), state.as_dict())

    def list_batches(self, job_id: str) -> list[BatchState]:
        prefix = STATE_PREFIX + job_id + "."
        states: list[BatchState] = []
        for key in keys_with_prefix(self._store, prefix):
            sequence_text = key.rsplit(".", 1)[-1]
            try:
                sequence = int(sequence_text)
            except ValueError:
                continue
            state = BatchState.from_dict(self._store.get_json(key))
            if state is not None and state.job_id == job_id and state.sequence == sequence:
                states.append(state)
        states.sort(key=lambda state: state.sequence)
        return states

    def mark_processing(
        self,
        job_id: str,
        sequence: int,
        start_position: Any | None,
        end_position: Any | None,
        row_count: int,
    ) -> BatchState:
        if self._store.has(batch_key(job_id, sequence)):
            existing = self.get_batch(job_id, sequence)
            if existing.stage == BatchStatus.COMMITTED.value:
                raise Conflict(f"batch {sequence} for job {job_id} is already committed")
        state = BatchState(
            job_id=job_id,
            sequence=sequence,
            stage=BatchStatus.PROCESSING.value,
            start_position=start_position,
            end_position=end_position,
            row_count=row_count,
        )
        self.save_batch(state)
        return state

    def mark_staged(self, state: BatchState) -> BatchState:
        updated = _replace(state, stage=BatchStatus.STAGED.value, reason="", detail="")
        self.save_batch(updated)
        return updated

    def mark_committed(self, state: BatchState) -> BatchState:
        updated = _replace(state, stage=BatchStatus.COMMITTED.value, reason="", detail="")
        self.save_batch(updated)
        return updated

    def mark_incomplete(
        self, state: BatchState, reason: IncompleteReason, detail: str = ""
    ) -> BatchState:
        if state.is_committed:
            raise Conflict(
                f"batch {state.sequence} for job {state.job_id} is already committed"
            )
        updated = _replace(
            state,
            stage=BatchStatus.INCOMPLETE.value,
            reason=reason.value,
            detail=detail,
        )
        self.save_batch(updated)
        return updated

    def mark_job_running(
        self, job_id: str, next_sequence: int, resume_position: Any | None
    ) -> JobState:
        job = JobState(
            job_id=job_id,
            status=JobStatus.RUNNING.value,
            next_sequence=next_sequence,
            resume_position=resume_position,
        )
        self.save_job(job)
        return job

    def mark_job_completed(self, job: JobState) -> JobState:
        completed = JobState(
            job_id=job.job_id,
            status=JobStatus.COMPLETED.value,
            next_sequence=job.next_sequence,
            resume_position=job.resume_position,
        )
        self.save_job(completed)
        return completed

    def mark_job_failed(self, job: JobState, detail: str) -> JobState:
        failed = JobState(
            job_id=job.job_id,
            status=JobStatus.FAILED.value,
            next_sequence=job.next_sequence,
            resume_position=job.resume_position,
            detail=detail,
        )
        self.save_job(failed)
        return failed

    def mark_job_interrupted(self, job: JobState, detail: str = "") -> JobState:
        interrupted = JobState(
            job_id=job.job_id,
            status=JobStatus.INTERRUPTED.value,
            next_sequence=job.next_sequence,
            resume_position=job.resume_position,
            detail=detail,
        )
        self.save_job(interrupted)
        return interrupted


def _replace(state: BatchState, **changes: Any) -> BatchState:
    return BatchState(**{**state.as_dict(), **changes})
