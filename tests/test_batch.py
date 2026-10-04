import tempfile
import threading
import time
import unittest
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from maskhub.batch import (
    BatchJournal,
    BatchRunner,
    BatchStatus,
    BoundedBatchBuffer,
    FileBatchDestination,
    IncompleteReason,
    JobStatus,
)
from maskhub.errors import Conflict
from maskhub.store.kvstore import RecordStore


class CommitRaisesAfterVisible(FileBatchDestination):
    def __init__(self, *args: Any, sequence: int = 1, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.sequence = sequence

    def commit(self, sequence: int) -> Any:
        result = super().commit(sequence)
        if sequence == self.sequence:
            raise RuntimeError("manifest became visible before commit returned")
        return result


class FailStage(FileBatchDestination):
    def __init__(self, *args: Any, sequence: int = 1, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.sequence = sequence
        self.staged: set[int] = set()

    def stage(self, sequence: int, rows: Iterable[Any]) -> Any:
        result = super().stage(sequence, rows)
        self.staged.add(sequence)
        if sequence == self.sequence:
            raise RuntimeError("crash while staging")
        return result


class BatchRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.state_dir = self.root / "state"
        self.out_dir = self.root / "out"
        self.store = RecordStore(self.state_dir)
        self.store.load()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def reload_store(self) -> RecordStore:
        store = RecordStore(self.state_dir)
        store.load()
        return store

    def test_batches_are_committed_as_visible_groups(self) -> None:
        dest = FileBatchDestination(self.out_dir)
        runner = BatchRunner(BatchJournal(self.store), dest, batch_size=3)

        result = runner.run("job", range(8), lambda row: row + 1)

        self.assertTrue(result.completed)
        self.assertEqual(3, result.committed)
        self.assertEqual([0, 1, 2], dest.committed_batches())
        self.assertEqual(
            ["1\n2\n3\n", "4\n5\n6\n", "7\n8\n"],
            [path.read_text() for path in dest.visible_files()],
        )
        # Uncommitted/unfinished data is never part of the visible namespace.
        visible_names = {path.name for path in self.out_dir.iterdir()}
        self.assertNotIn("00000000000000000002.jsonl", visible_names)
        self.assertIn("00000000000000000002.manifest.json", visible_names)

    def test_completed_job_never_reruns_committed_batches(self) -> None:
        class TrackingSource:
            def __init__(self) -> None:
                self.start_positions: list[Any] = []

            def __call__(self, position: Any) -> Iterable[int]:
                self.start_positions.append(position)
                return iter(range(position or 0, 7))

        source = TrackingSource()
        runner = BatchRunner(BatchJournal(self.store), FileBatchDestination(self.out_dir), 3)
        first = runner.run("job", source)

        second = runner.run("job", source)

        self.assertTrue(first.completed)
        self.assertTrue(second.completed)
        self.assertEqual([None], source.start_positions)
        self.assertEqual("job already completed; committed batches were not rerun", second.detail)

    def test_unfinished_batch_is_marked_and_rewritten_after_restart(self) -> None:
        journal = BatchJournal(self.store)
        runner = BatchRunner(journal, FailStage(self.out_dir, sequence=1), 3)

        result = runner.run("job", lambda position: iter(range(position or 0, 8)))

        self.assertFalse(result.completed)
        self.assertEqual((1,), result.incomplete_sequences)
        state = journal.get_batch("job", 1)
        self.assertEqual(BatchStatus.INCOMPLETE.value, state.stage)
        self.assertEqual(IncompleteReason.FAILED.value, state.reason)
        self.assertEqual(JobStatus.FAILED.value, journal.get_job("job").status)

        new_store = self.reload_store()
        new_journal = BatchJournal(new_store)
        recovered = BatchRunner(new_journal, FileBatchDestination(self.out_dir), 3)
        positions: list[Any] = []

        def source(position: Any) -> Iterable[int]:
            positions.append(position)
            return iter(range(position if position is not None else 0, 8))

        restarted = recovered.run("job", source)

        self.assertTrue(restarted.completed)
        self.assertEqual([3], positions)  # Checkpoint is the next source offset.
        self.assertEqual([0, 1, 2], FileBatchDestination(self.out_dir).committed_batches())
        self.assertEqual(
            BatchStatus.COMMITTED.value,
            new_journal.get_batch("job", 1).stage,
        )

    def test_commit_visible_before_journal_is_adopted_not_rerun(self) -> None:
        journal = BatchJournal(self.store)
        dest = FileBatchDestination(self.out_dir)
        journal.create_job("job")
        state0 = journal.mark_processing("job", 0, None, 3, 3)
        dest.stage(0, [1, 2, 3])
        dest.commit(0)
        journal.mark_committed(state0)
        journal.mark_job_running("job", 1, 3)

        # Simulate an OS process crash after destination.commit() but before
        # the journal update: batch one's manifest exists, its journal marker
        # is still in-flight, and the job marker remains running.
        crash_store = self.reload_store()
        crash_journal = BatchJournal(crash_store)
        crash_dest = FileBatchDestination(self.out_dir)
        crash_journal.mark_processing("job", 1, 3, 6, 3)
        crash_dest.stage(1, [4, 5, 6])
        crash_dest.commit(1)

        recovered = BatchRunner(
            BatchJournal(self.reload_store()),
            FileBatchDestination(self.out_dir),
            3,
        )
        starts: list[Any] = []

        def source(position: Any) -> Iterable[int]:
            starts.append(position)
            return iter(range(position if position is not None else 0, 8))

        restarted = recovered.run("job", source, transform=lambda row: row + 1)

        self.assertTrue(restarted.completed)
        self.assertEqual([6], starts)  # Checkpoint is the next source offset.
        self.assertEqual([0, 1, 2], FileBatchDestination(self.out_dir).committed_batches())
        self.assertEqual(
            ["1\n2\n3\n", "4\n5\n6\n", "7\n8\n"],
            [path.read_text() for path in FileBatchDestination(self.out_dir).visible_files()],
        )

    def test_commit_visible_when_commit_raises_after_manifest(self) -> None:
        journal = BatchJournal(self.store)
        dest = CommitRaisesAfterVisible(self.out_dir, sequence=0)
        runner = BatchRunner(journal, dest, 3)

        result = runner.run("job", lambda position: iter(range(position or 0, 8)))

        self.assertTrue(result.completed)
        self.assertEqual(BatchStatus.COMMITTED.value, journal.get_batch("job", 0).stage)
        self.assertEqual([0, 1, 2], FileBatchDestination(self.out_dir).committed_batches())

    def test_process_marked_running_after_crash_is_recovered_as_interrupted(self) -> None:
        journal = BatchJournal(self.store)
        runner = BatchRunner(journal, FileBatchDestination(self.out_dir), 3)
        # Create an in-flight marker as if the process died during processing.
        journal.create_job("job")
        journal.mark_processing("job", 0, None, 2, 3)

        recovered = runner.resume("job")

        self.assertEqual(JobStatus.INTERRUPTED.value, recovered.status)
        self.assertEqual(BatchStatus.INCOMPLETE.value, journal.get_batch("job", 0).stage)
        self.assertEqual(IncompleteReason.INTERRUPTED.value, journal.get_batch("job", 0).reason)

    def test_committed_journal_batch_without_destination_manifest_is_conflict_resolved(self) -> None:
        # The journal should never point after a non-visible batch. Keeping this
        # as Conflict prevents silently rerunning/overwriting unknown output.
        journal = BatchJournal(self.store)
        runner = BatchRunner(journal, FileBatchDestination(self.out_dir), 3)
        journal.create_job("job")
        state = journal.mark_processing("job", 0, None, 0, 1)
        journal.mark_committed(state)

        with self.assertRaises(Conflict):
            runner.run("job", lambda position: iter(range(1)))

    def test_buffer_applies_backpressure(self) -> None:
        batch_size = 10
        yielded = 0
        first_batch_ready = threading.Event()

        def source() -> Iterator[int]:
            nonlocal yielded
            while True:
                value = yielded
                yield value
                yielded += 1
                if value == batch_size - 1:
                    first_batch_ready.set()

        with BoundedBatchBuffer[int](source(), batch_size) as buffer:
            iterator = iter(buffer.batches())
            first_batch_ready.wait(1)
            batch = next(iterator)
            self.assertEqual(list(range(batch_size)), batch)
            # One batch is being processed; one completed and one partial
            # batch can exist on the producer side.
            time.sleep(0.05)
            self.assertLessEqual(yielded, batch_size * 3)
            buffer.close()


if __name__ == "__main__":
    unittest.main()
