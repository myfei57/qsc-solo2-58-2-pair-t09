"""Behaviour tests for the resumable batch pipeline (stdlib unittest)."""

from __future__ import annotations

import gc
import json
import tempfile
import unittest
from pathlib import Path

from maskhub.pipeline import (
    BatchAlreadyCommitted,
    BatchConfig,
    BatchPipeline,
    BatchRunner,
    BatchState,
    CommittedReader,
    DirectorySink,
    IterableSource,
    Manifest,
)
from maskhub.pipeline.manifest import JsonlManifestStore
from maskhub.pipeline.pipeline import estimate_row_bytes


def rows(n: int):
    return [{"id": i, "payload": f"row-{i}"} for i in range(n)]


class _SimulatedCrash(RuntimeError):
    pass


class CrashAfterStageSink(DirectorySink):
    """Dies right after data was staged, before the manifest flip."""

    def __init__(self, directory, crash_on_sequence: int):
        super().__init__(directory)
        self._crash_on_sequence = crash_on_sequence

    def stage(self, sequence, batch_rows):
        size = super().stage(sequence, batch_rows)
        if sequence == self._crash_on_sequence:
            raise _SimulatedCrash("killed after stage")
        return size


class CrashDuringPromotionSink(DirectorySink):
    """Dies while promoting, after manifest said STAGED."""

    def __init__(self, directory, crash_on_sequence: int):
        super().__init__(directory)
        self._crash_on_sequence = crash_on_sequence

    def commit(self, sequence: int) -> None:
        if sequence == self._crash_on_sequence:
            raise _SimulatedCrash("killed during promotion")
        super().commit(sequence)


def manifest_for(path: Path) -> Manifest:
    return Manifest(JsonlManifestStore(path / "manifest.jsonl"))


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.out = self.tmp / "out"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_commits_everything_in_batches(self) -> None:
        sink = DirectorySink(self.out)
        manifest = manifest_for(self.out)

        result = BatchPipeline(
            IterableSource(rows(10)), sink, manifest, BatchConfig(batch_rows=4)
        ).run()

        self.assertEqual(result.committed_batches, 3)
        self.assertEqual(result.last_committed_sequence, 3)
        committed = manifest.committed()
        self.assertEqual([r.sequence for r in committed], [1, 2, 3])
        self.assertEqual([r.row_count for r in committed], [4, 4, 2])
        self.assertEqual(
            [len(sink.read_committed(seq)) for seq in (1, 2, 3)], [4, 4, 2]
        )
        self.assertEqual(list((self.out / "staging").iterdir()), [])

    def test_batch_is_invisible_until_committed(self) -> None:
        observed = []

        class ObservingSink(DirectorySink):
            def commit(self, sequence: int) -> None:
                observed.append((sequence, sorted(self.committed_sequences())))
                super().commit(sequence)

        sink = ObservingSink(self.out)
        manifest = manifest_for(self.out)
        BatchPipeline(
            IterableSource(rows(6)), sink, manifest, BatchConfig(batch_rows=2)
        ).run()

        self.assertEqual(
            observed,
            [(1, []), (2, [1]), (3, [1, 2])],
        )

    def test_resume_after_crash_continues_without_touching_finished_batches(self):
        sink1 = CrashAfterStageSink(self.out, crash_on_sequence=3)
        with self.assertRaises(_SimulatedCrash):
            BatchPipeline(
                IterableSource(rows(10)),
                sink1,
                manifest_for(self.out),
                BatchConfig(batch_rows=3),
            ).run()

        on_disk = manifest_for(self.out)
        states = {r.sequence: r.state for r in on_disk.records()}
        self.assertEqual(states[1], BatchState.COMMITTED.value)
        self.assertEqual(states[2], BatchState.COMMITTED.value)
        self.assertIn(
            states[3], {BatchState.OPEN.value, BatchState.STAGED.value}
        )
        self.assertEqual(DirectorySink(self.out).committed_sequences(), [1, 2])

        pulled: list = []

        class ReplayedSource:
            def rows(self, start_offset: int = 0):
                pulled.append(("start", start_offset))
                for i in range(start_offset, 10):
                    pulled.append(i)
                    yield rows(10)[i]

        fresh_manifest = manifest_for(self.out)
        result = BatchPipeline(
            ReplayedSource(),
            DirectorySink(self.out),
            fresh_manifest,
            BatchConfig(batch_rows=3),
        ).run()

        abandoned = {
            r.sequence: r
            for r in fresh_manifest.records()
            if r.state == BatchState.ABANDONED.value
        }
        self.assertEqual(set(abandoned), {3})
        self.assertIn("interrupted", abandoned[3].error)
        self.assertEqual(result.started_at_sequence, 4)
        self.assertEqual(result.abandoned_sequences, (3,))
        self.assertEqual(
            [r.sequence for r in fresh_manifest.committed()], [1, 2, 4, 5]
        )

        sink = DirectorySink(self.out)
        self.assertEqual(
            [len(sink.read_committed(seq)) for seq in (1, 2, 4, 5)],
            [3, 3, 3, 1],
        )
        self.assertEqual(pulled, [("start", 6), 6, 7, 8, 9])
        self.assertEqual(list((self.out / "staging").iterdir()), [])

        reader = CommittedReader(sink, fresh_manifest)
        self.assertEqual([row["id"] for row in reader.rows()], list(range(10)))
        self.assertEqual(
            [record.sequence for record, _ in reader.batches()], [1, 2, 4, 5]
        )

    def test_crash_during_promotion_leaves_no_partial_visibility(self):
        with self.assertRaises(_SimulatedCrash):
            BatchPipeline(
                IterableSource(rows(4)),
                CrashDuringPromotionSink(self.out, crash_on_sequence=2),
                manifest_for(self.out),
                BatchConfig(batch_rows=2),
            ).run()

        self.assertEqual(DirectorySink(self.out).committed_sequences(), [1])

        fresh_manifest = manifest_for(self.out)
        result = BatchPipeline(
            IterableSource(rows(4)),
            DirectorySink(self.out),
            fresh_manifest,
            BatchConfig(batch_rows=2),
        ).run()

        self.assertEqual(result.started_at_sequence, 3)
        self.assertEqual(
            [r.sequence for r in fresh_manifest.committed()], [1, 3]
        )
        all_records = {r.sequence: r.state for r in fresh_manifest.records()}
        self.assertEqual(all_records[2], BatchState.ABANDONED.value)

    def test_committed_batches_cannot_be_overwritten(self) -> None:
        sink = DirectorySink(self.out)
        manifest = manifest_for(self.out)
        BatchPipeline(
            IterableSource(rows(6)), sink, manifest, BatchConfig(batch_rows=3)
        ).run()

        with self.assertRaises(BatchAlreadyCommitted):
            sink.stage(1, [{"id": "evil"}])

        foreign = Manifest(JsonlManifestStore(self.tmp / "foreign.jsonl"))
        with self.assertRaisesRegex(RuntimeError, "unknown to manifest"):
            BatchPipeline(
                IterableSource(rows(6)), sink, foreign, BatchConfig(batch_rows=3)
            ).run()

    def test_empty_source_makes_no_batches(self) -> None:
        manifest = manifest_for(self.out)
        result = BatchPipeline(
            IterableSource([]), DirectorySink(self.out), manifest
        ).run()
        self.assertEqual(result.committed_batches, 0)
        self.assertEqual(result.last_committed_sequence, 0)
        self.assertEqual(manifest.records(), [])

    def test_byte_bound_triggers_flush(self) -> None:
        manifest = manifest_for(self.out)
        big = [{"id": i, "blob": "x" * 100} for i in range(5)]
        result = BatchPipeline(
            IterableSource(big),
            DirectorySink(self.out),
            manifest,
            BatchConfig(batch_rows=1000, batch_bytes=200),
        ).run()
        self.assertGreaterEqual(result.committed_batches, 3)
        self.assertTrue(
            all(r.row_count <= 2 for r in manifest.committed())
        )

    def test_memory_is_bounded_for_large_lazy_source(self) -> None:
        observed: list[int] = []
        total_rows = 300_000
        batch_rows = 500

        def huge():
            for i in range(total_rows):
                yield {"id": i}

        class CountingPipeline(BatchPipeline):
            def _commit_group(self, sequence, batch_rows, source_start, source_end):
                observed.append(len(batch_rows))
                assert len(batch_rows) <= 500
                super()._commit_group(
                    sequence, batch_rows, source_start, source_end
                )

        manifest = manifest_for(self.out)
        CountingPipeline(
            IterableSource(huge),
            DirectorySink(self.out),
            manifest,
            BatchConfig(batch_rows=batch_rows),
        ).run()

        expected_batches = total_rows // batch_rows
        sink = DirectorySink(self.out)
        self.assertEqual(
            sink.committed_sequences(), list(range(1, expected_batches + 1))
        )
        self.assertEqual(max(observed), batch_rows)
        self.assertEqual(len(observed), expected_batches)
        gc.collect()
        self.assertEqual(list((self.out / "staging").iterdir()), [])

    def test_runner_resume_and_status(self) -> None:
        run_dir = self.tmp / "run"
        runner = BatchRunner(run_dir, BatchConfig(batch_rows=5))
        status1 = runner.run(IterableSource(rows(12)))
        self.assertEqual(status1.result.committed_batches, 3)
        self.assertEqual(status1.resume_from, 4)

        status2 = runner.run(IterableSource(rows(12)))
        self.assertEqual(status2.result.started_at_sequence, 4)
        self.assertEqual(status2.result.committed_batches, 0)

        records = runner.manifest().committed()
        self.assertEqual([r.sequence for r in records], [1, 2, 3])
        self.assertEqual(sum(r.row_count for r in records), 12)

    def test_manifest_log_is_clean_jsonl(self) -> None:
        runner = BatchRunner(self.tmp / "run", BatchConfig(batch_rows=2))
        runner.run(IterableSource(rows(5)))
        lines = [
            json.loads(line)
            for line in (self.tmp / "run" / "manifest.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]
        # Replaying the event log rebuilds the committed table.
        replayed = runner.manifest()
        self.assertEqual([b.sequence for b in replayed.committed()], [1, 2, 3])
        self.assertTrue(lines)
        leftovers = [
            p.name
            for p in (self.tmp / "run").iterdir()
            if p.suffix == ".tmp"
        ]
        self.assertEqual(leftovers, [])

    def test_torn_final_event_is_ignored_on_replay(self) -> None:
        out = self.tmp / "out"
        manifest = manifest_for(out)
        BatchPipeline(
            IterableSource(rows(2)),
            DirectorySink(out),
            manifest,
            BatchConfig(batch_rows=1),
        ).run()
        log = out / "manifest.jsonl"
        good = log.read_text()
        # Simulate a crash mid-append: a half-written final event line.
        log.write_text(good + '{"sequence": 3, "sta')

        replayed = manifest_for(out)
        self.assertEqual([r.sequence for r in replayed.committed()], [1, 2])
        # The torn batch was never registered; next sequence is 3.
        self.assertEqual(replayed.next_sequence(), 3)

    def test_log_compaction_bounds_size(self) -> None:
        out = self.tmp / "out"
        manifest = Manifest(
            JsonlManifestStore(out / "manifest.jsonl"),
            compaction_threshold=4,
        )
        for seq in range(1, 7):
            manifest.open(seq)
            manifest.stage(seq, 1, 10, seq - 1, seq)
            manifest.commit(seq)
        # After compaction the log holds one folding event per batch only.
        lines = [
            line
            for line in (out / "manifest.jsonl").read_text().splitlines()
            if line.strip()
        ]
        self.assertEqual(len(lines), 6)
        replayed = manifest_for(out)
        self.assertEqual(
            [r.sequence for r in replayed.committed()], [1, 2, 3, 4, 5, 6]
        )

    def test_failed_commit_is_invisible_and_resumable(self) -> None:
        class FailingSink(DirectorySink):
            def commit(self, sequence):
                raise RuntimeError("disk full")

        with self.assertRaisesRegex(RuntimeError, "disk full"):
            BatchPipeline(
                IterableSource(rows(4)),
                FailingSink(self.out),
                manifest_for(self.out),
                BatchConfig(batch_rows=2),
            ).run()

        fresh_manifest = manifest_for(self.out)
        record = fresh_manifest.get(1)
        self.assertIsNotNone(record)
        self.assertEqual(record.state, BatchState.STAGED.value)
        self.assertEqual(DirectorySink(self.out).committed_sequences(), [])

        result = BatchPipeline(
            IterableSource(rows(4)),
            DirectorySink(self.out),
            fresh_manifest,
            BatchConfig(batch_rows=2),
        ).run()
        self.assertEqual(result.abandoned_sequences, (1,))
        self.assertEqual(
            [r.sequence for r in fresh_manifest.committed()], [2, 3]
        )

    def test_config_validation(self) -> None:
        with self.assertRaises(ValueError):
            BatchConfig(batch_rows=0)
        with self.assertRaises(ValueError):
            BatchConfig(batch_bytes=-1)
        self.assertGreater(estimate_row_bytes({"a": 1}), 0)

    def test_iterable_source_resume_skips_prefix(self) -> None:
        source = IterableSource(rows(7))
        self.assertEqual([r["id"] for r in source.rows(0)], list(range(7)))
        self.assertEqual([r["id"] for r in source.rows(5)], [5, 6])
        self.assertEqual(list(source.rows(99)), [])

        # Generator factory also works.
        gen_source = IterableSource(lambda: ({"id": i} for i in range(3)))
        self.assertEqual([r["id"] for r in gen_source.rows(1)], [1, 2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
