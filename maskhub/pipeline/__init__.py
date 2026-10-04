"""Batch-oriented data plane: bounded buffering and group commits.

A :class:`BatchPipeline` pulls rows from a source iterator, groups them into
numbered batches and hands each batch to a sink. A batch becomes visible
atomically: its data is staged first and the manifest flip to ``committed``
is the single commit point. After a crash the run resumes at a fresh batch
number following the last committed one; partially written batches are
marked ``abandoned`` and never partially observed.
"""

from maskhub.pipeline.manifest import (
    BatchRecord,
    BatchState,
    JsonlManifestStore,
    Manifest,
    ManifestStore,
)
from maskhub.pipeline.pipeline import BatchConfig, BatchPipeline, PipelineResult
from maskhub.pipeline.reader import CommittedReader
from maskhub.pipeline.runner import BatchRunner
from maskhub.pipeline.sink import (
    BatchAlreadyCommitted,
    BatchSink,
    DirectorySink,
)
from maskhub.pipeline.source import BatchSource, IterableSource

__all__ = [
    "BatchAlreadyCommitted",
    "BatchConfig",
    "BatchRecord",
    "BatchRunner",
    "BatchPipeline",
    "BatchSink",
    "BatchSource",
    "BatchState",
    "CommittedReader",
    "DirectorySink",
    "IterableSource",
    "JsonlManifestStore",
    "Manifest",
    "ManifestStore",
    "PipelineResult",
]
