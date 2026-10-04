"""Convenience entry point wiring a sink directory to a pipeline run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from maskhub.pipeline.manifest import (
    JsonlManifestStore,
    Manifest,
    MANIFEST_FILENAME,
)
from maskhub.pipeline.pipeline import BatchConfig, BatchPipeline, PipelineResult
from maskhub.pipeline.sink import DirectorySink
from maskhub.pipeline.source import BatchSource


@dataclass(frozen=True)
class RunStatus:
    result: PipelineResult
    manifest_path: Path

    @property
    def resume_from(self) -> int:
        """Next sequence a subsequent run will process."""
        return self.result.last_committed_sequence + 1


class BatchRunner:
    """Run batches into a directory and remember progress durably."""

    def __init__(
        self,
        directory: str | Path,
        config: BatchConfig | None = None,
    ) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._directory / MANIFEST_FILENAME
        self._config = config or BatchConfig()

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    def run(self, source: BatchSource) -> RunStatus:
        manifest = Manifest(JsonlManifestStore(self._manifest_path))
        sink = DirectorySink(self._directory)
        pipeline = BatchPipeline(source, sink, manifest, self._config)
        result = pipeline.run()
        return RunStatus(result=result, manifest_path=self._manifest_path)

    def manifest(self) -> Manifest:
        return Manifest(JsonlManifestStore(self._manifest_path))
