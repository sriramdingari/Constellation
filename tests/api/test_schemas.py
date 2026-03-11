"""Tests for API Pydantic schemas."""

from constellation.api.schemas import (
    HealthResponse,
    IndexRequest,
    IndexResponse,
    JobProgress,
    JobStatus,
    RepositoryInfo,
)


class TestIndexRequest:
    def test_minimal_creation(self):
        req = IndexRequest(source="/path/to/repo")
        assert req.source == "/path/to/repo"
        assert req.name is None
        assert req.exclude_patterns is None
        assert req.reindex is False

    def test_full_creation(self):
        req = IndexRequest(
            source="https://github.com/owner/repo",
            name="my-repo",
            exclude_patterns=["*.min.js", "node_modules"],
            reindex=True,
        )
        assert req.source == "https://github.com/owner/repo"
        assert req.name == "my-repo"
        assert req.exclude_patterns == ["*.min.js", "node_modules"]
        assert req.reindex is True


class TestIndexResponse:
    def test_defaults(self):
        resp = IndexResponse(job_id="abc-123", repository="my-repo")
        assert resp.job_id == "abc-123"
        assert resp.repository == "my-repo"
        assert resp.status == "queued"

    def test_custom_status(self):
        resp = IndexResponse(
            job_id="abc-123", repository="my-repo", status="in_progress"
        )
        assert resp.status == "in_progress"


class TestJobStatus:
    def test_queued_state(self):
        status = JobStatus(job_id="abc-123", status="queued")
        assert status.job_id == "abc-123"
        assert status.status == "queued"
        assert status.progress is None
        assert status.result is None
        assert status.error is None

    def test_in_progress_state(self):
        status = JobStatus(job_id="abc-123", status="in_progress")
        assert status.status == "in_progress"

    def test_completed_state(self):
        status = JobStatus(
            job_id="abc-123",
            status="completed",
            result={"entities_created": 42},
        )
        assert status.status == "completed"
        assert status.result == {"entities_created": 42}

    def test_failed_state(self):
        status = JobStatus(
            job_id="abc-123",
            status="failed",
            error="Connection refused",
        )
        assert status.status == "failed"
        assert status.error == "Connection refused"

    def test_with_progress(self):
        progress = JobProgress(
            files_total=100, files_processed=50, entities_found=200
        )
        status = JobStatus(
            job_id="abc-123", status="in_progress", progress=progress
        )
        assert status.progress is not None
        assert status.progress.files_total == 100
        assert status.progress.files_processed == 50
        assert status.progress.entities_found == 200


class TestJobProgress:
    def test_defaults(self):
        progress = JobProgress()
        assert progress.files_total == 0
        assert progress.files_processed == 0
        assert progress.entities_found == 0


class TestRepositoryInfo:
    def test_defaults(self):
        info = RepositoryInfo(name="my-repo")
        assert info.name == "my-repo"
        assert info.source is None
        assert info.last_indexed_at is None
        assert info.last_commit_sha is None
        assert info.entity_count == 0
        assert info.languages is None

    def test_full_creation(self):
        info = RepositoryInfo(
            name="my-repo",
            source="https://github.com/owner/repo",
            last_indexed_at="2025-01-01T00:00:00Z",
            last_commit_sha="abc123def",
            entity_count=150,
            languages=["python", "java"],
        )
        assert info.source == "https://github.com/owner/repo"
        assert info.last_indexed_at == "2025-01-01T00:00:00Z"
        assert info.last_commit_sha == "abc123def"
        assert info.entity_count == 150
        assert info.languages == ["python", "java"]


class TestHealthResponse:
    def test_creation(self):
        health = HealthResponse(status="ok", neo4j="connected", redis="connected")
        assert health.status == "ok"
        assert health.neo4j == "connected"
        assert health.redis == "connected"

    def test_degraded(self):
        health = HealthResponse(
            status="degraded", neo4j="disconnected", redis="connected"
        )
        assert health.status == "degraded"
        assert health.neo4j == "disconnected"
