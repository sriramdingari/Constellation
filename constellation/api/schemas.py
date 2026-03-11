"""Pydantic schemas for the Constellation API."""

from pydantic import BaseModel


class IndexRequest(BaseModel):
    source: str
    name: str | None = None
    exclude_patterns: list[str] | None = None
    reindex: bool = False


class IndexResponse(BaseModel):
    job_id: str
    repository: str
    status: str = "queued"


class JobProgress(BaseModel):
    files_total: int = 0
    files_processed: int = 0
    entities_found: int = 0


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: JobProgress | None = None
    result: dict | None = None
    error: str | None = None


class RepositoryInfo(BaseModel):
    name: str
    source: str | None = None
    last_indexed_at: str | None = None
    last_commit_sha: str | None = None
    entity_count: int = 0
    languages: list[str] | None = None


class HealthResponse(BaseModel):
    status: str
    neo4j: str
    redis: str
