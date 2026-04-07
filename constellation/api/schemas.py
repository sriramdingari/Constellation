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
    # Canonical fields — prefer these for new clients and monitoring.
    # `backend` defaults to "disconnected" because the route's exception
    # path relies on the default. `backend_type` has NO default: any code
    # path that constructs HealthResponse must explicitly state the active
    # backend, since a silent default would lie on Postgres deployments.
    backend: str = "disconnected"
    backend_type: str
    # Legacy alias kept for backward compatibility with existing clients
    # and dashboards. Populated with the same value as `backend`.
    # Deprecated; use `backend` in new code.
    neo4j: str
    redis: str
