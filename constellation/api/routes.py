"""FastAPI routes for the Constellation API."""

from __future__ import annotations

import redis
from uuid import uuid4
from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException

from constellation.api.schemas import (
    HealthResponse,
    IndexRequest,
    IndexResponse,
    JobProgress,
    JobStatus,
    RepositoryInfo,
)
from constellation.config import get_settings
from constellation.graph.client import GraphClient
from constellation.indexer.collector import derive_repo_name
from constellation.locking import (
    INDEX_LOCK_TTL_SECONDS,
    acquire_dispatch_claim,
    release_claim,
    start_claim_heartbeat,
)
from constellation.worker.celery_app import celery_app
from constellation.worker.tasks import index_repository

router = APIRouter()

# Map Celery states to API-facing status strings.
_STATE_MAP = {
    "PENDING": "queued",
    "STARTED": "in_progress",
    "PROGRESS": "in_progress",
    "SUCCESS": "completed",
    "FAILURE": "failed",
    "RETRY": "queued",
    "REVOKED": "failed",
}


async def _get_graph_client() -> GraphClient:
    """Create and connect a GraphClient. Caller must close it."""
    settings = get_settings()
    client = GraphClient(settings)
    await client.connect()
    return client


@router.post("/repositories/index", status_code=202, response_model=IndexResponse)
async def index_repo(request: IndexRequest):
    """Dispatch a repository indexing task."""
    repo_name = request.name or derive_repo_name(request.source)

    settings = get_settings()
    r = redis.from_url(settings.redis_url)
    lock_key = f"constellation:lock:{repo_name}"
    lock_token = uuid4().hex
    if not acquire_dispatch_claim(r, lock_key, lock_token):
        raise HTTPException(
            status_code=409,
            detail=f"Indexing already in progress for {repo_name}",
        )

    try:
        task = index_repository.delay(
            request.source,
            request.name,
            request.exclude_patterns,
            request.reindex,
            lock_token=lock_token,
        )
    except Exception:
        release_claim(r, lock_key, lock_token)
        raise

    def _task_is_still_queued() -> bool:
        return AsyncResult(task.id, app=celery_app).state in {"PENDING", "RETRY"}

    start_claim_heartbeat(
        r,
        lock_key,
        lock_token,
        should_continue=_task_is_still_queued,
        max_lifetime=INDEX_LOCK_TTL_SECONDS,
    )
    return IndexResponse(job_id=task.id, repository=repo_name)


@router.get("/repositories", response_model=list[RepositoryInfo])
async def list_repos():
    """List all indexed repositories."""
    client = await _get_graph_client()
    try:
        repos = await client.list_repositories()
        return [RepositoryInfo(**repo) for repo in repos]
    finally:
        await client.close()


@router.get("/repositories/{name}", response_model=RepositoryInfo)
async def get_repo(name: str):
    """Get information about a specific repository."""
    client = await _get_graph_client()
    try:
        repo = await client.get_repository(name)
        if repo is None:
            raise HTTPException(
                status_code=404, detail=f"Repository '{name}' not found"
            )
        return RepositoryInfo(**repo)
    finally:
        await client.close()


@router.delete("/repositories/{name}", status_code=204)
async def delete_repo(name: str):
    """Delete a repository and all its indexed data."""
    client = await _get_graph_client()
    try:
        repo = await client.get_repository(name)
        if repo is None:
            raise HTTPException(
                status_code=404, detail=f"Repository '{name}' not found"
            )
        await client.delete_repository(name)
    finally:
        await client.close()


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    """Get the status of an indexing job."""
    result = AsyncResult(job_id, app=celery_app)
    state = result.state
    status = _STATE_MAP.get(state, "unknown")

    progress = None
    job_result = None
    error = None

    if state == "PROGRESS" and isinstance(result.info, dict):
        progress = JobProgress(
            files_total=result.info.get("files_total", 0),
            files_processed=result.info.get("files_processed", 0),
            entities_found=result.info.get("entities_found", 0),
        )
    elif state == "SUCCESS":
        job_result = result.result
    elif state == "FAILURE":
        error = str(result.result)

    return JobStatus(
        job_id=job_id,
        status=status,
        progress=progress,
        result=job_result,
        error=error,
    )


@router.get("/health", response_model=HealthResponse)
async def health():
    """Check health of Neo4j and Redis dependencies."""
    neo4j_status = "disconnected"
    redis_status = "disconnected"

    # Check Neo4j
    try:
        client = await _get_graph_client()
        try:
            neo4j_status = "connected"
        finally:
            await client.close()
    except Exception:
        neo4j_status = "disconnected"

    # Check Redis
    try:
        settings = get_settings()
        r = redis.from_url(settings.redis_url)
        r.ping()
        redis_status = "connected"
    except Exception:
        redis_status = "disconnected"

    overall = "ok" if neo4j_status == "connected" and redis_status == "connected" else "degraded"
    return HealthResponse(status=overall, neo4j=neo4j_status, redis=redis_status)
