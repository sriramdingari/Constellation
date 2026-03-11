"""Celery tasks for repository indexing."""

from __future__ import annotations

import asyncio

import redis

from constellation.config import get_settings
from constellation.embeddings.factory import create_embedding_provider
from constellation.graph.client import GraphClient
from constellation.indexer.collector import derive_repo_name
from constellation.indexer.pipeline import IndexingPipeline
from constellation.parsers.registry import get_default_registry
from constellation.worker.celery_app import celery_app

settings = get_settings()


@celery_app.task(bind=True, max_retries=2)
def index_repository(
    self,
    source: str,
    name: str | None = None,
    exclude_patterns: list[str] | None = None,
    reindex: bool = False,
):
    """Index a repository. Uses Redis lock for concurrency control."""
    repo_name = name or derive_repo_name(source)
    lock_key = f"constellation:lock:{repo_name}"

    r = redis.from_url(settings.redis_url)
    lock = r.lock(lock_key, timeout=3600)  # 1 hour timeout

    if not lock.acquire(blocking=False):
        raise RuntimeError(f"Indexing already in progress for {repo_name}")

    try:

        def progress_callback(files_total, files_processed, entities_found):
            self.update_state(
                state="PROGRESS",
                meta={
                    "files_total": files_total,
                    "files_processed": files_processed,
                    "entities_found": entities_found,
                },
            )

        result = asyncio.run(
            _run_indexing(source, name, exclude_patterns, reindex, progress_callback)
        )
        return {
            "repository": result.repository,
            "files_total": result.files_total,
            "files_processed": result.files_processed,
            "files_skipped": result.files_skipped,
            "entities_created": result.entities_created,
            "relationships_created": result.relationships_created,
            "errors": result.errors,
        }
    except Exception as exc:
        try:
            raise self.retry(exc=exc, countdown=2 ** self.request.retries * 10)
        except self.MaxRetriesExceededError:
            raise exc
    finally:
        try:
            lock.release()
        except redis.exceptions.LockNotOwnedError:
            pass


async def _run_indexing(source, name, exclude_patterns, reindex, progress_callback):
    """Run the indexing pipeline asynchronously."""
    task_settings = get_settings()
    graph_client = GraphClient(task_settings)
    embedding_provider = create_embedding_provider(
        task_settings.embedding_provider, task_settings
    )
    parser_registry = get_default_registry()

    try:
        await graph_client.connect()
        await graph_client.initialize_schema()

        pipeline = IndexingPipeline(
            graph_client=graph_client,
            embedding_provider=embedding_provider,
            parser_registry=parser_registry,
            settings=task_settings,
        )
        return await pipeline.run(
            source=source,
            name=name,
            exclude_patterns=exclude_patterns,
            reindex=reindex,
            progress_callback=progress_callback,
        )
    finally:
        await graph_client.close()
