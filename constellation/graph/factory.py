"""Factory for creating WriteBackend instances."""

from __future__ import annotations

from constellation.config import Settings
from constellation.graph.base import WriteBackend


def create_write_backend(settings: Settings) -> WriteBackend:
    """Return the configured write backend.

    Reads settings.storage_backend to select the implementation.
    """
    if settings.storage_backend == "postgres":
        from constellation.graph.postgres import PostgresWriteBackend
        return PostgresWriteBackend(
            dsn=settings.postgres_dsn,
            embedding_dimensions=settings.resolved_embedding_dimensions(),
            embedding_model=settings.resolved_embedding_model(),
        )
    elif settings.storage_backend == "neo4j":
        from constellation.graph.neo4j import Neo4jWriteBackend
        return Neo4jWriteBackend(settings)
    else:
        raise ValueError(
            f"Unknown storage backend: {settings.storage_backend!r}. "
            f"Expected 'neo4j' or 'postgres'."
        )
