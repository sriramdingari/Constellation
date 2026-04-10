"""Tests for create_write_backend factory."""
import pytest
from unittest.mock import patch

from constellation.config import Settings
from constellation.graph.factory import create_write_backend
from constellation.graph.neo4j import Neo4jWriteBackend
from constellation.graph.base import WriteBackend


def test_factory_returns_neo4j_by_default():
    settings = Settings()
    backend = create_write_backend(settings)
    assert isinstance(backend, Neo4jWriteBackend)
    assert isinstance(backend, WriteBackend)


def test_factory_returns_neo4j_explicitly():
    settings = Settings(storage_backend="neo4j")
    backend = create_write_backend(settings)
    assert isinstance(backend, Neo4jWriteBackend)


def test_factory_raises_on_unknown_backend():
    settings = Settings()
    object.__setattr__(settings, "storage_backend", "sqlite")
    with pytest.raises(ValueError, match="Unknown storage backend"):
        create_write_backend(settings)


def test_factory_returns_postgres_when_configured():
    from unittest.mock import MagicMock, patch
    mock_postgres_cls = MagicMock()
    with patch.dict("sys.modules", {"constellation.graph.postgres": MagicMock(PostgresWriteBackend=mock_postgres_cls)}):
        settings = Settings(storage_backend="postgres", postgres_dsn="postgresql://test@localhost/db")
        backend = create_write_backend(settings)
        mock_postgres_cls.assert_called_once_with(
            dsn="postgresql://test@localhost/db",
            embedding_dimensions=settings.resolved_embedding_dimensions(),
            embedding_model=settings.resolved_embedding_model(),
        )


def test_factory_passes_embedding_model_to_postgres_backend():
    """The factory must pass the configured embedding_model to PostgresWriteBackend,
    not just embedding_dimensions."""
    from unittest.mock import MagicMock, patch
    mock_postgres_cls = MagicMock()
    with patch.dict(
        "sys.modules",
        {"constellation.graph.postgres": MagicMock(PostgresWriteBackend=mock_postgres_cls)},
    ):
        settings = Settings(
            storage_backend="postgres",
            postgres_dsn="postgresql://test@localhost/db",
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
        )
        create_write_backend(settings)
        kwargs = mock_postgres_cls.call_args.kwargs
        assert kwargs["embedding_model"] == "text-embedding-3-small"
        assert kwargs["embedding_dimensions"] == 1536


def test_write_backend_has_apply_spooled_indexing_changes():
    """WriteBackend must declare apply_spooled_indexing_changes
    so the interface contract is honest."""
    assert hasattr(WriteBackend, "apply_spooled_indexing_changes")


@pytest.mark.asyncio
async def test_write_backend_default_spooled_raises():
    """Default implementation raises NotImplementedError for backends
    that don't support spooled indexing."""
    from pathlib import Path

    class DummyBackend(WriteBackend):
        async def connect(self): pass
        async def close(self): pass
        async def initialize_schema(self): pass
        async def upsert_repository(self, **kw): pass
        async def get_repository(self, name): pass
        async def list_repositories(self): pass
        async def delete_repository(self, name): pass
        async def get_file_hashes(self, repo): return {}
        async def apply_indexing_changes(self, **kw): return (0, 0, 0)
        async def count_repository_entities(self, repo): return 0

    backend = DummyBackend()
    with pytest.raises(NotImplementedError, match="DummyBackend"):
        await backend.apply_spooled_indexing_changes(spool_dir=Path("/tmp/fake"))
