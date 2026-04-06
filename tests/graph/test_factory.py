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
        mock_postgres_cls.assert_called_once_with(dsn="postgresql://test@localhost/db")
