"""Unit tests for PostgresWriteBackend — mocked asyncpg pool."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from constellation.graph.postgres import PostgresWriteBackend
from constellation.graph.base import WriteBackend
from constellation.models import CodeEntity, CodeRelationship, EntityType, RelationshipType


@pytest.fixture
def mock_pool():
    pool = AsyncMock()
    pool.execute = AsyncMock(return_value="OK")
    pool.fetch = AsyncMock(return_value=[])
    pool.fetchrow = AsyncMock(return_value=None)
    pool.fetchval = AsyncMock(return_value=0)
    pool.acquire = AsyncMock()
    pool.__aenter__ = AsyncMock(return_value=pool)
    pool.__aexit__ = AsyncMock(return_value=False)
    return pool


@pytest.fixture
def backend(mock_pool):
    b = PostgresWriteBackend(dsn="postgresql://test:test@localhost/test")
    b._pool = mock_pool
    return b


def test_implements_write_backend():
    assert issubclass(PostgresWriteBackend, WriteBackend)


@pytest.mark.asyncio
async def test_list_repositories_returns_empty(backend, mock_pool):
    mock_pool.fetch.return_value = []
    result = await backend.list_repositories()
    assert result == []


@pytest.mark.asyncio
async def test_get_repository_returns_none_when_missing(backend, mock_pool):
    mock_pool.fetchrow.return_value = None
    result = await backend.get_repository("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_count_repository_entities_returns_zero(backend, mock_pool):
    mock_pool.fetchval.return_value = 0
    result = await backend.count_repository_entities("test-repo")
    assert result == 0


@pytest.mark.asyncio
async def test_get_file_hashes_returns_empty(backend, mock_pool):
    mock_pool.fetch.return_value = []
    result = await backend.get_file_hashes("test-repo")
    assert result == {}


@pytest.mark.asyncio
async def test_apply_indexing_changes_with_no_data(backend, mock_pool):
    mock_pool.fetchval.return_value = 0
    # Mock the acquire context manager to return a connection mock
    conn_mock = AsyncMock()
    conn_mock.execute = AsyncMock(return_value="INSERT 0 0")
    conn_mock.fetchval = AsyncMock(return_value=0)
    tx_cm = AsyncMock()
    tx_cm.__aenter__ = AsyncMock(return_value=None)
    tx_cm.__aexit__ = AsyncMock(return_value=False)
    conn_mock.transaction = MagicMock(return_value=tx_cm)
    acquire_cm = AsyncMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn_mock)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire = MagicMock(return_value=acquire_cm)

    entities_created, rels_created, total = await backend.apply_indexing_changes(
        repository="test-repo",
        source="/tmp/test",
        commit_sha=None,
        reindex_preparations=[],
        entities=[],
        relationships=[],
        stale_file_paths=[],
    )
    assert entities_created == 0
    assert rels_created == 0
    assert total == 0


@pytest.mark.asyncio
async def test_delete_repository_deletes_symbols_then_repo(backend, mock_pool):
    await backend.delete_repository("my-repo")
    assert mock_pool.execute.call_count == 2
    first_call = mock_pool.execute.call_args_list[0][0][0]
    second_call = mock_pool.execute.call_args_list[1][0][0]
    assert "code_symbols" in first_call
    assert "code_repos" in second_call
