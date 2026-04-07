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


@pytest.mark.asyncio
async def test_upsert_relationships_skips_when_endpoint_missing(backend):
    """Edges to non-existent symbols must be silently skipped, not abort the txn."""
    # Mock conn.execute to return "INSERT 0 0" (no row inserted because EXISTS check failed)
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="INSERT 0 0")

    rel = CodeRelationship(
        source_id="repo::missing.source",
        target_id="repo::missing.target",
        relationship_type=RelationshipType.EXTENDS,
    )
    created = await backend._upsert_relationships(conn, [rel])

    # Edge was skipped, but no exception raised
    assert created == 0
    # Verify the SQL contains specific EXISTS guards on both endpoints
    sql = conn.execute.call_args[0][0]
    assert "WHERE EXISTS (SELECT 1 FROM code_symbols WHERE id = $1)" in sql
    assert "AND EXISTS (SELECT 1 FROM code_symbols WHERE id = $2)" in sql


@pytest.mark.asyncio
async def test_upsert_relationships_counts_inserted_when_endpoints_exist(backend):
    """When both endpoints exist, the row is inserted and counted as created."""
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="INSERT 0 1")

    rel = CodeRelationship(
        source_id="repo::A",
        target_id="repo::B",
        relationship_type=RelationshipType.CALLS,
    )
    created = await backend._upsert_relationships(conn, [rel])

    assert created == 1


@pytest.mark.asyncio
async def test_upsert_relationships_does_not_count_double_digit_results(backend):
    """Defensive: 'INSERT 0 11' must NOT be counted as one creation.

    The change from endswith('1') to endswith(' 1') is what enables this distinction.
    """
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="INSERT 0 11")

    rel = CodeRelationship(
        source_id="repo::A",
        target_id="repo::B",
        relationship_type=RelationshipType.CALLS,
    )
    created = await backend._upsert_relationships(conn, [rel])

    # Per-row inserts should never return "INSERT 0 11", but if they did,
    # the endswith(' 1') check correctly rejects it.
    assert created == 0


@pytest.mark.asyncio
async def test_initialize_schema_drops_embeddings_when_dimensions_change(mock_pool):
    """If existing code_embeddings has wrong vector dimension, drop+recreate.

    Verifies the call sequence: _DDL → fetchval(pg_attribute) → DROP → CREATE.
    The dimension check must happen after _DDL runs (which only creates the
    non-embeddings tables) so the table inspection sees the actual production state.
    """
    backend = PostgresWriteBackend(dsn="postgresql://test/test", embedding_dimensions=768)
    backend._pool = mock_pool

    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="OK")
    # Simulate: existing code_embeddings has 1536 dim
    conn.fetchval = AsyncMock(return_value=1536)
    acquire_cm = AsyncMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire = MagicMock(return_value=acquire_cm)

    await backend.initialize_schema()

    # Must have issued a DROP TABLE for code_embeddings
    executed_sql = [call.args[0] for call in conn.execute.call_args_list]
    assert any("DROP TABLE" in sql and "code_embeddings" in sql for sql in executed_sql), \
        f"Expected DROP TABLE code_embeddings; got: {executed_sql}"
    # Must have re-created with vector(768)
    assert any("vector(768)" in sql for sql in executed_sql)
    # Verify ordering: DROP must come BEFORE the CREATE TABLE for code_embeddings
    drop_idx = next(i for i, s in enumerate(executed_sql) if "DROP TABLE" in s and "code_embeddings" in s)
    create_idx = next(i for i, s in enumerate(executed_sql) if "vector(768)" in s)
    assert drop_idx < create_idx, "DROP must precede CREATE in the call sequence"


@pytest.mark.asyncio
async def test_initialize_schema_keeps_embeddings_when_dimensions_match(mock_pool):
    """If existing code_embeddings already has the right dim, do not drop."""
    backend = PostgresWriteBackend(dsn="postgresql://test/test", embedding_dimensions=1536)
    backend._pool = mock_pool

    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="OK")
    conn.fetchval = AsyncMock(return_value=1536)  # already correct
    acquire_cm = AsyncMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire = MagicMock(return_value=acquire_cm)

    await backend.initialize_schema()

    # Must NOT have issued a DROP TABLE
    executed_sql = [call.args[0] for call in conn.execute.call_args_list]
    assert not any("DROP TABLE" in sql for sql in executed_sql)
