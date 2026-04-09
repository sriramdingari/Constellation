"""Unit tests for PostgresWriteBackend — mocked asyncpg pool."""
from pathlib import Path

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from constellation.graph.postgres import PostgresWriteBackend
from constellation.graph.base import WriteBackend
from constellation.indexer.spool import (
    ChunkPreparation,
    RunManifest,
    SpoolChunkPaths,
    create_spool_dir,
    write_chunk_preparation,
    write_run_manifest,
)
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
async def test_upsert_entities_batches_with_executemany(backend):
    """_upsert_entities must use a single executemany call, not N per-row calls."""
    conn = AsyncMock()
    # Pre-count existing: mock returns 1 existing row (repo::A)
    conn.fetch = AsyncMock(return_value=[{"id": "repo::A"}])
    conn.executemany = AsyncMock(return_value=None)
    conn.fetchval = AsyncMock()
    conn.execute = AsyncMock()

    e1 = CodeEntity(
        id="repo::A", name="A", entity_type=EntityType.CLASS,
        repository="repo", file_path="a.py", line_number=1, language="python",
    )
    e2 = CodeEntity(
        id="repo::B", name="B", entity_type=EntityType.CLASS,
        repository="repo", file_path="b.py", line_number=1, language="python",
    )
    created = await backend._upsert_entities(conn, [e1, e2])

    # Only one new entity (repo::B) — repo::A was in the pre-count result
    assert created == 1

    # Must have called executemany exactly once with both rows
    assert conn.executemany.call_count == 1
    rows_arg = conn.executemany.call_args[0][1]
    assert len(rows_arg) == 2

    # Must NOT have made per-row execute calls for the upsert
    assert conn.execute.call_count == 0


@pytest.mark.asyncio
async def test_upsert_entities_counts_new_entities_via_prefetch(backend):
    """_upsert_entities must compute `created` via pre-fetch, not row xmax."""
    conn = AsyncMock()
    # Pre-fetch returns one existing ID
    conn.fetch = AsyncMock(return_value=[{"id": "repo::A"}])
    conn.executemany = AsyncMock(return_value=None)

    e1 = CodeEntity(
        id="repo::A", name="A", entity_type=EntityType.CLASS,
        repository="repo", file_path="a.py", line_number=1, language="python",
    )
    e2 = CodeEntity(
        id="repo::B", name="B", entity_type=EntityType.CLASS,
        repository="repo", file_path="b.py", line_number=1, language="python",
    )
    created = await backend._upsert_entities(conn, [e1, e2])

    # repo::B is new, repo::A already existed → created == 1
    assert created == 1

    # Verify the pre-fetch queried the correct column/table
    assert conn.fetch.call_count == 1
    fetch_sql = conn.fetch.call_args[0][0]
    assert "SELECT id FROM code_symbols" in fetch_sql
    assert "id = ANY" in fetch_sql


@pytest.mark.asyncio
async def test_upsert_relationships_batches_with_executemany(backend):
    """_upsert_relationships must batch all rows in a single executemany."""
    conn = AsyncMock()
    conn.executemany = AsyncMock(return_value=None)
    # Pre-count returns 2 for both before and after (so created = 0)
    conn.fetchval = AsyncMock(return_value=2)
    conn.fetch = AsyncMock(return_value=[])
    conn.execute = AsyncMock()

    rels = [
        CodeRelationship(
            source_id="repo::A",
            target_id="repo::B",
            relationship_type=RelationshipType.CALLS,
        ),
        CodeRelationship(
            source_id="repo::C",
            target_id="repo::D",
            relationship_type=RelationshipType.EXTENDS,
        ),
    ]
    created = await backend._upsert_relationships(conn, rels)

    # Before=2, after=2 → created=0
    assert created == 0

    assert conn.executemany.call_count == 1
    rows_arg = conn.executemany.call_args[0][1]
    assert len(rows_arg) == 2
    assert conn.execute.call_count == 0


@pytest.mark.asyncio
async def test_upsert_embeddings_batches_with_executemany(backend):
    """_upsert_embeddings must batch all rows in a single executemany."""
    conn = AsyncMock()
    conn.executemany = AsyncMock(return_value=None)
    conn.execute = AsyncMock()

    e1 = CodeEntity(
        id="repo::A", name="A", entity_type=EntityType.METHOD,
        repository="repo", file_path="a.py", line_number=1, language="python",
        embedding=[0.1] * 1536, content_hash="h1",
    )
    e2 = CodeEntity(
        id="repo::B", name="B", entity_type=EntityType.CLASS,
        repository="repo", file_path="b.py", line_number=1, language="python",
        embedding=[0.2] * 1536, content_hash="h2",
    )
    await backend._upsert_embeddings(conn, [e1, e2])

    assert conn.executemany.call_count == 1
    rows_arg = conn.executemany.call_args[0][1]
    assert len(rows_arg) == 2
    assert conn.execute.call_count == 0


@pytest.mark.asyncio
async def test_initialize_schema_drops_embeddings_when_dimensions_change(mock_pool):
    """If existing code_embeddings has wrong vector dimension, drop+recreate.

    Verifies the call sequence: _DDL → fetchval(pg_attribute) → DROP → UPDATE(clear hashes) → CREATE.
    The dimension check must happen after _DDL runs (which only creates the
    non-embeddings tables) so the table inspection sees the actual production state.
    """
    backend = PostgresWriteBackend(
        dsn="postgresql://test/test",
        embedding_dimensions=768,
        embedding_model="text-embedding-3-small",
    )
    backend._pool = mock_pool

    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="OK")
    # fetchval is only called via to_regclass in the migration branch.
    # This test takes the drift branch (fetchrow returns a row), so fetchval
    # is never called for the dimension check — set to None defensively.
    conn.fetchval = AsyncMock(return_value=None)
    # Stored metadata: dim=1536 (drift!), model still matches configured
    conn.fetchrow = AsyncMock(return_value={
        "dimensions": 1536,
        "model": "text-embedding-3-small",
    })
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

    # Verify the UPDATE content_hash clear also fires between DROP and CREATE
    # (from Round 2 Task 1 — dimension drift must trigger re-embed)
    clear_idx = next(
        (i for i, s in enumerate(executed_sql) if "content_hash = NULL" in s),
        None,
    )
    assert clear_idx is not None, \
        f"Expected content_hash clear between DROP and CREATE; got: {executed_sql}"
    assert drop_idx < clear_idx < create_idx, \
        f"Expected DROP → UPDATE → CREATE order; got drop={drop_idx}, clear={clear_idx}, create={create_idx}"


@pytest.mark.asyncio
async def test_orphan_package_cleanup_loops_until_stable(backend, mock_pool):
    """Orphan package cleanup must repeat until no more rows are deleted."""
    # With empty stale_file_paths, reindex_preparations, entities, relationships:
    # the only conn.execute() calls in apply_indexing_changes are:
    #   - The orphan cleanup loop (N passes until DELETE 0)
    #   - The final repo metadata upsert (INSERT 0 1)
    # No fetch/fetchval calls fire either because all input lists are empty.
    conn = AsyncMock()
    conn.execute = AsyncMock(side_effect=[
        # orphan cleanup pass 1: deletes 2 leaf packages
        "DELETE 2",
        # orphan cleanup pass 2: deletes 1 parent now exposed
        "DELETE 1",
        # orphan cleanup pass 3: nothing more, loop terminates
        "DELETE 0",
        # final repository metadata upsert
        "INSERT 0 1",
    ])
    conn.fetchval = AsyncMock(return_value=0)
    tx_cm = AsyncMock()
    tx_cm.__aenter__ = AsyncMock(return_value=None)
    tx_cm.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=tx_cm)
    acquire_cm = AsyncMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire = MagicMock(return_value=acquire_cm)

    await backend.apply_indexing_changes(
        repository="test",
        source="/tmp/test",
        commit_sha=None,
        reindex_preparations=[],
        entities=[],
        relationships=[],
        stale_file_paths=[],
    )

    # Count how many DELETE calls targeted code_symbols WHERE symbol_type = 'Package'
    package_cleanups = [
        call for call in conn.execute.call_args_list
        if "code_symbols" in call.args[0] and "symbol_type = 'Package'" in call.args[0]
    ]
    # Must have looped at least 3 times (2 deletions, 1 deletion, 0 deletions)
    assert len(package_cleanups) >= 3, f"Expected ≥3 cleanup passes, got {len(package_cleanups)}"


@pytest.mark.asyncio
async def test_initialize_schema_keeps_embeddings_when_dimensions_match(mock_pool):
    """If existing code_embeddings already has the right dim, do not drop."""
    backend = PostgresWriteBackend(
        dsn="postgresql://test/test",
        embedding_dimensions=1536,
        embedding_model="text-embedding-3-small",
    )
    backend._pool = mock_pool

    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="OK")
    # fetchval is only called via to_regclass in the migration branch.
    # This test takes the no-drift branch (fetchrow returns matching metadata),
    # so fetchval is never called — set to None defensively.
    conn.fetchval = AsyncMock(return_value=None)
    # Stored metadata exactly matches configured
    conn.fetchrow = AsyncMock(return_value={
        "dimensions": 1536,
        "model": "text-embedding-3-small",
    })
    acquire_cm = AsyncMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire = MagicMock(return_value=acquire_cm)

    await backend.initialize_schema()

    # Must NOT have issued a DROP TABLE
    executed_sql = [call.args[0] for call in conn.execute.call_args_list]
    assert not any("DROP TABLE" in sql for sql in executed_sql)


@pytest.mark.asyncio
async def test_initialize_schema_clears_file_hashes_on_dimension_drift(mock_pool):
    """When dimensions drift, all File.content_hash values must be cleared
    so the next pipeline run treats every file as changed and re-embeds them.
    Otherwise unchanged files keep their stored hash, skip reprocessing, and
    silently lose their embeddings.
    """
    backend = PostgresWriteBackend(
        dsn="postgresql://test/test",
        embedding_dimensions=768,
        embedding_model="text-embedding-3-small",
    )
    backend._pool = mock_pool

    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="OK")
    # fetchval is only called via to_regclass in the migration branch.
    # This test takes the drift branch, so fetchval is never called —
    # set to None defensively.
    conn.fetchval = AsyncMock(return_value=None)
    # Stored metadata: old dim, model still matches configured
    conn.fetchrow = AsyncMock(return_value={
        "dimensions": 1536,
        "model": "text-embedding-3-small",
    })
    acquire_cm = AsyncMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire = MagicMock(return_value=acquire_cm)

    await backend.initialize_schema()

    executed_sql = [call.args[0] for call in conn.execute.call_args_list]
    # Must have issued UPDATE code_symbols SET content_hash = NULL for File entities
    clear_hash_stmts = [
        s for s in executed_sql
        if "UPDATE code_symbols" in s
        and "content_hash = NULL" in s
        and "symbol_type = 'File'" in s
    ]
    assert len(clear_hash_stmts) == 1, \
        f"Expected exactly one UPDATE to clear File content_hash; got: {executed_sql}"
    # Verify the ordering: DROP → UPDATE → CREATE
    drop_idx = next(i for i, s in enumerate(executed_sql) if "DROP TABLE" in s and "code_embeddings" in s)
    clear_idx = next(i for i, s in enumerate(executed_sql) if "content_hash = NULL" in s)
    create_idx = next(i for i, s in enumerate(executed_sql) if "vector(768)" in s)
    assert drop_idx < clear_idx < create_idx, \
        f"Expected DROP → UPDATE → CREATE order; indices: drop={drop_idx}, clear={clear_idx}, create={create_idx}"


@pytest.mark.asyncio
async def test_initialize_schema_does_not_clear_hashes_when_dimensions_match(mock_pool):
    """When dimensions already match, do NOT clear File content_hash."""
    backend = PostgresWriteBackend(
        dsn="postgresql://test/test",
        embedding_dimensions=1536,
        embedding_model="text-embedding-3-small",
    )
    backend._pool = mock_pool

    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="OK")
    # fetchval is only called via to_regclass in the migration branch.
    # This test takes the no-drift branch (fetchrow returns matching metadata),
    # so fetchval is never called — set to None defensively.
    conn.fetchval = AsyncMock(return_value=None)
    # Stored metadata exactly matches configured
    conn.fetchrow = AsyncMock(return_value={
        "dimensions": 1536,
        "model": "text-embedding-3-small",
    })
    acquire_cm = AsyncMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire = MagicMock(return_value=acquire_cm)

    await backend.initialize_schema()

    executed_sql = [call.args[0] for call in conn.execute.call_args_list]
    # Must NOT have issued the content_hash clear
    assert not any("content_hash = NULL" in s for s in executed_sql), \
        f"Expected no UPDATE content_hash = NULL when dimensions match; got: {executed_sql}"


@pytest.mark.asyncio
async def test_initialize_schema_clears_embeddings_on_model_drift(mock_pool):
    """When embedding_model changes but dimensions stay the same, drop
    code_embeddings and clear File content hashes so the next indexing
    run regenerates embeddings with the new model.
    """
    backend = PostgresWriteBackend(
        dsn="postgresql://test/test",
        embedding_dimensions=1536,
        embedding_model="text-embedding-3-small",
    )
    backend._pool = mock_pool

    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="OK")
    conn.fetchval = AsyncMock(return_value=1536)  # existing dim unchanged
    # fetchrow returns the existing metadata row: old model, same dim
    conn.fetchrow = AsyncMock(return_value={
        "dimensions": 1536,
        "model": "text-embedding-ada-002",  # DIFFERENT model
    })
    acquire_cm = AsyncMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire = MagicMock(return_value=acquire_cm)

    await backend.initialize_schema()

    executed_sql = [call.args[0] for call in conn.execute.call_args_list]
    # Must have issued DROP TABLE code_embeddings
    assert any(
        "DROP TABLE" in sql and "code_embeddings" in sql for sql in executed_sql
    ), f"Expected DROP TABLE on model drift; got: {executed_sql}"
    # Must have cleared File content hashes
    assert any(
        "UPDATE code_symbols" in sql
        and "content_hash = NULL" in sql
        and "symbol_type = 'File'" in sql
        for sql in executed_sql
    ), f"Expected content_hash clear on model drift; got: {executed_sql}"
    # Must have updated embedding_metadata with new model
    assert any(
        "embedding_metadata" in sql and "UPDATE" in sql
        for sql in executed_sql
    ), f"Expected embedding_metadata UPDATE; got: {executed_sql}"


@pytest.mark.asyncio
async def test_initialize_schema_drops_embeddings_when_both_dim_and_model_drift(mock_pool):
    """When BOTH dimensions AND model change simultaneously, still trigger
    the drift branch. Protects against a future bug where `or` gets changed
    to `and` in the drift condition."""
    backend = PostgresWriteBackend(
        dsn="postgresql://test/test",
        embedding_dimensions=768,
        embedding_model="nomic-embed-text",
    )
    backend._pool = mock_pool

    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="OK")
    conn.fetchval = AsyncMock(return_value=None)
    # Stored metadata differs on BOTH fields
    conn.fetchrow = AsyncMock(return_value={
        "dimensions": 1536,  # was OpenAI 1536
        "model": "text-embedding-3-small",  # different model
    })
    acquire_cm = AsyncMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire = MagicMock(return_value=acquire_cm)

    await backend.initialize_schema()

    executed_sql = [call.args[0] for call in conn.execute.call_args_list]
    # Must have dropped code_embeddings
    assert any(
        "DROP TABLE" in sql and "code_embeddings" in sql for sql in executed_sql
    ), f"Expected DROP on combined dim+model drift; got: {executed_sql}"
    # Must have cleared File content hashes
    assert any(
        "content_hash = NULL" in sql for sql in executed_sql
    ), f"Expected content_hash clear; got: {executed_sql}"
    # Must have updated metadata with new dim+model
    assert any(
        "embedding_metadata" in sql and "UPDATE" in sql for sql in executed_sql
    ), f"Expected embedding_metadata UPDATE; got: {executed_sql}"
    # Must have re-created code_embeddings with the new dimension (768)
    assert any("vector(768)" in sql for sql in executed_sql), \
        f"Expected vector(768) recreation; got: {executed_sql}"


@pytest.mark.asyncio
async def test_initialize_schema_no_drop_when_model_and_dim_match(mock_pool):
    """When both dim AND model match existing metadata, do NOT drop
    code_embeddings and do NOT clear File content hashes."""
    backend = PostgresWriteBackend(
        dsn="postgresql://test/test",
        embedding_dimensions=1536,
        embedding_model="text-embedding-3-small",
    )
    backend._pool = mock_pool

    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="OK")
    conn.fetchval = AsyncMock(return_value=1536)
    conn.fetchrow = AsyncMock(return_value={
        "dimensions": 1536,
        "model": "text-embedding-3-small",  # matches configured
    })
    acquire_cm = AsyncMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire = MagicMock(return_value=acquire_cm)

    await backend.initialize_schema()

    executed_sql = [call.args[0] for call in conn.execute.call_args_list]
    assert not any("DROP TABLE" in sql for sql in executed_sql), \
        f"Expected no DROP when model+dim match; got: {executed_sql}"
    assert not any("content_hash = NULL" in sql for sql in executed_sql), \
        f"Expected no content_hash clear when model+dim match; got: {executed_sql}"


@pytest.mark.asyncio
async def test_initialize_schema_inserts_metadata_on_fresh_install(mock_pool):
    """On fresh install (no embedding_metadata row, no code_embeddings),
    insert the configured dim+model values. No drop/clear should fire."""
    backend = PostgresWriteBackend(
        dsn="postgresql://test/test",
        embedding_dimensions=1536,
        embedding_model="text-embedding-3-small",
    )
    backend._pool = mock_pool

    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="OK")
    # fetchrow returns None (no metadata row yet)
    conn.fetchrow = AsyncMock(return_value=None)
    # fetchval returns None for to_regclass check (code_embeddings doesn't exist)
    conn.fetchval = AsyncMock(return_value=None)
    acquire_cm = AsyncMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire = MagicMock(return_value=acquire_cm)

    await backend.initialize_schema()

    executed_sql = [call.args[0] for call in conn.execute.call_args_list]
    # Must have inserted into embedding_metadata
    assert any(
        "embedding_metadata" in sql and "INSERT" in sql for sql in executed_sql
    ), f"Expected INSERT into embedding_metadata on fresh install; got: {executed_sql}"
    # Must NOT have dropped code_embeddings (nothing to drop on fresh install)
    assert not any(
        "DROP TABLE" in sql and "code_embeddings" in sql for sql in executed_sql
    ), f"Expected no DROP on fresh install; got: {executed_sql}"


@pytest.mark.asyncio
async def test_initialize_schema_migration_clears_existing_embeddings_without_metadata(mock_pool):
    """Migration boundary: if embedding_metadata is empty but code_embeddings
    already exists (old deployment), treat as drift and clear the corpus so
    the next indexing run regenerates embeddings with the configured model."""
    backend = PostgresWriteBackend(
        dsn="postgresql://test/test",
        embedding_dimensions=1536,
        embedding_model="text-embedding-3-small",
    )
    backend._pool = mock_pool

    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="OK")
    # fetchrow returns None (metadata empty) — but code_embeddings exists
    conn.fetchrow = AsyncMock(return_value=None)
    # fetchval returns a truthy value for the to_regclass check on code_embeddings
    # This simulates an existing deployment with populated embeddings
    conn.fetchval = AsyncMock(return_value="public.code_embeddings")
    acquire_cm = AsyncMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire = MagicMock(return_value=acquire_cm)

    await backend.initialize_schema()

    executed_sql = [call.args[0] for call in conn.execute.call_args_list]
    # Must have dropped code_embeddings (migration drift)
    assert any(
        "DROP TABLE" in sql and "code_embeddings" in sql for sql in executed_sql
    ), f"Expected DROP on migration; got: {executed_sql}"
    # Must have cleared File content hashes
    assert any(
        "content_hash = NULL" in sql for sql in executed_sql
    ), f"Expected content_hash clear on migration; got: {executed_sql}"
    # Must have inserted baseline metadata
    assert any(
        "embedding_metadata" in sql and "INSERT" in sql for sql in executed_sql
    ), f"Expected INSERT into embedding_metadata on migration; got: {executed_sql}"


@pytest.mark.asyncio
async def test_upsert_entities_deduplicates_input_by_id(backend):
    """Duplicate entity IDs in the input must not overcount `created`."""
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])  # nothing exists yet
    conn.executemany = AsyncMock(return_value=None)

    e1 = CodeEntity(
        id="repo::A", name="A", entity_type=EntityType.CLASS,
        repository="repo", file_path="a.py", line_number=1, language="python",
    )
    # Same ID appears twice
    e1_dup = CodeEntity(
        id="repo::A", name="A", entity_type=EntityType.CLASS,
        repository="repo", file_path="a.py", line_number=2, language="python",
    )
    created = await backend._upsert_entities(conn, [e1, e1_dup])

    # Only one unique ID, so created must be 1 (not 2)
    assert created == 1

    # executemany should receive exactly 1 row (last-writer-wins dedup)
    rows_arg = conn.executemany.call_args[0][1]
    assert len(rows_arg) == 1


@pytest.mark.asyncio
async def test_apply_spooled_indexing_changes_replays_multiple_chunks(
    backend, mock_pool, tmp_path: Path,
):
    """apply_spooled_indexing_changes reads a RunManifest + per-chunk spool
    files and replays all changes inside a single Postgres transaction."""
    conn = AsyncMock()
    # Call sequence (no stale files, 2 chunks each with 1 FILE entity, 0 rels):
    #   Per chunk (x2):
    #     conn.fetch  -> [] (reindex prep: existing ids for file)
    #     conn.fetch  -> [] (_upsert_entities: pre-fetch existing)
    #     conn.executemany   (_upsert_entities: batch upsert)
    #   After all chunks:
    #     conn.execute -> "DELETE 0"   (package cleanup, terminates loop)
    #     conn.fetchval -> 4           (final entity count)
    #     conn.execute -> "INSERT 0 1" (repo metadata upsert)
    conn.fetch = AsyncMock(return_value=[])
    conn.executemany = AsyncMock(return_value=None)
    conn.execute = AsyncMock(side_effect=[
        "DELETE 0",      # package cleanup loop — first pass, terminates
        "INSERT 0 1",   # repo metadata upsert
    ])
    conn.fetchval = AsyncMock(return_value=4)
    tx_cm = AsyncMock()
    tx_cm.__aenter__ = AsyncMock(return_value=None)
    tx_cm.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=tx_cm)
    acquire_cm = AsyncMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire = MagicMock(return_value=acquire_cm)

    # --- Set up spool directory with manifest + 2 chunks ---
    spool_dir = create_spool_dir(tmp_path, "run-1")
    manifest = RunManifest(
        run_id="run-1",
        repository="repo",
        source="/tmp/repo",
        commit_sha="abc123",
        stale_file_paths=[],
        chunk_indices=[1, 2],
        files_total=2,
        files_processed=2,
        files_skipped=0,
    )
    write_run_manifest(spool_dir / "run_manifest.json", manifest)

    chunk1 = ChunkPreparation(
        chunk_index=1,
        files=[("a.py", "ha", True)],
        reindex_preparations=[("a.py", {"repo::a.py"})],
        entities=[
            CodeEntity(
                id="repo::a.py",
                name="a.py",
                entity_type=EntityType.FILE,
                repository="repo",
                file_path="a.py",
                line_number=1,
                language="python",
                content_hash="ha",
            )
        ],
        relationships=[],
    )
    chunk2 = ChunkPreparation(
        chunk_index=2,
        files=[("b.py", "hb", True)],
        reindex_preparations=[("b.py", {"repo::b.py"})],
        entities=[
            CodeEntity(
                id="repo::b.py",
                name="b.py",
                entity_type=EntityType.FILE,
                repository="repo",
                file_path="b.py",
                line_number=1,
                language="python",
                content_hash="hb",
            )
        ],
        relationships=[],
    )
    write_chunk_preparation(SpoolChunkPaths.for_chunk(spool_dir, 1), chunk1)
    write_chunk_preparation(SpoolChunkPaths.for_chunk(spool_dir, 2), chunk2)

    entities_created, rels_created, total = await backend.apply_spooled_indexing_changes(
        spool_dir=spool_dir,
    )

    # Each chunk has 1 new entity, _upsert_entities returns 1 per chunk -> 2 total
    assert entities_created == 2
    assert rels_created == 0
    # fetchval returns 4 as the final entity count
    assert total == 4
    # executemany called once per chunk (entity upsert), 2 chunks total
    assert conn.executemany.call_count >= 2


@pytest.mark.asyncio
async def test_apply_spooled_indexing_changes_rolls_back_on_chunk_failure(
    backend, mock_pool, tmp_path,
):
    """When replay raises inside the transaction, asyncpg rolls back."""
    conn = AsyncMock()
    conn.execute = AsyncMock(side_effect=RuntimeError("boom"))
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchval = AsyncMock(return_value=0)
    conn.executemany = AsyncMock(return_value=None)
    tx_cm = AsyncMock()
    tx_cm.__aenter__ = AsyncMock(return_value=None)
    tx_cm.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=tx_cm)
    acquire_cm = AsyncMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire = MagicMock(return_value=acquire_cm)

    spool_dir = create_spool_dir(tmp_path, "run-rollback")
    write_run_manifest(
        spool_dir / "run_manifest.json",
        RunManifest(
            run_id="run-rollback",
            repository="repo",
            source="/tmp/repo",
            commit_sha=None,
            stale_file_paths=["old.py"],
            chunk_indices=[],
            files_total=0,
            files_processed=0,
            files_skipped=0,
        ),
    )

    with pytest.raises(RuntimeError, match="boom"):
        await backend.apply_spooled_indexing_changes(spool_dir=spool_dir)

    conn.transaction.assert_called_once()


@pytest.mark.asyncio
async def test_apply_spooled_cross_chunk_relationships_are_preserved(
    backend, mock_pool, tmp_path: Path,
):
    """Cross-chunk relationships must survive the replay.

    Chunk 1 declares entities A.py + Method Foo AND a CALLS relationship
    from Foo -> B.py#Bar.  Chunk 2 introduces entities B.py + Method Bar.

    Before the fix, relationships were upserted per-chunk inside the loop,
    so chunk 1's CALLS edge was attempted before chunk 2's entity existed —
    the WHERE EXISTS guard silently dropped it.

    After the fix (two-pass replay), all entities are upserted first, then
    all relationships are upserted in a second pass.
    """
    conn = AsyncMock()
    # Call sequence for two-pass replay:
    #   Pass 1 (per chunk x2):
    #     conn.fetch  -> [] (reindex prep: existing ids for file)
    #     conn.fetch  -> [] (_upsert_entities: pre-fetch existing)
    #     conn.executemany   (_upsert_entities: batch upsert)
    #     conn.executemany   (_upsert_embeddings: no embeddable => not called)
    #   Pass 2 (all relationships):
    #     conn.fetchval -> 0 (_upsert_relationships: before_count)
    #     conn.executemany   (_upsert_relationships: batch upsert)
    #     conn.fetchval -> 1 (_upsert_relationships: after_count)
    #   After all chunks:
    #     conn.execute -> "DELETE 0"   (package cleanup, terminates loop)
    #     conn.fetchval -> 6           (final entity count)
    #     conn.execute -> "INSERT 0 1" (repo metadata upsert)
    conn.fetch = AsyncMock(return_value=[])
    conn.executemany = AsyncMock(return_value=None)
    conn.execute = AsyncMock(side_effect=[
        "DELETE 0",      # package cleanup loop — first pass, terminates
        "INSERT 0 1",   # repo metadata upsert
    ])
    # fetchval call sequence:
    #   1. _upsert_relationships before_count -> 0
    #   2. _upsert_relationships after_count  -> 1 (one new edge)
    #   3. final entity count                 -> 6
    conn.fetchval = AsyncMock(side_effect=[0, 1, 6])
    tx_cm = AsyncMock()
    tx_cm.__aenter__ = AsyncMock(return_value=None)
    tx_cm.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=tx_cm)
    acquire_cm = AsyncMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire = MagicMock(return_value=acquire_cm)

    # --- Set up spool directory with manifest + 2 chunks ---
    spool_dir = create_spool_dir(tmp_path, "run-cross")
    manifest = RunManifest(
        run_id="run-cross",
        repository="cross-repo",
        source="/tmp/cross-repo",
        commit_sha="def456",
        stale_file_paths=[],
        chunk_indices=[1, 2],
        files_total=2,
        files_processed=2,
        files_skipped=0,
    )
    write_run_manifest(spool_dir / "run_manifest.json", manifest)

    # Chunk 1: entities A.py + Method Foo, plus a CALLS edge Foo -> Bar
    # (Bar doesn't exist yet — it's in chunk 2)
    chunk1 = ChunkPreparation(
        chunk_index=1,
        files=[("a.py", "ha", True)],
        reindex_preparations=[("a.py", {"cross-repo::a.py", "cross-repo::a.py::Foo"})],
        entities=[
            CodeEntity(
                id="cross-repo::a.py",
                name="a.py",
                entity_type=EntityType.FILE,
                repository="cross-repo",
                file_path="a.py",
                line_number=1,
                language="python",
                content_hash="ha",
            ),
            CodeEntity(
                id="cross-repo::a.py::Foo",
                name="Foo",
                entity_type=EntityType.METHOD,
                repository="cross-repo",
                file_path="a.py",
                line_number=5,
                language="python",
            ),
        ],
        relationships=[
            CodeRelationship(
                source_id="cross-repo::a.py::Foo",
                target_id="cross-repo::b.py::Bar",
                relationship_type=RelationshipType.CALLS,
            ),
        ],
    )

    # Chunk 2: entities B.py + Method Bar (the target of the cross-chunk edge)
    chunk2 = ChunkPreparation(
        chunk_index=2,
        files=[("b.py", "hb", True)],
        reindex_preparations=[("b.py", {"cross-repo::b.py", "cross-repo::b.py::Bar"})],
        entities=[
            CodeEntity(
                id="cross-repo::b.py",
                name="b.py",
                entity_type=EntityType.FILE,
                repository="cross-repo",
                file_path="b.py",
                line_number=1,
                language="python",
                content_hash="hb",
            ),
            CodeEntity(
                id="cross-repo::b.py::Bar",
                name="Bar",
                entity_type=EntityType.METHOD,
                repository="cross-repo",
                file_path="b.py",
                line_number=5,
                language="python",
            ),
        ],
        relationships=[],
    )

    write_chunk_preparation(SpoolChunkPaths.for_chunk(spool_dir, 1), chunk1)
    write_chunk_preparation(SpoolChunkPaths.for_chunk(spool_dir, 2), chunk2)

    entities_created, rels_created, total = await backend.apply_spooled_indexing_changes(
        spool_dir=spool_dir,
    )

    # 4 new entities (2 per chunk), 1 cross-chunk CALLS edge
    assert entities_created == 4
    assert rels_created == 1, (
        "Cross-chunk CALLS edge (Foo -> Bar) must survive; "
        "before the two-pass fix it was silently dropped"
    )
    assert total == 6
