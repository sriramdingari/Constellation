"""
Integration tests for PostgresWriteBackend against a real pgvector database.

Uses testcontainers to auto-manage a pgvector/pgvector:pg16 container for the
entire test session. Gracefully skips the whole module if Docker is unavailable.
"""
import os
import pytest
import pytest_asyncio

from constellation.config import Settings
from constellation.graph.postgres import PostgresWriteBackend
from constellation.models import CodeEntity, CodeRelationship, EntityType, RelationshipType

# Try to import testcontainers; if Docker or the library is unavailable, skip
# the entire module rather than erroring at collection time.
_skip_reason = None
try:
    from testcontainers.postgres import PostgresContainer
except ImportError as e:
    _skip_reason = f"testcontainers not installed: {e}"

pytestmark = [pytest.mark.postgres_integration]
if _skip_reason:
    pytestmark.append(pytest.mark.skip(reason=_skip_reason))


@pytest.fixture(scope="session")
def postgres_container():
    """Session-scoped pgvector container. Starts once, tears down at session end."""
    try:
        container = PostgresContainer(
            image="pgvector/pgvector:pg16",
            username="constellation",
            password="secret",
            dbname="constellation",
        )
        container.start()
    except Exception as exc:
        pytest.skip(f"Docker unavailable for postgres integration tests: {exc}")
    yield container
    # Note: if container.start() raised above, pytest.skip() raises Skipped
    # (a BaseException, not an Exception), which propagates past the yield
    # and this line is never reached. Safe to call unconditionally on the
    # happy path.
    container.stop()


@pytest.fixture
def postgres_dsn(postgres_container):
    """asyncpg-compatible DSN for the running container."""
    # testcontainers returns a psycopg2 URL; normalize to asyncpg's form.
    raw = postgres_container.get_connection_url()
    return raw.replace("postgresql+psycopg2://", "postgresql://")


@pytest_asyncio.fixture
async def write_backend(postgres_dsn):
    """Fresh PostgresWriteBackend connected to the session-scoped container.

    Initializes the schema and cleans up any leftover test repositories at
    the start of each test.
    """
    backend = PostgresWriteBackend(
        dsn=postgres_dsn,
        embedding_dimensions=1536,
        embedding_model="text-embedding-3-small",
    )
    await backend.connect()
    await backend.initialize_schema()
    # Defensive: clean up any repos from a prior failed test
    for name in ["test-repo", "test-repo-delete", "stable-reindex-test",
                 "orphan-reindex-test", "stale-rel-test", "file-hashes-test"]:
        try:
            await backend.delete_repository(name)
        except Exception:
            pass
    yield backend
    await backend.close()


async def test_initialize_schema_is_idempotent(write_backend):
    """Calling initialize_schema twice should not error."""
    await write_backend.initialize_schema()
    await write_backend.initialize_schema()


async def test_upsert_repository_creates_row(write_backend):
    await write_backend.upsert_repository(
        name="test-repo",
        source="/tmp/test-repo",
        commit_sha="abc123",
        entity_count=0,
    )
    repo = await write_backend.get_repository("test-repo")
    assert repo is not None
    assert repo["name"] == "test-repo"
    assert repo["source"] == "/tmp/test-repo"


async def test_list_repositories_includes_test_repo(write_backend):
    await write_backend.upsert_repository(
        name="test-repo",
        source="/tmp/test-repo",
        commit_sha="abc123",
        entity_count=0,
    )
    repos = await write_backend.list_repositories()
    names = [r["name"] for r in repos]
    assert "test-repo" in names


async def test_apply_indexing_changes_writes_entities(write_backend):
    """Verify entities and relationships are persisted via apply_indexing_changes."""
    # First clean any existing data
    await write_backend.delete_repository("test-repo")

    file_entity = CodeEntity(
        id="test-repo::src/foo.py",
        name="foo.py",
        entity_type=EntityType.FILE,
        repository="test-repo",
        file_path="src/foo.py",
        line_number=1,
        language="python",
        content_hash="hash1",
    )
    class_entity = CodeEntity(
        id="test-repo::src/foo.py::MyClass",
        name="MyClass",
        entity_type=EntityType.CLASS,
        repository="test-repo",
        file_path="src/foo.py",
        line_number=1,
        language="python",
        signature="class MyClass:",
    )
    method_entity = CodeEntity(
        id="test-repo::src/foo.py::MyClass.do_thing",
        name="do_thing",
        entity_type=EntityType.METHOD,
        repository="test-repo",
        file_path="src/foo.py",
        line_number=2,
        language="python",
        signature="def do_thing(self):",
    )

    contains_rel = CodeRelationship(
        source_id="test-repo::src/foo.py",
        target_id="test-repo::src/foo.py::MyClass",
        relationship_type=RelationshipType.CONTAINS,
    )
    has_method_rel = CodeRelationship(
        source_id="test-repo::src/foo.py::MyClass",
        target_id="test-repo::src/foo.py::MyClass.do_thing",
        relationship_type=RelationshipType.HAS_METHOD,
    )

    entities_created, rels_created, total = await write_backend.apply_indexing_changes(
        repository="test-repo",
        source="/tmp/test-repo",
        commit_sha="abc123",
        reindex_preparations=[],
        entities=[file_entity, class_entity, method_entity],
        relationships=[contains_rel, has_method_rel],
        stale_file_paths=[],
    )

    assert total >= 3, f"Expected at least 3 entities, got {total}"

    # Verify the entities are queryable
    count = await write_backend.count_repository_entities("test-repo")
    assert count >= 3


async def test_get_file_hashes_returns_indexed_files(write_backend):
    """get_file_hashes returns {file_path: content_hash} for all File entities."""
    await write_backend.delete_repository("file-hashes-test")

    file_a = CodeEntity(
        id="file-hashes-test::src/a.py",
        name="a.py",
        entity_type=EntityType.FILE,
        repository="file-hashes-test",
        file_path="src/a.py",
        line_number=1,
        language="python",
        content_hash="hash-a",
    )
    file_b = CodeEntity(
        id="file-hashes-test::src/b.py",
        name="b.py",
        entity_type=EntityType.FILE,
        repository="file-hashes-test",
        file_path="src/b.py",
        line_number=1,
        language="python",
        content_hash="hash-b",
    )
    await write_backend.apply_indexing_changes(
        repository="file-hashes-test",
        source="/tmp/test",
        commit_sha=None,
        reindex_preparations=[
            ("src/a.py", {file_a.id}),
            ("src/b.py", {file_b.id}),
        ],
        entities=[file_a, file_b],
        relationships=[],
        stale_file_paths=[],
    )

    hashes = await write_backend.get_file_hashes("file-hashes-test")

    assert hashes == {
        "src/a.py": "hash-a",
        "src/b.py": "hash-b",
    }

    await write_backend.delete_repository("file-hashes-test")


async def test_declaration_stable_reindex_reports_zero_new_entities(write_backend):
    """A reindex with no new entities must report entities_created == 0,
    matching Neo4j's MERGE semantics. Mirrors test_integration.py:209."""
    # Clean slate
    await write_backend.delete_repository("stable-reindex-test")

    file_entity = CodeEntity(
        id="stable-reindex-test::svc.py",
        name="svc.py",
        entity_type=EntityType.FILE,
        repository="stable-reindex-test",
        file_path="svc.py",
        line_number=1,
        language="python",
        content_hash="hash-v1",
    )
    class_entity = CodeEntity(
        id="stable-reindex-test::svc.py::UserService",
        name="UserService",
        entity_type=EntityType.CLASS,
        repository="stable-reindex-test",
        file_path="svc.py",
        line_number=1,
        language="python",
    )

    # First indexing: 2 entities created
    e1, _, _ = await write_backend.apply_indexing_changes(
        repository="stable-reindex-test",
        source="/tmp/test",
        commit_sha=None,
        reindex_preparations=[("svc.py", {"stable-reindex-test::svc.py", "stable-reindex-test::svc.py::UserService"})],
        entities=[file_entity, class_entity],
        relationships=[],
        stale_file_paths=[],
    )
    assert e1 == 2, f"First index: expected 2 created, got {e1}"

    # Second indexing with EXACT same entities: must be 0 created
    e2, _, _ = await write_backend.apply_indexing_changes(
        repository="stable-reindex-test",
        source="/tmp/test",
        commit_sha=None,
        reindex_preparations=[("svc.py", {"stable-reindex-test::svc.py", "stable-reindex-test::svc.py::UserService"})],
        entities=[file_entity, class_entity],
        relationships=[],
        stale_file_paths=[],
    )
    assert e2 == 0, f"Stable reindex: expected 0 created, got {e2}"

    await write_backend.delete_repository("stable-reindex-test")


async def test_reindex_removes_stale_entities_within_file(write_backend):
    """When a file is reindexed and an entity disappears, it must be deleted."""
    await write_backend.delete_repository("orphan-reindex-test")

    file_entity = CodeEntity(
        id="orphan-reindex-test::svc.py",
        name="svc.py", entity_type=EntityType.FILE,
        repository="orphan-reindex-test", file_path="svc.py",
        line_number=1, language="python", content_hash="h1",
    )
    method_a = CodeEntity(
        id="orphan-reindex-test::svc.py::method_a",
        name="method_a", entity_type=EntityType.METHOD,
        repository="orphan-reindex-test", file_path="svc.py",
        line_number=1, language="python",
    )
    method_b = CodeEntity(
        id="orphan-reindex-test::svc.py::method_b",
        name="method_b", entity_type=EntityType.METHOD,
        repository="orphan-reindex-test", file_path="svc.py",
        line_number=2, language="python",
    )

    # Initial: file + 2 methods
    await write_backend.apply_indexing_changes(
        repository="orphan-reindex-test",
        source="/tmp", commit_sha=None,
        reindex_preparations=[("svc.py", {file_entity.id, method_a.id, method_b.id})],
        entities=[file_entity, method_a, method_b],
        relationships=[],
        stale_file_paths=[],
    )
    assert await write_backend.count_repository_entities("orphan-reindex-test") == 3

    # Reindex: method_b is gone (only method_a remains)
    await write_backend.apply_indexing_changes(
        repository="orphan-reindex-test",
        source="/tmp", commit_sha=None,
        reindex_preparations=[("svc.py", {file_entity.id, method_a.id})],
        entities=[file_entity, method_a],
        relationships=[],
        stale_file_paths=[],
    )
    # method_b must be gone — count is now 2
    count = await write_backend.count_repository_entities("orphan-reindex-test")
    assert count == 2, f"Expected 2 entities after stale removal, got {count}"

    await write_backend.delete_repository("orphan-reindex-test")


async def test_reindex_removes_stale_relationships(write_backend):
    """When a relationship disappears in a refactor, the old code_references row
    must be deleted. Otherwise stale EXTENDS edges (and friends) survive forever
    because the new entity-preserving snapshot+diff logic doesn't cascade them."""
    await write_backend.delete_repository("stale-rel-test")

    file_entity = CodeEntity(
        id="stale-rel-test::svc.py",
        name="svc.py", entity_type=EntityType.FILE,
        repository="stale-rel-test", file_path="svc.py",
        line_number=1, language="python", content_hash="h1",
    )
    base_class = CodeEntity(
        id="stale-rel-test::Base",
        name="Base", entity_type=EntityType.CLASS,
        repository="stale-rel-test", file_path="svc.py",
        line_number=1, language="python",
    )
    my_class = CodeEntity(
        id="stale-rel-test::MyClass",
        name="MyClass", entity_type=EntityType.CLASS,
        repository="stale-rel-test", file_path="svc.py",
        line_number=2, language="python",
    )
    extends_rel = CodeRelationship(
        source_id="stale-rel-test::MyClass",
        target_id="stale-rel-test::Base",
        relationship_type=RelationshipType.EXTENDS,
    )

    # Initial: MyClass extends Base
    await write_backend.apply_indexing_changes(
        repository="stale-rel-test",
        source="/tmp", commit_sha=None,
        reindex_preparations=[("svc.py", {file_entity.id, base_class.id, my_class.id})],
        entities=[file_entity, base_class, my_class],
        relationships=[extends_rel],
        stale_file_paths=[],
    )

    # Verify the relationship row exists
    pool = write_backend._require_pool()
    rel_count = await pool.fetchval(
        "SELECT COUNT(*) FROM code_references WHERE source_symbol_id = $1 AND ref_type = 'EXTENDS'",
        "stale-rel-test::MyClass",
    )
    assert rel_count == 1, f"Initial: expected 1 EXTENDS row, got {rel_count}"

    # Refactor: MyClass no longer extends Base. Same entities, no relationships.
    await write_backend.apply_indexing_changes(
        repository="stale-rel-test",
        source="/tmp", commit_sha=None,
        reindex_preparations=[("svc.py", {file_entity.id, base_class.id, my_class.id})],
        entities=[file_entity, base_class, my_class],
        relationships=[],
        stale_file_paths=[],
    )

    # The stale EXTENDS row must be gone
    rel_count_after = await pool.fetchval(
        "SELECT COUNT(*) FROM code_references WHERE source_symbol_id = $1 AND ref_type = 'EXTENDS'",
        "stale-rel-test::MyClass",
    )
    assert rel_count_after == 0, f"After refactor: expected 0 EXTENDS rows, got {rel_count_after}"

    await write_backend.delete_repository("stale-rel-test")


async def test_fresh_deployment_can_list_repositories_without_indexing(postgres_dsn):
    """On a truly fresh Postgres DB (no indexing run yet), a backend that
    calls initialize_schema() before list_repositories() must return an
    empty list, not raise UndefinedTableError."""
    backend = PostgresWriteBackend(
        dsn=postgres_dsn,
        embedding_dimensions=1536,
        embedding_model="text-embedding-3-small",
    )
    try:
        await backend.connect()
        # Drop any tables left over from previous tests in this session so
        # we're testing the truly-fresh case
        pool = backend._require_pool()
        async with pool.acquire() as conn:
            await conn.execute("DROP TABLE IF EXISTS code_embeddings CASCADE")
            await conn.execute("DROP TABLE IF EXISTS code_references CASCADE")
            await conn.execute("DROP TABLE IF EXISTS code_symbols CASCADE")
            await conn.execute("DROP TABLE IF EXISTS code_repos CASCADE")
            await conn.execute("DROP TABLE IF EXISTS embedding_metadata CASCADE")

        # A naive call to list_repositories without initialize_schema would
        # now raise UndefinedTableError. With the bootstrap fix, calling
        # initialize_schema first makes this work:
        await backend.initialize_schema()
        repos = await backend.list_repositories()
        assert repos == []
    finally:
        await backend.close()


async def test_delete_repository_removes_entities(write_backend):
    # Setup: ensure there is a repo
    await write_backend.upsert_repository(
        name="test-repo-delete",
        source="/tmp/test",
        commit_sha=None,
        entity_count=0,
    )

    file_entity = CodeEntity(
        id="test-repo-delete::src/bar.py",
        name="bar.py",
        entity_type=EntityType.FILE,
        repository="test-repo-delete",
        file_path="src/bar.py",
        line_number=1,
        language="python",
        content_hash="hash2",
    )
    await write_backend.apply_indexing_changes(
        repository="test-repo-delete",
        source="/tmp/test",
        commit_sha=None,
        reindex_preparations=[],
        entities=[file_entity],
        relationships=[],
        stale_file_paths=[],
    )

    count_before = await write_backend.count_repository_entities("test-repo-delete")
    assert count_before >= 1

    await write_backend.delete_repository("test-repo-delete")

    count_after = await write_backend.count_repository_entities("test-repo-delete")
    assert count_after == 0

    # Repository row should also be gone
    repo = await write_backend.get_repository("test-repo-delete")
    assert repo is None


async def test_model_drift_drops_embeddings_and_clears_hashes(postgres_dsn):
    """Switching embedding model (same dimension) must drop code_embeddings
    and clear File.content_hash so the next indexing run regenerates
    embeddings with the new model."""
    # Fresh backend with model A
    backend_a = PostgresWriteBackend(
        dsn=postgres_dsn,
        embedding_dimensions=1536,
        embedding_model="text-embedding-3-small",
    )
    try:
        await backend_a.connect()
        # Drop any leftover state from previous tests
        pool = backend_a._require_pool()
        async with pool.acquire() as conn:
            await conn.execute("DROP TABLE IF EXISTS code_embeddings CASCADE")
            await conn.execute("DROP TABLE IF EXISTS code_references CASCADE")
            await conn.execute("DROP TABLE IF EXISTS code_symbols CASCADE")
            await conn.execute("DROP TABLE IF EXISTS code_repos CASCADE")
            await conn.execute("DROP TABLE IF EXISTS embedding_metadata CASCADE")

        await backend_a.initialize_schema()

        # Seed a File entity with a content_hash so we can verify it gets cleared
        file_entity = CodeEntity(
            id="model-drift-test::src/foo.py",
            name="foo.py",
            entity_type=EntityType.FILE,
            repository="model-drift-test",
            file_path="src/foo.py",
            line_number=1,
            language="python",
            content_hash="original-hash",
        )
        await backend_a.apply_indexing_changes(
            repository="model-drift-test",
            source="/tmp/test",
            commit_sha=None,
            reindex_preparations=[("src/foo.py", {file_entity.id})],
            entities=[file_entity],
            relationships=[],
            stale_file_paths=[],
        )
        # Confirm the hash was stored
        hashes = await backend_a.get_file_hashes("model-drift-test")
        assert hashes.get("src/foo.py") == "original-hash"
    finally:
        await backend_a.close()

    # New backend with model B — same dim, different model
    backend_b = PostgresWriteBackend(
        dsn=postgres_dsn,
        embedding_dimensions=1536,
        embedding_model="text-embedding-ada-002",
    )
    try:
        await backend_b.connect()
        await backend_b.initialize_schema()  # should detect drift, drop embeddings, clear hashes

        # The file's content_hash should now be NULL
        hashes_after = await backend_b.get_file_hashes("model-drift-test")
        assert "src/foo.py" not in hashes_after, \
            f"Expected content_hash cleared after model drift; got: {hashes_after}"

        # Cleanup
        await backend_b.delete_repository("model-drift-test")
    finally:
        await backend_b.close()
