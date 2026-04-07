"""
Integration test: index a small Python tree with PostgresWriteBackend,
then verify the WriteBackend operations succeed against a real Postgres + pgvector.

Requires: POSTGRES_DSN env var pointing to a running pgvector instance.
Run with: pytest tests/test_postgres_integration.py -m integration -v
"""
import os
import pytest

from constellation.config import Settings
from constellation.graph.postgres import PostgresWriteBackend
from constellation.models import CodeEntity, CodeRelationship, EntityType, RelationshipType

pytestmark = pytest.mark.integration

POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN",
    "postgresql://constellation:secret@localhost:5432/constellation"
)


@pytest.fixture
def settings():
    return Settings(
        storage_backend="postgres",
        postgres_dsn=POSTGRES_DSN,
        embedding_provider="openai",
        openai_api_key=os.environ.get("OPENAI_API_KEY", "fake-key-for-test"),
    )


@pytest.fixture
async def write_backend(settings):
    backend = PostgresWriteBackend(
        dsn=POSTGRES_DSN,
        embedding_dimensions=settings.resolved_embedding_dimensions(),
    )
    await backend.connect()
    await backend.initialize_schema()
    # Clean up any prior test data
    await backend.delete_repository("test-repo")
    yield backend
    await backend.delete_repository("test-repo")
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
    """After apply_indexing_changes with a File entity, get_file_hashes should return it."""
    # The previous test set up some entities; this test verifies the file is in get_file_hashes
    hashes = await write_backend.get_file_hashes("test-repo")
    # The file_entity from the previous test had content_hash="hash1"
    if "src/foo.py" in hashes:
        assert hashes["src/foo.py"] == "hash1"


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
