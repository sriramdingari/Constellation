import pytest
from unittest.mock import AsyncMock

from constellation.models import CodeEntity, EntityType


@pytest.fixture
def sample_class_entity():
    return CodeEntity(
        id="test-repo::com.example.UserService",
        name="UserService", entity_type=EntityType.CLASS,
        repository="test-repo", file_path="src/UserService.java",
        line_number=5, line_end=50, language="java",
        signature="public class UserService",
        stereotypes=["spring_service"],
    )


@pytest.fixture
def sample_method_entity():
    return CodeEntity(
        id="test-repo::com.example.UserService.getUser(int)",
        name="getUser", entity_type=EntityType.METHOD,
        repository="test-repo", file_path="src/UserService.java",
        line_number=10, line_end=15, language="java",
        code='public User getUser(int id) { return repo.find(id); }',
        signature="public User getUser(int id)",
        return_type="User", docstring="Fetch user by ID",
        modifiers=["public"], stereotypes=["endpoint"],
    )


@pytest.fixture
def mock_graph_client():
    client = AsyncMock()
    client.connect = AsyncMock()
    client.close = AsyncMock()
    client.query = AsyncMock(return_value=[])
    client.get_file_hashes = AsyncMock(return_value={})
    client.upsert_entities = AsyncMock(return_value=5)
    client.create_relationships = AsyncMock(return_value=3)
    client.prepare_file_reindex = AsyncMock(return_value=False)
    client.delete_stale_files = AsyncMock()
    client.cleanup_orphan_packages = AsyncMock()
    client.cleanup_orphan_references = AsyncMock()
    client.count_repository_entities = AsyncMock(return_value=5)
    client.upsert_repository = AsyncMock()
    client.initialize_schema = AsyncMock()
    client.get_repository = AsyncMock(return_value=None)
    client.list_repositories = AsyncMock(return_value=[])
    client.delete_repository = AsyncMock()

    async def _apply_indexing_changes(
        *,
        repository,
        source,
        commit_sha,
        reindex_preparations,
        entities,
        relationships,
        stale_file_paths,
    ):
        changed_existing_files = False
        for file_path, current_entity_ids in reindex_preparations:
            changed_existing_files = await client.prepare_file_reindex(
                repository=repository,
                file_path=file_path,
                current_entity_ids=current_entity_ids,
            ) or changed_existing_files

        entities_created = 0
        if entities:
            entities_created = await client.upsert_entities(entities)

        if stale_file_paths:
            await client.delete_stale_files(repository=repository, file_paths=stale_file_paths)

        relationships_created = 0
        if relationships:
            relationships_created = await client.create_relationships(relationships)

        if not stale_file_paths and changed_existing_files:
            await client.cleanup_orphan_packages(repository)
            await client.cleanup_orphan_references(repository)

        repository_entity_count = await client.count_repository_entities(repository)
        await client.upsert_repository(
            name=repository,
            source=source,
            commit_sha=commit_sha,
            entity_count=repository_entity_count,
        )
        return entities_created, relationships_created, repository_entity_count

    client.apply_indexing_changes = AsyncMock(side_effect=_apply_indexing_changes)
    return client


@pytest.fixture
def mock_embedding_provider():
    provider = AsyncMock()
    provider.model_name = "test-model"
    provider.dimensions = 1536
    provider.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
    return provider
