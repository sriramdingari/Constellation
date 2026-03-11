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
    client.delete_stale_files = AsyncMock()
    client.upsert_repository = AsyncMock()
    client.initialize_schema = AsyncMock()
    client.get_repository = AsyncMock(return_value=None)
    client.list_repositories = AsyncMock(return_value=[])
    client.delete_repository = AsyncMock()
    return client


@pytest.fixture
def mock_embedding_provider():
    provider = AsyncMock()
    provider.model_name = "test-model"
    provider.dimensions = 1536
    provider.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
    return provider
