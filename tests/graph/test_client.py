"""Tests for constellation.graph.client — Neo4j async graph client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from constellation.config import Settings
from constellation.models import CodeEntity, CodeRelationship, EntityType, RelationshipType
from constellation.graph import schema, queries
from constellation.graph.client import GraphClient


@pytest.fixture
def settings():
    return Settings(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="testpass",
        embedding_dimensions=1536,
    )


@pytest.fixture
def mock_result():
    """Create a mock Neo4j Result that supports async iteration."""
    result = AsyncMock()
    result.__aiter__.return_value = []
    return result


@pytest.fixture
def mock_session(mock_result):
    """Create a mock Neo4j async session."""
    session = AsyncMock()
    session.run = AsyncMock(return_value=mock_result)
    session.begin_transaction = AsyncMock()
    # Support async context manager
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


@pytest.fixture
def mock_tx(mock_result):
    """Create a mock Neo4j async transaction."""
    tx = AsyncMock()
    tx.run = AsyncMock(return_value=mock_result)
    tx.commit = AsyncMock()
    tx.rollback = AsyncMock()
    return tx


@pytest.fixture
def mock_driver(mock_session, mock_tx):
    """Create a mock Neo4j async driver."""
    driver = AsyncMock()
    driver.verify_connectivity = AsyncMock()
    driver.close = AsyncMock()
    driver.session = MagicMock(return_value=mock_session)
    mock_session.begin_transaction.return_value = mock_tx
    return driver


@pytest.fixture
def client(settings, mock_driver):
    """Create a GraphClient with a pre-injected mock driver."""
    c = GraphClient(settings)
    c._driver = mock_driver
    return c


class TestConnect:
    """Test GraphClient.connect()."""

    async def test_connect_creates_driver_and_verifies(self, settings):
        with patch("constellation.graph.client.AsyncGraphDatabase") as mock_agd:
            mock_drv = AsyncMock()
            mock_drv.verify_connectivity = AsyncMock()
            mock_agd.driver.return_value = mock_drv

            client = GraphClient(settings)
            await client.connect()

            mock_agd.driver.assert_called_once_with(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
            mock_drv.verify_connectivity.assert_awaited_once()
            assert client._driver is mock_drv


class TestClose:
    """Test GraphClient.close()."""

    async def test_close_closes_driver(self, client, mock_driver):
        await client.close()
        mock_driver.close.assert_awaited_once()
        assert client._driver is None

    async def test_close_when_not_connected_no_error(self, settings):
        client = GraphClient(settings)
        assert client._driver is None
        await client.close()  # Should not raise


class TestQuery:
    """Test GraphClient.query()."""

    async def test_query_runs_cypher_and_returns_records(
        self, client, mock_session, mock_result
    ):
        record1 = MagicMock()
        record1.data.return_value = {"name": "Alice"}
        record2 = MagicMock()
        record2.data.return_value = {"name": "Bob"}
        mock_result.__aiter__.return_value = [record1, record2]

        result = await client.query("MATCH (n) RETURN n", limit=10)

        mock_session.run.assert_awaited_once_with("MATCH (n) RETURN n", {"limit": 10})
        assert result == [{"name": "Alice"}, {"name": "Bob"}]

    async def test_query_returns_all_records_without_truncation(
        self, client, mock_session, mock_result
    ):
        records = []
        for i in range(1001):
            record = MagicMock()
            record.data.return_value = {"idx": i}
            records.append(record)

        mock_result.__aiter__.return_value = records

        result = await client.query("MATCH (n) RETURN n")

        assert len(result) == 1001

    async def test_query_auto_connects_if_not_connected(self, settings):
        with patch("constellation.graph.client.AsyncGraphDatabase") as mock_agd:
            mock_result = AsyncMock()
            mock_result.__aiter__.return_value = []

            mock_sess = AsyncMock()
            mock_sess.run = AsyncMock(return_value=mock_result)
            mock_sess.__aenter__ = AsyncMock(return_value=mock_sess)
            mock_sess.__aexit__ = AsyncMock(return_value=False)

            mock_drv = AsyncMock()
            mock_drv.verify_connectivity = AsyncMock()
            mock_drv.session = MagicMock(return_value=mock_sess)
            mock_agd.driver.return_value = mock_drv

            client = GraphClient(settings)
            assert client._driver is None

            result = await client.query("RETURN 1")

            # Should have auto-connected
            mock_agd.driver.assert_called_once()
            mock_drv.verify_connectivity.assert_awaited_once()
            assert result == []


class TestInitializeSchema:
    """Test GraphClient.initialize_schema()."""

    async def test_initialize_schema_runs_all_constraints_indexes_vectors(
        self, client, mock_session, mock_result
    ):
        mock_result.__aiter__.return_value = []
        await client.initialize_schema()

        # Count expected calls:
        # len(CONSTRAINTS) + len(COMPOSITE_INDEXES) + 2 * len(EMBEDDABLE_LABELS)
        # (SHOW INDEXES + CREATE VECTOR INDEX per label)
        expected_count = (
            len(schema.CONSTRAINTS)
            + len(schema.COMPOSITE_INDEXES)
            + 2 * len(schema.EMBEDDABLE_LABELS)
        )
        assert mock_session.run.await_count == expected_count

        # Verify each constraint was run
        all_calls = [call.args[0] for call in mock_session.run.await_args_list]
        for constraint in schema.CONSTRAINTS:
            assert constraint in all_calls

        # Verify each composite index was run
        for index in schema.COMPOSITE_INDEXES:
            assert index in all_calls

        assert all_calls.count(queries.GET_VECTOR_INDEX) == len(schema.EMBEDDABLE_LABELS)

        # Verify vector indexes were created for each embeddable label
        for label in schema.EMBEDDABLE_LABELS:
            index_name = f"vector_{label.lower()}_embedding"
            expected_q = queries.CREATE_VECTOR_INDEX.format(
                index_name=index_name, label=label
            )
            assert expected_q in all_calls

    async def test_initialize_schema_uses_provider_resolved_dimensions(
        self, mock_session, mock_result
    ):
        settings = Settings(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="testpass",
            embedding_provider="ollama",
            embedding_dimensions=1536,
            ollama_embedding_dimensions=768,
        )
        client = GraphClient(settings)
        client._driver = AsyncMock()
        client._driver.session = MagicMock(return_value=mock_session)

        mock_result.__aiter__.return_value = []

        await client.initialize_schema()

        vector_index_calls = [
            call.args[1]
            for call in mock_session.run.await_args_list
            if "VECTOR INDEX" in call.args[0]
        ]
        assert vector_index_calls
        assert all(params["dimensions"] == 768 for params in vector_index_calls)

    async def test_initialize_schema_recreates_vector_indexes_on_dimension_change(
        self, client, mock_session
    ):
        existing_index_record = MagicMock()
        existing_index_record.data.return_value = {
            "options": {"indexConfig": {"vector.dimensions": 768}}
        }
        existing_index_result = AsyncMock()
        existing_index_result.__aiter__.return_value = [existing_index_record]

        empty_result = AsyncMock()
        empty_result.__aiter__.return_value = []

        async def run_side_effect(cypher, params):
            if cypher == queries.GET_VECTOR_INDEX:
                return existing_index_result
            return empty_result

        mock_session.run = AsyncMock(side_effect=run_side_effect)

        await client.initialize_schema()

        all_calls = [call.args[0] for call in mock_session.run.await_args_list]
        for label in schema.EMBEDDABLE_LABELS:
            index_name = f"vector_{label.lower()}_embedding"
            assert queries.drop_index_query(index_name) in all_calls


class TestUpsertEntities:
    """Test GraphClient.upsert_entities()."""

    async def test_upsert_entities_groups_by_label_and_calls_query(
        self, client, mock_session, mock_result
    ):
        record = MagicMock()
        record.data.return_value = {"count": 2}
        mock_result.__aiter__.return_value = [record]

        entities = [
            CodeEntity(
                id="repo::MyClass",
                name="MyClass",
                entity_type=EntityType.CLASS,
                repository="repo",
                file_path="src/MyClass.java",
                line_number=1,
                line_end=50,
                language="java",
                signature="public class MyClass",
                stereotypes=["service"],
            ),
            CodeEntity(
                id="repo::getUser",
                name="getUser",
                entity_type=EntityType.METHOD,
                repository="repo",
                file_path="src/MyClass.java",
                line_number=10,
                line_end=20,
                language="java",
                code="public void getUser() {}",
                signature="public void getUser()",
                return_type="void",
                docstring="Gets user",
                modifiers=["public"],
                content_hash="abc123",
            ),
        ]

        total = await client.upsert_entities(entities)

        # Should have called query twice (one for Class, one for Method)
        assert mock_session.run.await_count == 2

        # Verify the queries used correct labels
        call_args = [call.args[0] for call in mock_session.run.await_args_list]
        class_query = queries.upsert_entities_query("Class")
        method_query = queries.upsert_entities_query("Method")
        assert class_query in call_args
        assert method_query in call_args

        # Total should be sum of counts from both calls
        assert total == 4  # 2 + 2

    async def test_upsert_entities_empty_list_returns_zero(self, client, mock_session):
        result = await client.upsert_entities([])
        assert result == 0
        mock_session.run.assert_not_awaited()

    async def test_upsert_entities_passes_correct_properties(
        self, client, mock_session, mock_result
    ):
        record = MagicMock()
        record.data.return_value = {"count": 1}
        mock_result.__aiter__.return_value = [record]

        entity = CodeEntity(
            id="repo::Foo",
            name="Foo",
            entity_type=EntityType.CLASS,
            repository="repo",
            file_path="src/Foo.java",
            line_number=1,
            language="java",
            properties={"field_type": "String"},
            embedding=[0.1, 0.2, 0.3],
        )

        await client.upsert_entities([entity])

        # Verify the entity dict passed to the query
        call_params = mock_session.run.await_args_list[0].args[1]
        entity_dicts = call_params["entities"]
        assert len(entity_dicts) == 1
        props = entity_dicts[0]["properties"]
        assert props["id"] == "repo::Foo"
        assert props["name"] == "Foo"
        assert props["repository"] == "repo"
        assert props["file_path"] == "src/Foo.java"
        assert props["line_number"] == 1
        assert props["language"] == "java"
        assert props["field_type"] == "String"
        assert props["embedding"] == [0.1, 0.2, 0.3]


class TestCreateRelationships:
    """Test GraphClient.create_relationships()."""

    async def test_create_relationships_groups_by_type(
        self, client, mock_session, mock_result
    ):
        record = MagicMock()
        record.data.return_value = {"count": 3}
        mock_result.__aiter__.return_value = [record]

        relationships = [
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
            CodeRelationship(
                source_id="repo::E",
                target_id="repo::F",
                relationship_type=RelationshipType.CALLS,
                properties={"weight": 1},
            ),
        ]

        total = await client.create_relationships(relationships)

        # Should have called query twice (CALLS and EXTENDS)
        assert mock_session.run.await_count == 2

        call_args = [call.args[0] for call in mock_session.run.await_args_list]
        calls_query = queries.create_relationships_query("CALLS")
        extends_query = queries.create_relationships_query("EXTENDS")
        assert calls_query in call_args
        assert extends_query in call_args

        # Total = 3 + 3 (same mock record returned for each)
        assert total == 6

    async def test_create_relationships_empty_list_returns_zero(
        self, client, mock_session
    ):
        result = await client.create_relationships([])
        assert result == 0
        mock_session.run.assert_not_awaited()

    async def test_create_relationships_passes_correct_dicts(
        self, client, mock_session, mock_result
    ):
        record = MagicMock()
        record.data.return_value = {"count": 1}
        mock_result.__aiter__.return_value = [record]

        rel = CodeRelationship(
            source_id="repo::A",
            target_id="repo::B",
            relationship_type=RelationshipType.IMPLEMENTS,
            properties={"since": "v1"},
        )

        await client.create_relationships([rel])

        call_params = mock_session.run.await_args_list[0].args[1]
        rel_dicts = call_params["relationships"]
        assert len(rel_dicts) == 1
        assert rel_dicts[0]["source_id"] == "repo::A"
        assert rel_dicts[0]["target_id"] == "repo::B"
        assert rel_dicts[0]["properties"] == {"since": "v1"}


class TestGetFileHashes:
    """Test GraphClient.get_file_hashes()."""

    async def test_get_file_hashes_returns_dict(
        self, client, mock_session, mock_result
    ):
        record1 = MagicMock()
        record1.data.return_value = {
            "file_path": "src/A.java",
            "content_hash": "hash1",
        }
        record2 = MagicMock()
        record2.data.return_value = {
            "file_path": "src/B.java",
            "content_hash": "hash2",
        }
        mock_result.__aiter__.return_value = [record1, record2]

        result = await client.get_file_hashes("my-repo")

        assert result == {"src/A.java": "hash1", "src/B.java": "hash2"}
        mock_session.run.assert_awaited_once_with(
            queries.GET_FILE_HASHES, {"repository": "my-repo"}
        )


class TestDeleteStaleFiles:
    """Test GraphClient.delete_stale_files()."""

    async def test_delete_stale_files_calls_query(
        self, client, mock_session, mock_result
    ):
        deleted_zero_record = MagicMock()
        deleted_zero_record.data.return_value = {"count": 0}
        mock_result.__aiter__.return_value = [deleted_zero_record]
        paths = ["src/Old.java", "src/Removed.java"]

        await client.delete_stale_files("my-repo", paths)

        mock_session.run.assert_any_await(
            queries.DELETE_STALE_FILES,
            {"repository": "my-repo", "file_paths": paths},
        )
        mock_session.run.assert_any_await(
            queries.DELETE_ORPHAN_PACKAGES,
            {"repository": "my-repo"},
        )
        mock_session.run.assert_any_await(
            queries.DELETE_ORPHAN_REFERENCES,
            {"repository": "my-repo"},
        )
        assert mock_session.run.await_count == 3

    async def test_delete_stale_files_empty_list_does_nothing(
        self, client, mock_session
    ):
        await client.delete_stale_files("my-repo", [])
        mock_session.run.assert_not_awaited()


class TestPrepareFileReindex:
    """Test GraphClient.prepare_file_reindex()."""

    async def test_prepare_file_reindex_replaces_stale_entities_only(
        self, client, mock_session
    ):
        snapshot_record_1 = MagicMock()
        snapshot_record_1.data.return_value = {
            "id": "repo::src/app.py",
            "labels": ["File"],
        }
        snapshot_record_2 = MagicMock()
        snapshot_record_2.data.return_value = {
            "id": "repo::src/app.py#UserService",
            "labels": ["Class"],
        }
        snapshot_record_3 = MagicMock()
        snapshot_record_3.data.return_value = {
            "id": "repo::src/app.py#old_method",
            "labels": ["Method"],
        }
        snapshot_record_4 = MagicMock()
        snapshot_record_4.data.return_value = {
            "id": "repo::com.example",
            "labels": ["Package"],
        }

        snapshot_result = AsyncMock()
        snapshot_result.__aiter__.return_value = [
            snapshot_record_1,
            snapshot_record_2,
            snapshot_record_3,
            snapshot_record_4,
        ]
        empty_result = AsyncMock()
        empty_result.__aiter__.return_value = []

        async def run_side_effect(cypher, params):
            if cypher == queries.GET_FILE_ENTITY_SNAPSHOT:
                return snapshot_result
            return empty_result

        mock_session.run = AsyncMock(side_effect=run_side_effect)

        await client.prepare_file_reindex(
            "repo",
            "src/app.py",
            {"repo::src/app.py", "repo::src/app.py#UserService"},
        )

        mock_session.run.assert_any_await(
            queries.GET_FILE_ENTITY_SNAPSHOT,
            {"repository": "repo", "file_path": "src/app.py"},
        )
        mock_session.run.assert_any_await(
            queries.DELETE_FILE_OUTBOUND_RELATIONSHIPS,
            {"repository": "repo", "file_path": "src/app.py"},
        )
        mock_session.run.assert_any_await(
            queries.DELETE_ENTITIES_BY_IDS,
            {"repository": "repo", "entity_ids": ["repo::src/app.py#old_method"]},
        )

    async def test_prepare_file_reindex_noops_when_file_has_no_existing_nodes(
        self, client, mock_session, mock_result
    ):
        mock_result.__aiter__.return_value = []

        await client.prepare_file_reindex(
            "repo",
            "src/new.py",
            {"repo::src/new.py"},
        )

        mock_session.run.assert_awaited_once_with(
            queries.GET_FILE_ENTITY_SNAPSHOT,
            {"repository": "repo", "file_path": "src/new.py"},
        )


class TestApplyIndexingChanges:
    """Test GraphClient.apply_indexing_changes()."""

    async def test_apply_indexing_changes_uses_single_transaction(
        self, client, mock_session, mock_tx
    ):
        empty_result = AsyncMock()
        empty_result.__aiter__.return_value = []

        count_record = MagicMock()
        count_record.data.return_value = {"count": 4}
        count_result = AsyncMock()
        count_result.__aiter__.return_value = [count_record]

        async def run_side_effect(cypher, params):
            if cypher == queries.GET_FILE_ENTITY_SNAPSHOT:
                return empty_result
            if cypher == queries.COUNT_REPOSITORY_ENTITIES:
                return count_result
            return empty_result

        mock_tx.run = AsyncMock(side_effect=run_side_effect)

        entities_created, relationships_created, repository_entity_count = (
            await client.apply_indexing_changes(
                repository="repo",
                source="/tmp/repo",
                commit_sha="abc123",
                reindex_preparations=[("src/app.py", {"repo::src/app.py"})],
                entities=[
                    CodeEntity(
                        id="repo::src/app.py",
                        name="app.py",
                        entity_type=EntityType.FILE,
                        repository="repo",
                        file_path="src/app.py",
                        line_number=1,
                        language="python",
                        content_hash="hash",
                    )
                ],
                relationships=[
                    CodeRelationship(
                        source_id="repo::src/app.py",
                        target_id="repo::src/app.py",
                        relationship_type=RelationshipType.CONTAINS,
                    )
                ],
                stale_file_paths=[],
            )
        )

        mock_session.begin_transaction.assert_awaited_once()
        mock_session.run.assert_not_awaited()
        mock_tx.commit.assert_awaited_once()
        mock_tx.rollback.assert_not_awaited()
        assert entities_created == 0
        assert relationships_created == 0
        assert repository_entity_count == 4

    async def test_apply_indexing_changes_rolls_back_on_failure(
        self, client, mock_tx
    ):
        snapshot_result = AsyncMock()
        snapshot_result.__aiter__.return_value = []

        async def run_side_effect(cypher, params):
            if cypher == queries.GET_FILE_ENTITY_SNAPSHOT:
                return snapshot_result
            if cypher == queries.upsert_entities_query("File"):
                raise RuntimeError("write failed")
            return snapshot_result

        mock_tx.run = AsyncMock(side_effect=run_side_effect)

        with pytest.raises(RuntimeError, match="write failed"):
            await client.apply_indexing_changes(
                repository="repo",
                source="/tmp/repo",
                commit_sha=None,
                reindex_preparations=[("src/app.py", {"repo::src/app.py"})],
                entities=[
                    CodeEntity(
                        id="repo::src/app.py",
                        name="app.py",
                        entity_type=EntityType.FILE,
                        repository="repo",
                        file_path="src/app.py",
                        line_number=1,
                        language="python",
                    )
                ],
                relationships=[],
                stale_file_paths=[],
            )

        mock_tx.commit.assert_not_awaited()
        mock_tx.rollback.assert_awaited_once()

    async def test_apply_indexing_changes_batches_graph_writes(
        self, mock_driver, mock_session, mock_tx
    ):
        settings = Settings(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="testpass",
            embedding_dimensions=1536,
            entity_batch_size=1,
        )
        client = GraphClient(settings)
        client._driver = mock_driver

        empty_result = AsyncMock()
        empty_result.__aiter__.return_value = []

        count_record = MagicMock()
        count_record.data.return_value = {"count": 2}
        count_result = AsyncMock()
        count_result.__aiter__.return_value = [count_record]

        async def run_side_effect(cypher, params):
            if cypher == queries.GET_FILE_ENTITY_SNAPSHOT:
                return empty_result
            if cypher == queries.COUNT_REPOSITORY_ENTITIES:
                return count_result
            return empty_result

        mock_tx.run = AsyncMock(side_effect=run_side_effect)

        await client.apply_indexing_changes(
            repository="repo",
            source="/tmp/repo",
            commit_sha="abc123",
            reindex_preparations=[],
            entities=[
                CodeEntity(
                    id="repo::src/a.py",
                    name="a.py",
                    entity_type=EntityType.FILE,
                    repository="repo",
                    file_path="src/a.py",
                    line_number=1,
                    language="python",
                ),
                CodeEntity(
                    id="repo::src/b.py",
                    name="b.py",
                    entity_type=EntityType.FILE,
                    repository="repo",
                    file_path="src/b.py",
                    line_number=1,
                    language="python",
                ),
            ],
            relationships=[
                CodeRelationship(
                    source_id="repo::src/a.py",
                    target_id="repo::src/a.py",
                    relationship_type=RelationshipType.CONTAINS,
                ),
                CodeRelationship(
                    source_id="repo::src/b.py",
                    target_id="repo::src/b.py",
                    relationship_type=RelationshipType.CONTAINS,
                ),
            ],
            stale_file_paths=[],
        )

        file_query = queries.upsert_entities_query("File")
        contains_query = queries.create_relationships_query("CONTAINS")
        all_queries = [call.args[0] for call in mock_tx.run.await_args_list]
        assert all_queries.count(file_query) == 2
        assert all_queries.count(contains_query) == 2


class TestCleanupOrphanPackages:
    """Test GraphClient.cleanup_orphan_packages()."""

    async def test_cleanup_orphan_packages_repeats_until_stable(
        self, client, mock_session
    ):
        deleted_one_record = MagicMock()
        deleted_one_record.data.return_value = {"count": 1}
        deleted_one_result = AsyncMock()
        deleted_one_result.__aiter__.return_value = [deleted_one_record]

        deleted_zero_record = MagicMock()
        deleted_zero_record.data.return_value = {"count": 0}
        deleted_zero_result = AsyncMock()
        deleted_zero_result.__aiter__.return_value = [deleted_zero_record]

        mock_session.run = AsyncMock(
            side_effect=[deleted_one_result, deleted_zero_result]
        )

        await client.cleanup_orphan_packages("my-repo")

        assert mock_session.run.await_count == 2
        mock_session.run.assert_any_await(
            queries.DELETE_ORPHAN_PACKAGES,
            {"repository": "my-repo"},
        )


class TestCountRepositoryEntities:
    """Test GraphClient.count_repository_entities()."""

    async def test_count_repository_entities_returns_count(
        self, client, mock_session, mock_result
    ):
        record = MagicMock()
        record.data.return_value = {"count": 42}
        mock_result.__aiter__.return_value = [record]

        result = await client.count_repository_entities("my-repo")

        assert result == 42
        mock_session.run.assert_awaited_once_with(
            queries.COUNT_REPOSITORY_ENTITIES,
            {"repository": "my-repo"},
        )

    async def test_count_repository_entities_returns_zero_when_empty(
        self, client, mock_session, mock_result
    ):
        mock_result.__aiter__.return_value = []

        result = await client.count_repository_entities("my-repo")

        assert result == 0


class TestUpsertRepository:
    """Test GraphClient.upsert_repository()."""

    async def test_upsert_repository_calls_query(
        self, client, mock_session, mock_result
    ):
        mock_result.__aiter__.return_value = []

        await client.upsert_repository(
            name="my-repo",
            source="/path/to/repo",
            commit_sha="abc123",
            entity_count=42,
        )

        mock_session.run.assert_awaited_once_with(
            queries.UPSERT_REPOSITORY,
            {
                "name": "my-repo",
                "source": "/path/to/repo",
                "commit_sha": "abc123",
                "entity_count": 42,
            },
        )


class TestGetRepository:
    """Test GraphClient.get_repository()."""

    async def test_get_repository_returns_dict(
        self, client, mock_session, mock_result
    ):
        record = MagicMock()
        record.data.return_value = {"r": {"name": "my-repo", "source": "/path"}}
        mock_result.__aiter__.return_value = [record]

        result = await client.get_repository("my-repo")

        assert result == {"name": "my-repo", "source": "/path"}
        mock_session.run.assert_awaited_once_with(
            queries.GET_REPOSITORY, {"name": "my-repo"}
        )

    async def test_get_repository_returns_none_when_not_found(
        self, client, mock_session, mock_result
    ):
        mock_result.__aiter__.return_value = []

        result = await client.get_repository("nonexistent")
        assert result is None


class TestListRepositories:
    """Test GraphClient.list_repositories()."""

    async def test_list_repositories_returns_list(
        self, client, mock_session, mock_result
    ):
        record1 = MagicMock()
        record1.data.return_value = {"r": {"name": "repo-a"}}
        record2 = MagicMock()
        record2.data.return_value = {"r": {"name": "repo-b"}}
        mock_result.__aiter__.return_value = [record1, record2]

        result = await client.list_repositories()

        assert result == [{"name": "repo-a"}, {"name": "repo-b"}]
        mock_session.run.assert_awaited_once_with(queries.LIST_REPOSITORIES, {})


class TestDeleteRepository:
    """Test GraphClient.delete_repository()."""

    async def test_delete_repository_calls_query(
        self, client, mock_session, mock_result
    ):
        mock_result.__aiter__.return_value = []

        await client.delete_repository("my-repo")

        mock_session.run.assert_awaited_once_with(
            queries.DELETE_REPOSITORY, {"name": "my-repo"}
        )
