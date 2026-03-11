"""Integration tests — require running Neo4j and Redis.

Run with: pytest tests/test_integration.py -v -m integration
Requires: docker compose up -d neo4j redis
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from constellation.config import Settings
from constellation.embeddings.base import BaseEmbeddingProvider
from constellation.graph.client import GraphClient
from constellation.indexer.pipeline import IndexingPipeline
from constellation.parsers.registry import get_default_registry


class FakeEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider that returns deterministic fake vectors."""

    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def dimensions(self) -> int:
        return 8

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1 * (i + 1)] * 8 for i in range(len(texts))]


@pytest.fixture
def integration_settings():
    return Settings(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="constellation",
        redis_url="redis://localhost:6379",
        embedding_provider="openai",
        embedding_model="fake",
        embedding_dimensions=8,
        embedding_batch_size=4,
        entity_batch_size=50,
    )


@pytest.fixture
def graph_client(integration_settings):
    return GraphClient(integration_settings)


@pytest.fixture
def embedding_provider():
    return FakeEmbeddingProvider()


@pytest.fixture
def parser_registry():
    return get_default_registry()


@pytest.fixture
def sample_repo(tmp_path):
    """Create a temporary repo with Java and Python files."""
    # Python file
    py_file = tmp_path / "service.py"
    py_file.write_text(
        'class UserService:\n'
        '    """Manages users."""\n'
        '    def get_user(self, user_id: int) -> dict:\n'
        '        return {"id": user_id}\n'
        '\n'
        '    def delete_user(self, user_id: int) -> None:\n'
        '        pass\n'
    )

    # Java file
    java_file = tmp_path / "App.java"
    java_file.write_text(
        'package com.example;\n'
        '\n'
        'public class App {\n'
        '    public String hello() {\n'
        '        return "Hello";\n'
        '    }\n'
        '}\n'
    )

    return tmp_path


REPO_NAME = "integration-test-repo"


@pytest.mark.integration
class TestFullIndexingPipeline:
    def test_index_local_repo(
        self, graph_client, embedding_provider, parser_registry,
        integration_settings, sample_repo,
    ):
        assert {".py", ".java", ".js", ".jsx", ".ts", ".tsx", ".cs"} <= (
            parser_registry.supported_extensions
        )

        async def run():
            try:
                await graph_client.connect()
                await graph_client.initialize_schema()

                pipeline = IndexingPipeline(
                    graph_client=graph_client,
                    embedding_provider=embedding_provider,
                    parser_registry=parser_registry,
                    settings=integration_settings,
                )
                result = await pipeline.run(
                    source=str(sample_repo),
                    name=REPO_NAME,
                )

                assert result.files_total == 2
                assert result.files_processed == 2
                assert result.entities_created > 0
                assert result.errors == []

                # Verify repository exists in graph
                repo = await graph_client.get_repository(REPO_NAME)
                assert repo is not None
            finally:
                await graph_client.close()

        asyncio.run(run())

    def test_reindex_unchanged_skips_files(
        self, graph_client, embedding_provider, parser_registry,
        integration_settings, sample_repo,
    ):
        async def run():
            try:
                await graph_client.connect()
                await graph_client.initialize_schema()

                pipeline = IndexingPipeline(
                    graph_client=graph_client,
                    embedding_provider=embedding_provider,
                    parser_registry=parser_registry,
                    settings=integration_settings,
                )

                # First index
                await pipeline.run(source=str(sample_repo), name=REPO_NAME)

                # Second index — unchanged files should be skipped
                result2 = await pipeline.run(source=str(sample_repo), name=REPO_NAME)
                assert result2.files_skipped == result2.files_total
                assert result2.files_processed == 0
            finally:
                await graph_client.close()

        asyncio.run(run())

    def test_modified_file_reprocessed(
        self, graph_client, embedding_provider, parser_registry,
        integration_settings, sample_repo,
    ):
        async def run():
            try:
                await graph_client.connect()
                await graph_client.initialize_schema()

                pipeline = IndexingPipeline(
                    graph_client=graph_client,
                    embedding_provider=embedding_provider,
                    parser_registry=parser_registry,
                    settings=integration_settings,
                )

                # First index
                await pipeline.run(source=str(sample_repo), name=REPO_NAME)

                # Modify a file
                py_file = sample_repo / "service.py"
                py_file.write_text(
                    'class UserService:\n'
                    '    def get_user(self, user_id: int) -> dict:\n'
                    '        return {"id": user_id, "name": "updated"}\n'
                )

                # Reindex — modified file should be reprocessed
                result = await pipeline.run(source=str(sample_repo), name=REPO_NAME)
                assert result.files_processed >= 1

                deleted_method_rows = await graph_client.query(
                    """
                    MATCH (m:Method {id: $id})
                    RETURN count(m) AS count
                    """,
                    id=f"{REPO_NAME}::service.UserService.delete_user",
                )
                assert deleted_method_rows[0]["count"] == 0
            finally:
                await graph_client.close()

        asyncio.run(run())

    def test_declaration_stable_reindex_reports_zero_new_entities(
        self, graph_client, embedding_provider, parser_registry,
        integration_settings, tmp_path,
    ):
        repo_name = f"{REPO_NAME}-stable-{tmp_path.name}"
        stable_repo = tmp_path / "stable"
        stable_repo.mkdir()
        service_file = stable_repo / "service.py"
        service_file.write_text(
            "class UserService:\n"
            "    def get_user(self):\n"
            "        return {'version': 1}\n"
        )

        async def run():
            try:
                await graph_client.connect()
                await graph_client.initialize_schema()

                pipeline = IndexingPipeline(
                    graph_client=graph_client,
                    embedding_provider=embedding_provider,
                    parser_registry=parser_registry,
                    settings=integration_settings,
                )

                await pipeline.run(source=str(stable_repo), name=repo_name)

                service_file.write_text(
                    "class UserService:\n"
                    "    def get_user(self):\n"
                    "        return {'version': 2}\n"
                )

                result = await pipeline.run(source=str(stable_repo), name=repo_name)
                assert result.entities_created == 0
            finally:
                await graph_client.delete_repository(repo_name)
                await graph_client.close()

        asyncio.run(run())

    def test_reintroduced_file_restores_cross_file_relationships(
        self, graph_client, embedding_provider, parser_registry,
        integration_settings, tmp_path,
    ):
        repo_name = f"{REPO_NAME}-restored-{tmp_path.name}"
        restored_repo = tmp_path / "restored"
        restored_repo.mkdir()
        base_file = restored_repo / "BaseEntity.java"
        service_file = restored_repo / "OrderService.java"
        base_file.write_text(
            "package com.example.shared;\n"
            "public class BaseEntity {}\n"
        )
        service_file.write_text(
            "package com.example.shared;\n"
            "public class OrderService extends BaseEntity {}\n"
        )

        async def run():
            try:
                await graph_client.connect()
                await graph_client.initialize_schema()

                pipeline = IndexingPipeline(
                    graph_client=graph_client,
                    embedding_provider=embedding_provider,
                    parser_registry=parser_registry,
                    settings=integration_settings,
                )

                await pipeline.run(source=str(restored_repo), name=repo_name)

                base_file.unlink()
                await pipeline.run(source=str(restored_repo), name=repo_name)

                removed_rows = await graph_client.query(
                    """
                    MATCH (:Class {id: $source})-[r:EXTENDS]->(:Class {id: $target})
                    RETURN count(r) AS count
                    """,
                    source=f"{repo_name}::com.example.shared.OrderService",
                    target=f"{repo_name}::com.example.shared.BaseEntity",
                )
                assert removed_rows[0]["count"] == 0

                base_file.write_text(
                    "package com.example.shared;\n"
                    "public class BaseEntity {}\n"
                )
                await pipeline.run(source=str(restored_repo), name=repo_name)

                restored_rows = await graph_client.query(
                    """
                    MATCH (:Class {id: $source})-[r:EXTENDS]->(:Class {id: $target})
                    RETURN count(r) AS count
                    """,
                    source=f"{repo_name}::com.example.shared.OrderService",
                    target=f"{repo_name}::com.example.shared.BaseEntity",
                )
                assert restored_rows[0]["count"] == 1
            finally:
                await graph_client.delete_repository(repo_name)
                await graph_client.close()

        asyncio.run(run())

    def test_java_calls_persist_with_method_and_reference_targets(
        self, graph_client, embedding_provider, parser_registry,
        integration_settings, tmp_path,
    ):
        repo_name = f"{REPO_NAME}-java-calls-{tmp_path.name}"
        java_repo = tmp_path / "java-calls"
        java_repo.mkdir()
        (java_repo / "OrderService.java").write_text(
            "package com.example;\n"
            "public class OrderService {\n"
            "    public void processOrder(String id) {\n"
            "        save();\n"
            "        audit.log();\n"
            "    }\n"
            "    public void save() {}\n"
            "}\n"
        )

        async def run():
            try:
                await graph_client.connect()
                await graph_client.initialize_schema()

                pipeline = IndexingPipeline(
                    graph_client=graph_client,
                    embedding_provider=embedding_provider,
                    parser_registry=parser_registry,
                    settings=integration_settings,
                )

                await pipeline.run(source=str(java_repo), name=repo_name)

                method_rows = await graph_client.query(
                    """
                    MATCH (:Method {id: $caller})-[:CALLS]->(target:Method)
                    RETURN count(target) AS count, collect(target.name) AS names
                    """,
                    caller=f"{repo_name}::com.example.OrderService.processOrder(String)",
                )
                reference_rows = await graph_client.query(
                    """
                    MATCH (:Method {id: $caller})-[:CALLS]->(target:Reference)
                    RETURN count(target) AS count, collect(target.name) AS names
                    """,
                    caller=f"{repo_name}::com.example.OrderService.processOrder(String)",
                )

                assert method_rows[0]["count"] == 1
                assert "save" in method_rows[0]["names"]
                assert reference_rows[0]["count"] == 1
                assert "audit.log" in reference_rows[0]["names"]
            finally:
                await graph_client.delete_repository(repo_name)
                await graph_client.close()

        asyncio.run(run())

    def test_reindex_clears_removed_optional_properties(
        self, graph_client, embedding_provider, parser_registry,
        integration_settings, tmp_path,
    ):
        repo_name = f"{REPO_NAME}-props-{tmp_path.name}"
        props_repo = tmp_path / "props"
        props_repo.mkdir()
        service_file = props_repo / "service.py"
        service_file.write_text(
            'class UserService:\n'
            '    """Initial docstring."""\n'
            '    pass\n'
        )

        async def run():
            try:
                await graph_client.connect()
                await graph_client.initialize_schema()

                pipeline = IndexingPipeline(
                    graph_client=graph_client,
                    embedding_provider=embedding_provider,
                    parser_registry=parser_registry,
                    settings=integration_settings,
                )

                await pipeline.run(source=str(props_repo), name=repo_name)
                initial_rows = await graph_client.query(
                    """
                    MATCH (c:Class {id: $id})
                    RETURN c.docstring AS docstring
                    """,
                    id=f"{repo_name}::service.py#UserService",
                )
                assert initial_rows[0]["docstring"] == "Initial docstring."

                service_file.write_text(
                    "class UserService:\n"
                    "    pass\n"
                )

                await pipeline.run(source=str(props_repo), name=repo_name)
                updated_rows = await graph_client.query(
                    """
                    MATCH (c:Class {id: $id})
                    RETURN c.docstring AS docstring
                    """,
                    id=f"{repo_name}::service.py#UserService",
                )
                assert updated_rows[0]["docstring"] is None
            finally:
                await graph_client.delete_repository(repo_name)
                await graph_client.close()

        asyncio.run(run())

    def test_parse_errors_do_not_mark_file_as_clean(
        self, graph_client, embedding_provider, parser_registry,
        integration_settings, tmp_path,
    ):
        repo_name = f"{REPO_NAME}-parse-{tmp_path.name}"
        parse_repo = tmp_path / "parse-errors"
        parse_repo.mkdir()
        broken_file = parse_repo / "service.py"
        broken_file.write_text(
            "class UserService:\n"
            "    def get_user(self):\n"
            "        return 1\n"
        )

        async def run():
            try:
                await graph_client.connect()
                await graph_client.initialize_schema()

                pipeline = IndexingPipeline(
                    graph_client=graph_client,
                    embedding_provider=embedding_provider,
                    parser_registry=parser_registry,
                    settings=integration_settings,
                )

                await pipeline.run(source=str(parse_repo), name=repo_name)

                broken_file.write_text(
                    "class UserService:\n"
                    "    def get_user(self):\n"
                    "        return\n"
                    "    def broken(\n"
                )

                result1 = await pipeline.run(source=str(parse_repo), name=repo_name)
                assert result1.errors

                result2 = await pipeline.run(source=str(parse_repo), name=repo_name)
                assert result2.files_processed == 1
                assert result2.files_skipped == 0
            finally:
                await graph_client.delete_repository(repo_name)
                await graph_client.close()

        asyncio.run(run())

    def test_deleted_file_removed_from_graph(
        self, graph_client, embedding_provider, parser_registry,
        integration_settings, sample_repo,
    ):
        async def run():
            try:
                await graph_client.connect()
                await graph_client.initialize_schema()

                pipeline = IndexingPipeline(
                    graph_client=graph_client,
                    embedding_provider=embedding_provider,
                    parser_registry=parser_registry,
                    settings=integration_settings,
                )

                # First index (both files)
                result1 = await pipeline.run(source=str(sample_repo), name=REPO_NAME)
                initial_total = result1.files_total

                # Delete Java file
                (sample_repo / "App.java").unlink()

                # Reindex — stale file should be removed
                result2 = await pipeline.run(source=str(sample_repo), name=REPO_NAME)
                assert result2.files_total < initial_total

                package_rows = await graph_client.query(
                    """
                    MATCH (p:Package {repository: $repository, name: $name})
                    RETURN count(p) AS count
                    """,
                    repository=REPO_NAME,
                    name="com.example",
                )
                assert package_rows[0]["count"] == 0
            finally:
                await graph_client.close()

        asyncio.run(run())

    def test_deleted_file_keeps_shared_live_nodes(
        self, graph_client, embedding_provider, parser_registry,
        integration_settings, tmp_path,
    ):
        repo_name = f"{REPO_NAME}-shared-{tmp_path.name}"
        shared_repo = tmp_path / "shared"
        shared_repo.mkdir()
        (shared_repo / "BaseEntity.java").write_text(
            "package com.example.shared;\n"
            "public class BaseEntity {}\n"
        )
        (shared_repo / "OrderService.java").write_text(
            "package com.example.shared;\n"
            "public class OrderService extends BaseEntity {}\n"
        )

        async def run():
            try:
                await graph_client.connect()
                await graph_client.initialize_schema()

                pipeline = IndexingPipeline(
                    graph_client=graph_client,
                    embedding_provider=embedding_provider,
                    parser_registry=parser_registry,
                    settings=integration_settings,
                )

                await pipeline.run(source=str(shared_repo), name=repo_name)
                (shared_repo / "OrderService.java").unlink()
                await pipeline.run(source=str(shared_repo), name=repo_name)

                base_rows = await graph_client.query(
                    """
                    MATCH (c:Class {id: $id})
                    RETURN count(c) AS count
                    """,
                    id=f"{repo_name}::com.example.shared.BaseEntity",
                )
                package_rows = await graph_client.query(
                    """
                    MATCH (p:Package {id: $id})
                    RETURN count(p) AS count
                    """,
                    id=f"{repo_name}::com.example.shared",
                )

                assert base_rows[0]["count"] == 1
                assert package_rows[0]["count"] == 1
            finally:
                await graph_client.delete_repository(repo_name)
                await graph_client.close()

        asyncio.run(run())

    def test_deleted_unrelated_file_keeps_nested_csharp_namespaces(
        self, graph_client, embedding_provider, parser_registry,
        integration_settings, tmp_path,
    ):
        repo_name = f"{REPO_NAME}-nested-ns-{tmp_path.name}"
        nested_repo = tmp_path / "nested-ns"
        nested_repo.mkdir()
        (nested_repo / "Nested.cs").write_text(
            "namespace SampleApp {\n"
            "    namespace Services {\n"
            "        public class UserService {}\n"
            "    }\n"
            "}\n"
        )
        (nested_repo / "helper.py").write_text("x = 1\n")

        async def run():
            try:
                await graph_client.connect()
                await graph_client.initialize_schema()

                pipeline = IndexingPipeline(
                    graph_client=graph_client,
                    embedding_provider=embedding_provider,
                    parser_registry=parser_registry,
                    settings=integration_settings,
                )

                await pipeline.run(source=str(nested_repo), name=repo_name)
                (nested_repo / "helper.py").unlink()
                await pipeline.run(source=str(nested_repo), name=repo_name)

                outer_rows = await graph_client.query(
                    """
                    MATCH (p:Package {id: $id})
                    RETURN count(p) AS count
                    """,
                    id=f"{repo_name}::SampleApp",
                )
                inner_rows = await graph_client.query(
                    """
                    MATCH (p:Package {id: $id})
                    RETURN count(p) AS count
                    """,
                    id=f"{repo_name}::SampleApp.Services",
                )

                assert outer_rows[0]["count"] == 1
                assert inner_rows[0]["count"] == 1
            finally:
                await graph_client.delete_repository(repo_name)
                await graph_client.close()

        asyncio.run(run())

    def test_cleanup_repository(
        self, graph_client, embedding_provider, parser_registry,
        integration_settings, sample_repo,
    ):
        async def run():
            try:
                await graph_client.connect()

                pipeline = IndexingPipeline(
                    graph_client=graph_client,
                    embedding_provider=embedding_provider,
                    parser_registry=parser_registry,
                    settings=integration_settings,
                )
                await pipeline.run(source=str(sample_repo), name=REPO_NAME)

                # Delete repository
                await graph_client.delete_repository(REPO_NAME)

                repo = await graph_client.get_repository(REPO_NAME)
                assert repo is None
            finally:
                await graph_client.close()

        asyncio.run(run())
