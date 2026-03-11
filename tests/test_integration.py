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

                assert result.files_processed > 0
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
            finally:
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
            finally:
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
