"""Tests for the indexing pipeline orchestrator."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from constellation.config import Settings
from constellation.models import CodeEntity, CodeRelationship, EntityType, RelationshipType
from constellation.parsers.base import ParseResult
from constellation.parsers.registry import ParserRegistry
from constellation.indexer.pipeline import IndexingPipeline, IndexingResult


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

def _make_settings(**overrides) -> Settings:
    defaults = dict(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="test",
        embedding_batch_size=8,
        entity_batch_size=100,
        openai_api_key="test-key",
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _make_entity(
    name: str,
    entity_type: EntityType = EntityType.CLASS,
    repository: str = "test-repo",
    file_path: str = "src/Foo.py",
    line_number: int = 1,
    language: str = "python",
) -> CodeEntity:
    return CodeEntity(
        id=f"{repository}::{name}",
        name=name,
        entity_type=entity_type,
        repository=repository,
        file_path=file_path,
        line_number=line_number,
        language=language,
        signature=f"class {name}",
    )


def _make_relationship(
    source: str = "test-repo::Foo",
    target: str = "test-repo::Bar",
    rel_type: RelationshipType = RelationshipType.CONTAINS,
) -> CodeRelationship:
    return CodeRelationship(
        source_id=source,
        target_id=target,
        relationship_type=rel_type,
    )


@pytest.fixture
def settings():
    return _make_settings()


@pytest.fixture
def mock_parser():
    """A mock parser that returns a ParseResult with one entity and one relationship."""
    parser = MagicMock()
    parser.language = "python"
    parser.file_extensions = [".py"]

    def _parse_file(file_path: Path, repository: str) -> ParseResult:
        entity = _make_entity(
            name=file_path.stem.capitalize(),
            repository=repository,
            file_path=str(file_path),
        )
        rel = _make_relationship(
            source=entity.id,
            target=f"{repository}::Other",
        )
        return ParseResult(
            file_path=str(file_path),
            language="python",
            entities=[entity],
            relationships=[rel],
        )

    parser.parse_file = MagicMock(side_effect=_parse_file)
    parser.can_parse = MagicMock(return_value=True)
    return parser


@pytest.fixture
def mock_registry(mock_parser):
    registry = MagicMock(spec=ParserRegistry)
    registry.get_parser_for_file = MagicMock(return_value=mock_parser)
    registry.supported_extensions = {".py"}
    return registry


@pytest.fixture
def pipeline(mock_graph_client, mock_embedding_provider, mock_registry, settings):
    return IndexingPipeline(
        graph_client=mock_graph_client,
        embedding_provider=mock_embedding_provider,
        parser_registry=mock_registry,
        settings=settings,
    )


def _create_py_file(directory: Path, name: str = "main.py", content: str = "x = 1") -> Path:
    """Create a .py file in the given directory."""
    f = directory / name
    f.write_text(content)
    return f


def _collect_upserted_entities(mock_graph_client) -> list[CodeEntity]:
    entities: list[CodeEntity] = []
    for call in mock_graph_client.upsert_entities.call_args_list:
        entities.extend(call[0][0] if call[0] else call[1].get("entities", []))
    return entities


def _collect_created_relationships(mock_graph_client) -> list[CodeRelationship]:
    relationships: list[CodeRelationship] = []
    for call in mock_graph_client.create_relationships.call_args_list:
        relationships.extend(
            call[0][0] if call[0] else call[1].get("relationships", [])
        )
    return relationships


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_local_source_all_files_processed(
        self, pipeline, mock_graph_client, mock_embedding_provider, tmp_path
    ):
        """Local source, all files parsed, embedded, stored -- correct stats."""
        _create_py_file(tmp_path, "main.py", "class Main: pass")
        _create_py_file(tmp_path, "helper.py", "class Helper: pass")

        with patch("constellation.indexer.pipeline.get_commit_sha", return_value="abc123"):
            result = await pipeline.run(source=str(tmp_path))

        assert isinstance(result, IndexingResult)
        assert result.files_total == 2
        assert result.files_processed == 2
        assert result.files_skipped == 0
        assert len(result.errors) == 0
        # Graph client should have been called to upsert entities & relationships
        assert mock_graph_client.upsert_entities.called
        assert mock_graph_client.create_relationships.called
        assert mock_graph_client.upsert_repository.called

    @pytest.mark.asyncio
    async def test_entities_created_count(
        self, pipeline, mock_graph_client, tmp_path
    ):
        """entities_created reflects the return value from upsert_entities."""
        _create_py_file(tmp_path, "a.py")
        mock_graph_client.upsert_entities = AsyncMock(return_value=7)

        with patch("constellation.indexer.pipeline.get_commit_sha", return_value=None):
            result = await pipeline.run(source=str(tmp_path))

        assert result.entities_created == 7

    @pytest.mark.asyncio
    async def test_relationships_created_count(
        self, pipeline, mock_graph_client, tmp_path
    ):
        """relationships_created reflects the return value from create_relationships."""
        _create_py_file(tmp_path, "a.py")
        mock_graph_client.create_relationships = AsyncMock(return_value=10)

        with patch("constellation.indexer.pipeline.get_commit_sha", return_value=None):
            result = await pipeline.run(source=str(tmp_path))

        assert result.relationships_created == 10


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------


class TestChangeDetection:
    @pytest.mark.asyncio
    async def test_unchanged_files_skipped(
        self, pipeline, mock_graph_client, mock_parser, tmp_path
    ):
        """Files with matching hash are skipped; parser NOT called for those."""
        f = _create_py_file(tmp_path, "unchanged.py", "original content")

        # Simulate that Neo4j already has this file with the same hash
        from constellation.indexer.collector import compute_file_hash
        existing_hash = compute_file_hash(f)
        mock_graph_client.get_file_hashes = AsyncMock(
            return_value={"unchanged.py": existing_hash}
        )

        with patch("constellation.indexer.pipeline.get_commit_sha", return_value=None):
            result = await pipeline.run(source=str(tmp_path))

        assert result.files_skipped == 1
        assert result.files_processed == 0
        # Parser should NOT have been called for the unchanged file
        mock_parser.parse_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_changed_files_processed(
        self, pipeline, mock_graph_client, mock_parser, tmp_path
    ):
        """Files with different hash are processed."""
        _create_py_file(tmp_path, "changed.py", "new content")

        mock_graph_client.get_file_hashes = AsyncMock(
            return_value={"changed.py": "old_hash_doesnt_match"}
        )

        with patch("constellation.indexer.pipeline.get_commit_sha", return_value=None):
            result = await pipeline.run(source=str(tmp_path))

        assert result.files_processed == 1
        assert result.files_skipped == 0
        mock_parser.parse_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_changed_files_refresh_relationships_from_unchanged_files(
        self, pipeline, mock_graph_client, mock_parser, mock_registry, tmp_path
    ):
        """A changed file should trigger relationship rebuilds from unchanged files."""
        caller_file = tmp_path / "Caller.java"
        caller_file.write_text("class Caller extends Target {}")
        target_file = tmp_path / "Target.java"
        target_file.write_text("class Target {}")

        from constellation.indexer.collector import compute_file_hash

        original_target_hash = compute_file_hash(target_file)
        target_file.write_text("class Target { int version = 2; }")

        mock_parser.language = "java"
        mock_parser.file_extensions = [".java"]
        mock_registry.supported_extensions = {".java"}
        mock_graph_client.get_file_hashes = AsyncMock(
            return_value={
                "Caller.java": compute_file_hash(caller_file),
                "Target.java": original_target_hash,
            }
        )

        def _parse_java(file_path: Path, repository: str) -> ParseResult:
            file_id = f"{repository}::{file_path}"
            if file_path.name == "Caller.java":
                caller_id = f"{repository}::example.Caller"
                return ParseResult(
                    file_path=str(file_path),
                    language="java",
                    entities=[
                        CodeEntity(
                            id=file_id,
                            name=file_path.name,
                            entity_type=EntityType.FILE,
                            repository=repository,
                            file_path=str(file_path),
                            line_number=1,
                            language="java",
                        ),
                        CodeEntity(
                            id=caller_id,
                            name="Caller",
                            entity_type=EntityType.CLASS,
                            repository=repository,
                            file_path=str(file_path),
                            line_number=1,
                            language="java",
                        ),
                    ],
                    relationships=[
                        CodeRelationship(
                            source_id=file_id,
                            target_id=caller_id,
                            relationship_type=RelationshipType.CONTAINS,
                        ),
                        CodeRelationship(
                            source_id=caller_id,
                            target_id=f"{repository}::example.Target",
                            relationship_type=RelationshipType.EXTENDS,
                        ),
                    ],
                )

            target_id = f"{repository}::example.Target"
            return ParseResult(
                file_path=str(file_path),
                language="java",
                entities=[
                    CodeEntity(
                        id=file_id,
                        name=file_path.name,
                        entity_type=EntityType.FILE,
                        repository=repository,
                        file_path=str(file_path),
                        line_number=1,
                        language="java",
                    ),
                    CodeEntity(
                        id=target_id,
                        name="Target",
                        entity_type=EntityType.CLASS,
                        repository=repository,
                        file_path=str(file_path),
                        line_number=1,
                        language="java",
                    ),
                ],
                relationships=[
                    CodeRelationship(
                        source_id=file_id,
                        target_id=target_id,
                        relationship_type=RelationshipType.CONTAINS,
                    )
                ],
            )

        mock_parser.parse_file = MagicMock(side_effect=_parse_java)

        with patch("constellation.indexer.pipeline.get_commit_sha", return_value=None):
            result = await pipeline.run(source=str(tmp_path), name="test-repo")

        assert result.files_processed == 1
        assert result.files_skipped == 1
        assert mock_parser.parse_file.call_count == 2

        relationships = _collect_created_relationships(mock_graph_client)
        assert any(
            relationship.source_id == "test-repo::example.Caller"
            and relationship.target_id == "test-repo::example.Target"
            and relationship.relationship_type == RelationshipType.EXTENDS
            for relationship in relationships
        )


# ---------------------------------------------------------------------------
# Reindex mode
# ---------------------------------------------------------------------------


class TestReindexMode:
    @pytest.mark.asyncio
    async def test_reindex_processes_all_files(
        self, pipeline, mock_graph_client, mock_parser, tmp_path
    ):
        """All files processed regardless of hash when reindex=True."""
        f = _create_py_file(tmp_path, "existing.py", "same content")

        from constellation.indexer.collector import compute_file_hash
        existing_hash = compute_file_hash(f)
        mock_graph_client.get_file_hashes = AsyncMock(
            return_value={"existing.py": existing_hash}
        )

        with patch("constellation.indexer.pipeline.get_commit_sha", return_value=None):
            result = await pipeline.run(source=str(tmp_path), reindex=True)

        assert result.files_processed == 1
        assert result.files_skipped == 0
        mock_parser.parse_file.assert_called_once()


# ---------------------------------------------------------------------------
# GitHub URL source
# ---------------------------------------------------------------------------


class TestGithubSource:
    @pytest.mark.asyncio
    async def test_github_url_clones_and_cleans_up(
        self, pipeline, mock_graph_client, tmp_path
    ):
        """Calls clone_repository, uses cloned path, calls cleanup_clone."""
        _create_py_file(tmp_path, "cloned.py")

        with patch("constellation.indexer.pipeline.clone_repository", return_value=tmp_path) as mock_clone, \
             patch("constellation.indexer.pipeline.cleanup_clone") as mock_cleanup, \
             patch("constellation.indexer.pipeline.get_commit_sha", return_value="sha123"), \
             patch("constellation.indexer.pipeline.is_github_url", return_value=True):
            result = await pipeline.run(source="https://github.com/user/repo")

        mock_clone.assert_called_once_with("https://github.com/user/repo")
        mock_cleanup.assert_called_once_with(tmp_path)
        assert result.repository is not None

    @pytest.mark.asyncio
    async def test_github_url_change_detection_uses_relative_paths(
        self, pipeline, mock_graph_client, mock_parser, tmp_path
    ):
        """Remote reindexing should compare hashes using repo-relative paths."""
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        f = _create_py_file(source_dir, "cloned.py")

        from constellation.indexer.collector import compute_file_hash

        mock_graph_client.get_file_hashes = AsyncMock(
            return_value={"src/cloned.py": compute_file_hash(f)}
        )

        with patch("constellation.indexer.pipeline.clone_repository", return_value=tmp_path), \
             patch("constellation.indexer.pipeline.cleanup_clone"), \
             patch("constellation.indexer.pipeline.get_commit_sha", return_value="sha123"), \
             patch("constellation.indexer.pipeline.is_github_url", return_value=True):
            result = await pipeline.run(source="https://github.com/user/repo")

        assert result.files_skipped == 1
        assert result.files_processed == 0
        mock_parser.parse_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_github_url_derives_repo_name(
        self, pipeline, tmp_path
    ):
        """Repo name derived from GitHub URL."""
        _create_py_file(tmp_path, "app.py")

        with patch("constellation.indexer.pipeline.clone_repository", return_value=tmp_path), \
             patch("constellation.indexer.pipeline.cleanup_clone"), \
             patch("constellation.indexer.pipeline.get_commit_sha", return_value="sha"), \
             patch("constellation.indexer.pipeline.is_github_url", return_value=True), \
             patch("constellation.indexer.pipeline.derive_repo_name", return_value="user/repo") as mock_derive:
            result = await pipeline.run(source="https://github.com/user/repo")

        assert result.repository == "user/repo"


# ---------------------------------------------------------------------------
# Parse error handling
# ---------------------------------------------------------------------------


class TestParseErrorHandling:
    @pytest.mark.asyncio
    async def test_parse_result_with_errors(
        self, pipeline, mock_parser, tmp_path
    ):
        """Parser returns ParseResult with errors: error added, pipeline continues."""
        _create_py_file(tmp_path, "bad.py", "syntax error here")

        def _error_parse(file_path, repository):
            return ParseResult(
                file_path=str(file_path),
                language="python",
                errors=["Syntax error at line 1"],
            )

        mock_parser.parse_file = MagicMock(side_effect=_error_parse)

        with patch("constellation.indexer.pipeline.get_commit_sha", return_value=None):
            result = await pipeline.run(source=str(tmp_path))

        assert len(result.errors) >= 1
        assert any("bad.py" in e or "Syntax error" in e for e in result.errors)
        pipeline._graph.upsert_entities.assert_not_awaited()
        pipeline._graph.prepare_file_reindex.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_parse_exception(
        self, pipeline, mock_parser, tmp_path
    ):
        """Parser throws exception: error added, pipeline continues."""
        _create_py_file(tmp_path, "crash.py", "x = 1")
        _create_py_file(tmp_path, "good.py", "y = 2")

        call_count = 0

        def _crash_then_ok(file_path, repository):
            nonlocal call_count
            call_count += 1
            if "crash" in str(file_path):
                raise RuntimeError("Parser crashed!")
            return ParseResult(
                file_path=str(file_path),
                language="python",
                entities=[_make_entity(name="Good", repository=repository, file_path=str(file_path))],
                relationships=[],
            )

        mock_parser.parse_file = MagicMock(side_effect=_crash_then_ok)

        with patch("constellation.indexer.pipeline.get_commit_sha", return_value=None):
            result = await pipeline.run(source=str(tmp_path))

        # The error should be recorded
        assert len(result.errors) >= 1
        assert any("crash" in e.lower() or "Parser crashed" in e for e in result.errors)
        # Good file should still have been processed
        assert result.files_processed >= 1


# ---------------------------------------------------------------------------
# Embedding failure
# ---------------------------------------------------------------------------


class TestEmbeddingFailure:
    @pytest.mark.asyncio
    async def test_embedding_error_entities_still_stored(
        self, pipeline, mock_graph_client, mock_embedding_provider, mock_parser, tmp_path
    ):
        """embed_batch raises exception: error logged, entities still stored without embeddings."""
        _create_py_file(tmp_path, "embed_fail.py")

        # Make the entity embeddable (CLASS type is embeddable)
        def _parse_embeddable(file_path, repository):
            return ParseResult(
                file_path=str(file_path),
                language="python",
                entities=[_make_entity(
                    name="EmbedMe",
                    entity_type=EntityType.CLASS,
                    repository=repository,
                    file_path=str(file_path),
                )],
                relationships=[],
            )

        mock_parser.parse_file = MagicMock(side_effect=_parse_embeddable)
        mock_embedding_provider.embed_batch = AsyncMock(
            side_effect=RuntimeError("Embedding API down")
        )

        with patch("constellation.indexer.pipeline.get_commit_sha", return_value=None):
            result = await pipeline.run(source=str(tmp_path))

        # Entities should still be upserted even without embeddings
        assert mock_graph_client.upsert_entities.called
        # Error should be recorded
        assert len(result.errors) >= 1
        assert any("embed" in e.lower() or "Embedding" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Stale file deletion
# ---------------------------------------------------------------------------


class TestStaleFileDeletion:
    @pytest.mark.asyncio
    async def test_stale_files_deleted(
        self, pipeline, mock_graph_client, tmp_path
    ):
        """Files in Neo4j but not in current discovery set get deleted."""
        _create_py_file(tmp_path, "current.py")

        # Neo4j has an extra file that no longer exists
        mock_graph_client.get_file_hashes = AsyncMock(
            return_value={
                "current.py": "some_hash",
                "deleted.py": "old_hash",
            }
        )

        with patch("constellation.indexer.pipeline.get_commit_sha", return_value=None):
            result = await pipeline.run(source=str(tmp_path))

        mock_graph_client.delete_stale_files.assert_called_once()
        call_args = mock_graph_client.delete_stale_files.call_args
        # Extract file_paths from keyword args
        stale_paths = call_args[1].get("file_paths", [])
        assert "deleted.py" in stale_paths


# ---------------------------------------------------------------------------
# Custom exclude patterns
# ---------------------------------------------------------------------------


class TestCustomExcludePatterns:
    @pytest.mark.asyncio
    async def test_custom_patterns_merged_with_defaults(
        self, pipeline, tmp_path
    ):
        """Custom exclude patterns are merged with defaults."""
        # Create files
        custom_dir = tmp_path / "custom_exclude"
        custom_dir.mkdir()
        _create_py_file(custom_dir, "excluded.py")
        _create_py_file(tmp_path, "included.py")

        with patch("constellation.indexer.pipeline.get_commit_sha", return_value=None):
            result = await pipeline.run(
                source=str(tmp_path),
                exclude_patterns=["custom_exclude"],
            )

        assert result.files_total == 1  # Only included.py


# ---------------------------------------------------------------------------
# Repository name derivation
# ---------------------------------------------------------------------------


class TestRepoNameDerivation:
    @pytest.mark.asyncio
    async def test_name_derived_from_local_path(
        self, pipeline, tmp_path
    ):
        """Repo name derived from local path when not provided."""
        _create_py_file(tmp_path, "a.py")

        with patch("constellation.indexer.pipeline.get_commit_sha", return_value=None):
            result = await pipeline.run(source=str(tmp_path))

        assert result.repository == tmp_path.name

    @pytest.mark.asyncio
    async def test_explicit_name_used(
        self, pipeline, tmp_path
    ):
        """Explicit name takes precedence."""
        _create_py_file(tmp_path, "a.py")

        with patch("constellation.indexer.pipeline.get_commit_sha", return_value=None):
            result = await pipeline.run(source=str(tmp_path), name="custom-repo")

        assert result.repository == "custom-repo"


# ---------------------------------------------------------------------------
# Repository node update
# ---------------------------------------------------------------------------


class TestRepositoryNodeUpdate:
    @pytest.mark.asyncio
    async def test_upsert_repository_called_with_correct_args(
        self, pipeline, mock_graph_client, tmp_path
    ):
        """upsert_repository called with correct stats."""
        _create_py_file(tmp_path, "a.py")

        with patch("constellation.indexer.pipeline.get_commit_sha", return_value="sha456"):
            result = await pipeline.run(source=str(tmp_path), name="my-repo")

        mock_graph_client.upsert_repository.assert_called_once()
        call_kwargs = mock_graph_client.upsert_repository.call_args
        # Check positional or keyword arguments
        assert "my-repo" in (call_kwargs[0] if call_kwargs[0] else ()) or \
               call_kwargs[1].get("name") == "my-repo"

    @pytest.mark.asyncio
    async def test_repository_entity_count_comes_from_graph(
        self, pipeline, mock_graph_client, tmp_path
    ):
        """Repository metadata should use the graph count after indexing."""
        _create_py_file(tmp_path, "a.py")
        mock_graph_client.count_repository_entities = AsyncMock(return_value=42)

        with patch("constellation.indexer.pipeline.get_commit_sha", return_value="sha456"):
            await pipeline.run(source=str(tmp_path), name="my-repo")

        mock_graph_client.count_repository_entities.assert_awaited_once_with("my-repo")
        _, kwargs = mock_graph_client.upsert_repository.call_args
        assert kwargs["entity_count"] == 42


# ---------------------------------------------------------------------------
# Commit SHA
# ---------------------------------------------------------------------------


class TestCommitSha:
    @pytest.mark.asyncio
    async def test_commit_sha_from_get_commit_sha(
        self, pipeline, mock_graph_client, tmp_path
    ):
        """Commit SHA is obtained from get_commit_sha."""
        _create_py_file(tmp_path, "a.py")

        with patch("constellation.indexer.pipeline.get_commit_sha", return_value="deadbeef") as mock_sha:
            result = await pipeline.run(source=str(tmp_path), name="repo")

        mock_sha.assert_called_once()
        # Verify SHA was passed to upsert_repository
        mock_graph_client.upsert_repository.assert_called_once()
        call_args = mock_graph_client.upsert_repository.call_args
        # SHA should be present in the call
        all_args = list(call_args[0]) + list(call_args[1].values())
        assert "deadbeef" in all_args


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------


class TestProgressCallback:
    @pytest.mark.asyncio
    async def test_progress_callback_called(
        self, pipeline, tmp_path
    ):
        """Progress callback called with correct totals."""
        _create_py_file(tmp_path, "a.py")
        _create_py_file(tmp_path, "b.py")

        callback = MagicMock()

        with patch("constellation.indexer.pipeline.get_commit_sha", return_value=None):
            result = await pipeline.run(
                source=str(tmp_path),
                progress_callback=callback,
            )

        assert callback.called
        # The callback should have been called with files_total, files_processed, entities_found
        for c in callback.call_args_list:
            args = c[1] if c[1] else {}
            pos_args = c[0] if c[0] else ()
            # At least one call should have the total
            assert len(pos_args) >= 1 or len(args) >= 1

    @pytest.mark.asyncio
    async def test_progress_reports_processed_files_not_examined(
        self, pipeline, mock_graph_client, tmp_path
    ):
        """Relationship refresh should not inflate files_processed progress."""
        unchanged = _create_py_file(tmp_path, "unchanged.py", "x = 1")
        _create_py_file(tmp_path, "changed.py", "x = 2")

        from constellation.indexer.collector import compute_file_hash

        mock_graph_client.get_file_hashes = AsyncMock(
            return_value={
                "unchanged.py": compute_file_hash(unchanged),
                "changed.py": "outdated-hash",
            }
        )
        callback = MagicMock()

        with patch("constellation.indexer.pipeline.get_commit_sha", return_value=None):
            result = await pipeline.run(
                source=str(tmp_path),
                progress_callback=callback,
            )

        assert result.files_processed == 1
        assert result.files_skipped == 1
        last_args = callback.call_args_list[-1].args
        assert last_args[0] == 2
        assert last_args[1] == 1


# ---------------------------------------------------------------------------
# Empty directory
# ---------------------------------------------------------------------------


class TestEmptyDirectory:
    @pytest.mark.asyncio
    async def test_empty_directory(self, pipeline, tmp_path):
        """0 files processed, no errors."""
        with patch("constellation.indexer.pipeline.get_commit_sha", return_value=None):
            result = await pipeline.run(source=str(tmp_path))

        assert result.files_total == 0
        assert result.files_processed == 0
        assert len(result.errors) == 0


# ---------------------------------------------------------------------------
# File entity creation
# ---------------------------------------------------------------------------


class TestFileEntityCreation:
    @pytest.mark.asyncio
    async def test_file_entity_created_with_content_hash(
        self, pipeline, mock_graph_client, tmp_path
    ):
        """Each processed file gets a FILE entity with content_hash."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        _create_py_file(src_dir, "module.py", "x = 1")

        with patch("constellation.indexer.pipeline.get_commit_sha", return_value=None):
            result = await pipeline.run(source=str(tmp_path), name="test-repo")

        # Inspect all entities passed to upsert_entities
        assert mock_graph_client.upsert_entities.called
        all_entities = []
        for c in mock_graph_client.upsert_entities.call_args_list:
            all_entities.extend(c[0][0] if c[0] else c[1].get("entities", []))

        file_entities = [e for e in all_entities if e.entity_type == EntityType.FILE]
        assert len(file_entities) == 1
        for fe in file_entities:
            assert fe.content_hash is not None
            assert len(fe.content_hash) == 32  # MD5 hex digest
            assert fe.id == "test-repo::src/module.py"
            assert fe.file_path == "src/module.py"

        non_file_entities = [e for e in all_entities if e.entity_type != EntityType.FILE]
        assert non_file_entities
        assert all(e.file_path == "src/module.py" for e in non_file_entities)


class TestScopedEntityNormalization:
    @pytest.mark.asyncio
    async def test_python_same_stem_files_get_distinct_entity_ids(
        self, pipeline, mock_graph_client, mock_parser, tmp_path
    ):
        """Python entities should be scoped to the repo-relative file path."""
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        _create_py_file(tmp_path / "a", "utils.py", "class Foo: pass")
        _create_py_file(tmp_path / "b", "utils.py", "class Foo: pass")

        def _parse_python(file_path: Path, repository: str) -> ParseResult:
            parser_file_id = f"{repository}::{file_path}"
            class_id = f"{repository}::utils.Foo"
            func_id = f"{repository}::utils.build"
            return ParseResult(
                file_path=str(file_path),
                language="python",
                entities=[
                    CodeEntity(
                        id=parser_file_id,
                        name=file_path.name,
                        entity_type=EntityType.FILE,
                        repository=repository,
                        file_path=str(file_path),
                        line_number=1,
                        language="python",
                    ),
                    CodeEntity(
                        id=class_id,
                        name="Foo",
                        entity_type=EntityType.CLASS,
                        repository=repository,
                        file_path=str(file_path),
                        line_number=1,
                        language="python",
                    ),
                    CodeEntity(
                        id=func_id,
                        name="build",
                        entity_type=EntityType.METHOD,
                        repository=repository,
                        file_path=str(file_path),
                        line_number=3,
                        language="python",
                    ),
                ],
                relationships=[
                    CodeRelationship(
                        source_id=parser_file_id,
                        target_id=class_id,
                        relationship_type=RelationshipType.CONTAINS,
                    ),
                    CodeRelationship(
                        source_id=parser_file_id,
                        target_id=func_id,
                        relationship_type=RelationshipType.CONTAINS,
                    ),
                    CodeRelationship(
                        source_id=func_id,
                        target_id=f"{repository}::Foo",
                        relationship_type=RelationshipType.CALLS,
                    ),
                ],
            )

        mock_parser.parse_file = MagicMock(side_effect=_parse_python)

        with patch("constellation.indexer.pipeline.get_commit_sha", return_value=None):
            await pipeline.run(source=str(tmp_path), name="test-repo")

        all_entities = _collect_upserted_entities(mock_graph_client)
        class_ids = {
            entity.id
            for entity in all_entities
            if entity.entity_type == EntityType.CLASS
        }
        assert class_ids == {
            "test-repo::a/utils.py#Foo",
            "test-repo::b/utils.py#Foo",
        }

        call_edges = {
            (relationship.source_id, relationship.target_id)
            for relationship in _collect_created_relationships(mock_graph_client)
            if relationship.relationship_type == RelationshipType.CALLS
        }
        assert call_edges == {
            ("test-repo::a/utils.py#build", "test-repo::a/utils.py#Foo"),
            ("test-repo::b/utils.py#build", "test-repo::b/utils.py#Foo"),
        }

    @pytest.mark.asyncio
    async def test_javascript_same_stem_files_get_distinct_entity_ids(
        self, pipeline, mock_graph_client, mock_parser, mock_registry, tmp_path
    ):
        """JavaScript/TypeScript entities should also be scoped to the file path."""
        mock_parser.language = "javascript"
        mock_parser.file_extensions = [".ts"]
        mock_registry.supported_extensions = {".ts"}

        (tmp_path / "x").mkdir()
        (tmp_path / "y").mkdir()
        js_x = tmp_path / "x" / "utils.ts"
        js_y = tmp_path / "y" / "utils.ts"
        js_x.write_text("export class Widget {}")
        js_y.write_text("export class Widget {}")

        def _parse_javascript(file_path: Path, repository: str) -> ParseResult:
            parser_file_id = f"{repository}::utils"
            class_id = f"{repository}::utils.Widget"
            method_id = f"{repository}::utils.Widget.run"
            return ParseResult(
                file_path=str(file_path),
                language="javascript",
                entities=[
                    CodeEntity(
                        id=parser_file_id,
                        name=file_path.name,
                        entity_type=EntityType.FILE,
                        repository=repository,
                        file_path=str(file_path),
                        line_number=1,
                        language="javascript",
                    ),
                    CodeEntity(
                        id=class_id,
                        name="Widget",
                        entity_type=EntityType.CLASS,
                        repository=repository,
                        file_path=str(file_path),
                        line_number=1,
                        language="javascript",
                    ),
                    CodeEntity(
                        id=method_id,
                        name="run",
                        entity_type=EntityType.METHOD,
                        repository=repository,
                        file_path=str(file_path),
                        line_number=2,
                        language="javascript",
                    ),
                ],
                relationships=[
                    CodeRelationship(
                        source_id=parser_file_id,
                        target_id=class_id,
                        relationship_type=RelationshipType.CONTAINS,
                    ),
                    CodeRelationship(
                        source_id=class_id,
                        target_id=method_id,
                        relationship_type=RelationshipType.HAS_METHOD,
                    ),
                ],
            )

        mock_parser.parse_file = MagicMock(side_effect=_parse_javascript)

        with patch("constellation.indexer.pipeline.get_commit_sha", return_value=None):
            await pipeline.run(source=str(tmp_path), name="test-repo")

        all_entities = _collect_upserted_entities(mock_graph_client)
        scoped_ids = {
            entity.id
            for entity in all_entities
            if entity.entity_type in {EntityType.CLASS, EntityType.METHOD}
        }
        assert scoped_ids == {
            "test-repo::x/utils.ts#Widget",
            "test-repo::x/utils.ts#Widget.run",
            "test-repo::y/utils.ts#Widget",
            "test-repo::y/utils.ts#Widget.run",
        }

    @pytest.mark.asyncio
    async def test_javascript_hooks_are_materialized_as_entities(
        self, pipeline, mock_graph_client, mock_parser, mock_registry, tmp_path
    ):
        """USES_HOOK targets should also be upserted as graph entities."""
        mock_parser.language = "javascript"
        mock_parser.file_extensions = [".tsx"]
        mock_registry.supported_extensions = {".tsx"}

        source_file = tmp_path / "component.tsx"
        source_file.write_text("export const App = () => null;")

        def _parse_javascript(file_path: Path, repository: str) -> ParseResult:
            parser_file_id = f"{repository}::component"
            component_id = f"{repository}::component.App"
            hook_id = "hook:useState"
            return ParseResult(
                file_path=str(file_path),
                language="javascript",
                entities=[
                    CodeEntity(
                        id=parser_file_id,
                        name=file_path.name,
                        entity_type=EntityType.FILE,
                        repository=repository,
                        file_path=str(file_path),
                        line_number=1,
                        language="javascript",
                    ),
                    CodeEntity(
                        id=component_id,
                        name="App",
                        entity_type=EntityType.METHOD,
                        repository=repository,
                        file_path=str(file_path),
                        line_number=1,
                        language="javascript",
                    ),
                    CodeEntity(
                        id=hook_id,
                        name="useState",
                        entity_type=EntityType.HOOK,
                        repository=repository,
                        file_path=str(file_path),
                        line_number=2,
                        language="javascript",
                    ),
                ],
                relationships=[
                    CodeRelationship(
                        source_id=parser_file_id,
                        target_id=component_id,
                        relationship_type=RelationshipType.CONTAINS,
                    ),
                    CodeRelationship(
                        source_id=component_id,
                        target_id=hook_id,
                        relationship_type=RelationshipType.USES_HOOK,
                    ),
                ],
            )

        mock_parser.parse_file = MagicMock(side_effect=_parse_javascript)

        with patch("constellation.indexer.pipeline.get_commit_sha", return_value=None):
            await pipeline.run(source=str(tmp_path), name="test-repo")

        all_entities = _collect_upserted_entities(mock_graph_client)
        hook_ids = {
            entity.id
            for entity in all_entities
            if entity.entity_type == EntityType.HOOK
        }
        assert hook_ids == {"test-repo::component.tsx#useState"}

    @pytest.mark.asyncio
    async def test_changed_file_prepares_reindex_and_preserves_current_ids(
        self, pipeline, mock_graph_client, mock_parser, tmp_path
    ):
        """Changed files should reconcile stale nodes before upserting new ones."""
        changed_file = _create_py_file(tmp_path, "changed.py", "class UserService: pass")

        mock_graph_client.get_file_hashes = AsyncMock(
            return_value={"changed.py": "old_hash_doesnt_match"}
        )

        def _parse_python(file_path: Path, repository: str) -> ParseResult:
            parser_file_id = f"{repository}::{file_path}"
            class_id = f"{repository}::changed.UserService"
            return ParseResult(
                file_path=str(file_path),
                language="python",
                entities=[
                    CodeEntity(
                        id=parser_file_id,
                        name=file_path.name,
                        entity_type=EntityType.FILE,
                        repository=repository,
                        file_path=str(file_path),
                        line_number=1,
                        language="python",
                    ),
                    CodeEntity(
                        id=class_id,
                        name="UserService",
                        entity_type=EntityType.CLASS,
                        repository=repository,
                        file_path=str(file_path),
                        line_number=1,
                        language="python",
                    ),
                ],
                relationships=[
                    CodeRelationship(
                        source_id=parser_file_id,
                        target_id=class_id,
                        relationship_type=RelationshipType.CONTAINS,
                    )
                ],
            )

        mock_parser.parse_file = MagicMock(side_effect=_parse_python)

        with patch("constellation.indexer.pipeline.get_commit_sha", return_value=None):
            await pipeline.run(source=str(tmp_path), name="test-repo")

        mock_graph_client.prepare_file_reindex.assert_awaited_once()
        _, kwargs = mock_graph_client.prepare_file_reindex.await_args
        assert kwargs["repository"] == "test-repo"
        assert kwargs["file_path"] == "changed.py"
        assert kwargs["current_entity_ids"] == {
            "test-repo::changed.py",
            "test-repo::changed.py#UserService",
        }


# ---------------------------------------------------------------------------
# Clone cleanup on error
# ---------------------------------------------------------------------------


class TestCloneCleanupOnError:
    @pytest.mark.asyncio
    async def test_clone_cleaned_up_on_error(
        self, pipeline, mock_graph_client, tmp_path
    ):
        """If indexing fails, clone dir is still cleaned up."""
        # Make upsert_entities raise to simulate an error during indexing
        mock_graph_client.upsert_entities = AsyncMock(
            side_effect=RuntimeError("DB connection lost")
        )
        _create_py_file(tmp_path, "file.py")

        with patch("constellation.indexer.pipeline.clone_repository", return_value=tmp_path) as mock_clone, \
             patch("constellation.indexer.pipeline.cleanup_clone") as mock_cleanup, \
             patch("constellation.indexer.pipeline.get_commit_sha", return_value="sha"), \
             patch("constellation.indexer.pipeline.is_github_url", return_value=True):
            # The pipeline should either handle the error or re-raise;
            # but cleanup must happen in either case.
            try:
                await pipeline.run(source="https://github.com/user/repo")
            except RuntimeError:
                pass  # Expected

        mock_cleanup.assert_called_once_with(tmp_path)
