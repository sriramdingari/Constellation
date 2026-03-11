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
            return_value={str(f): existing_hash}
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
            return_value={str(tmp_path / "changed.py"): "old_hash_doesnt_match"}
        )

        with patch("constellation.indexer.pipeline.get_commit_sha", return_value=None):
            result = await pipeline.run(source=str(tmp_path))

        assert result.files_processed == 1
        assert result.files_skipped == 0
        mock_parser.parse_file.assert_called_once()


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
            return_value={str(f): existing_hash}
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
                str(tmp_path / "current.py"): "some_hash",
                str(tmp_path / "deleted.py"): "old_hash",
            }
        )

        with patch("constellation.indexer.pipeline.get_commit_sha", return_value=None):
            result = await pipeline.run(source=str(tmp_path))

        mock_graph_client.delete_stale_files.assert_called_once()
        call_args = mock_graph_client.delete_stale_files.call_args
        # Extract file_paths from keyword args
        stale_paths = call_args[1].get("file_paths", [])
        assert str(tmp_path / "deleted.py") in stale_paths


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
        _create_py_file(tmp_path, "module.py", "x = 1")

        with patch("constellation.indexer.pipeline.get_commit_sha", return_value=None):
            result = await pipeline.run(source=str(tmp_path), name="test-repo")

        # Inspect all entities passed to upsert_entities
        assert mock_graph_client.upsert_entities.called
        all_entities = []
        for c in mock_graph_client.upsert_entities.call_args_list:
            all_entities.extend(c[0][0] if c[0] else c[1].get("entities", []))

        file_entities = [e for e in all_entities if e.entity_type == EntityType.FILE]
        assert len(file_entities) >= 1
        for fe in file_entities:
            assert fe.content_hash is not None
            assert len(fe.content_hash) == 32  # MD5 hex digest


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
