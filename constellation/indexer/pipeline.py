"""Indexing pipeline orchestrator.

Coordinates file collection, parsing, embedding, and graph storage
into a single end-to-end indexing workflow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from constellation.config import Settings
from constellation.embeddings.base import BaseEmbeddingProvider, is_embeddable, prepare_embedding_text
from constellation.indexer.cloner import clone_repository, cleanup_clone, get_commit_sha
from constellation.indexer.collector import (
    collect_files,
    compute_file_hash,
    derive_repo_name,
    is_github_url,
    DEFAULT_EXCLUSIONS,
)
from constellation.models import CodeEntity, CodeRelationship, EntityType
from constellation.parsers.base import ParseResult
from constellation.parsers.registry import ParserRegistry

logger = logging.getLogger(__name__)


@dataclass
class IndexingResult:
    """Summary of an indexing run."""

    repository: str
    files_total: int
    files_processed: int
    files_skipped: int
    entities_created: int
    relationships_created: int
    errors: list[str] = field(default_factory=list)


class IndexingPipeline:
    """Orchestrates the full indexing pipeline.

    Parameters
    ----------
    graph_client:
        Async Neo4j graph client for storing entities/relationships.
    embedding_provider:
        Provider for generating vector embeddings.
    parser_registry:
        Registry mapping file extensions to language parsers.
    settings:
        Application settings (batch sizes, etc.).
    """

    def __init__(
        self,
        graph_client,
        embedding_provider: BaseEmbeddingProvider,
        parser_registry: ParserRegistry,
        settings: Settings,
    ) -> None:
        self._graph = graph_client
        self._embedder = embedding_provider
        self._registry = parser_registry
        self._settings = settings

    async def run(
        self,
        source: str,
        name: str | None = None,
        exclude_patterns: list[str] | None = None,
        reindex: bool = False,
        progress_callback: Callable | None = None,
    ) -> IndexingResult:
        """Execute the full indexing pipeline.

        Parameters
        ----------
        source:
            Local filesystem path or GitHub URL.
        name:
            Repository name. Derived from *source* when ``None``.
        exclude_patterns:
            Extra patterns merged with ``DEFAULT_EXCLUSIONS``.
        reindex:
            When ``True``, skip change detection and reprocess all files.
        progress_callback:
            Optional callable invoked as ``cb(files_total, files_processed, entities_found)``.

        Returns
        -------
        IndexingResult
        """
        cloned_path: Path | None = None
        repo_name = name or derive_repo_name(source)

        try:
            # ----------------------------------------------------------
            # 1. Determine source path
            # ----------------------------------------------------------
            if is_github_url(source):
                cloned_path = clone_repository(source)
                source_path = cloned_path
            else:
                source_path = Path(source)

            # ----------------------------------------------------------
            # 2. Commit SHA
            # ----------------------------------------------------------
            commit_sha = get_commit_sha(source_path)

            # ----------------------------------------------------------
            # 3. Build exclusion set
            # ----------------------------------------------------------
            if exclude_patterns:
                merged_exclusions = frozenset(
                    set(DEFAULT_EXCLUSIONS) | set(exclude_patterns)
                )
            else:
                merged_exclusions = DEFAULT_EXCLUSIONS

            # ----------------------------------------------------------
            # 4. Collect files
            # ----------------------------------------------------------
            discovered_files = collect_files(
                root=source_path,
                supported_extensions=self._registry.supported_extensions,
                exclude_patterns=merged_exclusions,
            )
            files_total = len(discovered_files)

            # ----------------------------------------------------------
            # 5. Change detection
            # ----------------------------------------------------------
            existing_hashes = await self._graph.get_file_hashes(repo_name)

            # Compute hashes and decide which files need processing
            files_to_process: list[tuple[Path, str]] = []  # (path, hash)
            files_skipped = 0
            discovered_paths_str: set[str] = set()

            for fpath in discovered_files:
                file_hash = compute_file_hash(fpath)
                fpath_str = str(fpath)
                discovered_paths_str.add(fpath_str)

                if not reindex and existing_hashes.get(fpath_str) == file_hash:
                    files_skipped += 1
                    continue

                files_to_process.append((fpath, file_hash))

            # ----------------------------------------------------------
            # 6. Parse files and collect entities/relationships
            # ----------------------------------------------------------
            all_entities: list[CodeEntity] = []
            all_relationships: list[CodeRelationship] = []
            errors: list[str] = []
            files_processed = 0

            for fpath, file_hash in files_to_process:
                # Create FILE entity
                file_entity = CodeEntity(
                    id=f"{repo_name}::{fpath}",
                    name=fpath.name,
                    entity_type=EntityType.FILE,
                    repository=repo_name,
                    file_path=str(fpath),
                    line_number=0,
                    language=fpath.suffix.lstrip(".") or "unknown",
                    content_hash=file_hash,
                )
                all_entities.append(file_entity)

                # Get parser and parse
                parser = self._registry.get_parser_for_file(fpath)
                if parser is None:
                    files_processed += 1
                    if progress_callback:
                        progress_callback(files_total, files_processed, len(all_entities))
                    continue

                try:
                    parse_result: ParseResult = parser.parse_file(fpath, repo_name)
                except Exception as exc:
                    err_msg = f"Exception parsing {fpath}: {exc}"
                    logger.error(err_msg)
                    errors.append(err_msg)
                    files_processed += 1
                    if progress_callback:
                        progress_callback(files_total, files_processed, len(all_entities))
                    continue

                if parse_result.errors:
                    for pe in parse_result.errors:
                        err_msg = f"Parse error in {fpath}: {pe}"
                        logger.warning(err_msg)
                        errors.append(err_msg)
                    # Still continue processing other files
                    files_processed += 1
                    if progress_callback:
                        progress_callback(files_total, files_processed, len(all_entities))
                    continue

                all_entities.extend(parse_result.entities)
                all_relationships.extend(parse_result.relationships)
                files_processed += 1

                if progress_callback:
                    progress_callback(files_total, files_processed, len(all_entities))

            # ----------------------------------------------------------
            # 7. Embed entities
            # ----------------------------------------------------------
            try:
                await self._embed_entities(all_entities)
            except Exception as exc:
                err_msg = f"Embedding failed: {exc}"
                logger.error(err_msg)
                errors.append(err_msg)

            # ----------------------------------------------------------
            # 8. Upsert entities to graph (batched)
            # ----------------------------------------------------------
            entities_created = 0
            for i in range(0, len(all_entities), self._settings.entity_batch_size):
                batch = all_entities[i : i + self._settings.entity_batch_size]
                entities_created += await self._graph.upsert_entities(batch)

            # ----------------------------------------------------------
            # 9. Create relationships
            # ----------------------------------------------------------
            relationships_created = await self._graph.create_relationships(
                all_relationships
            )

            # ----------------------------------------------------------
            # 10. Delete stale files
            # ----------------------------------------------------------
            stale_paths = [
                p for p in existing_hashes
                if p not in discovered_paths_str
            ]
            if stale_paths:
                await self._graph.delete_stale_files(
                    repository=repo_name,
                    file_paths=stale_paths,
                )

            # ----------------------------------------------------------
            # 11. Upsert repository metadata
            # ----------------------------------------------------------
            await self._graph.upsert_repository(
                name=repo_name,
                source=source,
                commit_sha=commit_sha,
                entity_count=entities_created,
            )

            return IndexingResult(
                repository=repo_name,
                files_total=files_total,
                files_processed=files_processed,
                files_skipped=files_skipped,
                entities_created=entities_created,
                relationships_created=relationships_created,
                errors=errors,
            )
        finally:
            # ----------------------------------------------------------
            # 12. Cleanup clone directory (always, even on error)
            # ----------------------------------------------------------
            if cloned_path is not None:
                cleanup_clone(cloned_path)

    async def _embed_entities(self, entities: list[CodeEntity]) -> None:
        """Generate embeddings for embeddable entities in batches."""
        embeddable = [e for e in entities if is_embeddable(e.entity_type)]
        if not embeddable:
            return

        batch_size = self._settings.embedding_batch_size
        for i in range(0, len(embeddable), batch_size):
            batch = embeddable[i : i + batch_size]
            texts = [prepare_embedding_text(e) for e in batch]
            vectors = await self._embedder.embed_batch(texts)
            for entity, vector in zip(batch, vectors):
                entity.embedding = vector
