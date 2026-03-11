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
from constellation.models import CodeEntity, CodeRelationship, EntityType, RelationshipType
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
            file_plans: list[tuple[Path, str, str, bool]] = []
            files_skipped = 0
            discovered_paths_str: set[str] = set()

            for fpath in discovered_files:
                relative_path = self._relative_file_path(source_path, fpath)
                file_hash = compute_file_hash(fpath)
                discovered_paths_str.add(relative_path)
                needs_reindex = reindex or existing_hashes.get(relative_path) != file_hash

                if not needs_reindex:
                    files_skipped += 1

                file_plans.append((fpath, relative_path, file_hash, needs_reindex))

            stale_paths = [
                file_path
                for file_path in existing_hashes
                if file_path not in discovered_paths_str
            ]
            needs_relationship_refresh = bool(
                stale_paths or any(plan[3] for plan in file_plans)
            )

            # ----------------------------------------------------------
            # 6. Parse files and collect entities/relationships
            # ----------------------------------------------------------
            entities_to_upsert: list[CodeEntity] = []
            all_relationships: list[CodeRelationship] = []
            errors: list[str] = []
            files_processed = 0
            files_examined = 0
            files_requiring_reindex_prep: list[tuple[str, set[str]]] = []

            for fpath, relative_path, file_hash, needs_reindex in file_plans:
                if not needs_relationship_refresh and not needs_reindex:
                    continue

                parser = self._registry.get_parser_for_file(fpath)
                file_entity_id = f"{repo_name}::{relative_path}"

                if parser is None:
                    if needs_reindex:
                        file_entity = CodeEntity(
                            id=file_entity_id,
                            name=fpath.name,
                            entity_type=EntityType.FILE,
                            repository=repo_name,
                            file_path=relative_path,
                            line_number=1,
                            language=fpath.suffix.lstrip(".") or "unknown",
                            content_hash=file_hash,
                        )
                        entities_to_upsert.append(file_entity)
                        files_requiring_reindex_prep.append(
                            (relative_path, {file_entity_id})
                        )
                        files_processed += 1
                    files_examined += 1
                    if progress_callback:
                        progress_callback(
                            files_total,
                            files_processed,
                            len(entities_to_upsert),
                        )
                    continue

                try:
                    parse_result: ParseResult = parser.parse_file(fpath, repo_name)
                except Exception as exc:
                    err_msg = f"Exception parsing {fpath}: {exc}"
                    logger.error(err_msg)
                    errors.append(err_msg)
                    if needs_reindex:
                        files_processed += 1
                    files_examined += 1
                    if progress_callback:
                        progress_callback(
                            files_total,
                            files_processed,
                            len(entities_to_upsert),
                        )
                    continue

                if parse_result.errors:
                    for pe in parse_result.errors:
                        err_msg = f"Parse error in {fpath}: {pe}"
                        logger.warning(err_msg)
                        errors.append(err_msg)
                    if needs_reindex:
                        files_processed += 1
                    files_examined += 1
                    if progress_callback:
                        progress_callback(
                            files_total,
                            files_processed,
                            len(entities_to_upsert),
                        )
                    continue

                normalized_entities, normalized_relationships = self._normalize_parse_result(
                    parse_result=parse_result,
                    relative_path=relative_path,
                    file_entity_id=file_entity_id,
                    language=parser.language,
                )
                all_relationships.extend(normalized_relationships)

                if needs_reindex:
                    file_entity = CodeEntity(
                        id=file_entity_id,
                        name=fpath.name,
                        entity_type=EntityType.FILE,
                        repository=repo_name,
                        file_path=relative_path,
                        line_number=1,
                        language=parser.language,
                        content_hash=file_hash,
                    )
                    entities_to_upsert.append(file_entity)
                    entities_to_upsert.extend(normalized_entities)
                    files_requiring_reindex_prep.append(
                        (
                            relative_path,
                            {file_entity_id}
                            | {entity.id for entity in normalized_entities},
                        )
                    )
                    files_processed += 1
                files_examined += 1

                if progress_callback:
                    progress_callback(
                        files_total,
                        files_processed,
                        len(entities_to_upsert),
                    )

            # ----------------------------------------------------------
            # 7. Embed entities
            # ----------------------------------------------------------
            try:
                await self._embed_entities(entities_to_upsert)
            except Exception as exc:
                err_msg = f"Embedding failed: {exc}"
                logger.error(err_msg)
                errors.append(err_msg)

            # ----------------------------------------------------------
            # 8. Apply graph changes atomically
            # ----------------------------------------------------------
            entities_created, relationships_created, _ = await self._graph.apply_indexing_changes(
                repository=repo_name,
                source=source,
                commit_sha=commit_sha,
                reindex_preparations=files_requiring_reindex_prep,
                entities=entities_to_upsert,
                relationships=all_relationships,
                stale_file_paths=stale_paths,
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

    @staticmethod
    def _relative_file_path(root: Path, file_path: Path) -> str:
        """Return a repository-relative path for stable file identity."""
        return str(file_path.relative_to(root))

    @staticmethod
    def _normalize_parse_result(
        parse_result: ParseResult,
        relative_path: str,
        file_entity_id: str,
        language: str,
    ) -> tuple[list[CodeEntity], list[CodeRelationship]]:
        """Rewrite parser output to the pipeline's canonical file identity."""
        parser_file_ids = {
            entity.id
            for entity in parse_result.entities
            if entity.entity_type == EntityType.FILE
        }
        id_map: dict[str, str] = {}
        call_aliases: dict[str, str] = {}

        if language in {"python", "javascript"}:
            id_map, call_aliases = IndexingPipeline._build_scoped_entity_maps(
                parse_result=parse_result,
                parser_file_ids=parser_file_ids,
                file_entity_id=file_entity_id,
            )

        normalized_entities: list[CodeEntity] = []
        for entity in parse_result.entities:
            if entity.entity_type == EntityType.FILE:
                continue
            entity.file_path = relative_path
            entity.id = id_map.get(entity.id, entity.id)
            normalized_entities.append(entity)

        normalized_relationships: list[CodeRelationship] = []
        for relationship in parse_result.relationships:
            if relationship.source_id in parser_file_ids:
                relationship.source_id = file_entity_id
            else:
                relationship.source_id = id_map.get(
                    relationship.source_id,
                    relationship.source_id,
                )

            if relationship.target_id in parser_file_ids:
                relationship.target_id = file_entity_id
            else:
                relationship.target_id = id_map.get(
                    relationship.target_id,
                    relationship.target_id,
                )
                if relationship.relationship_type == RelationshipType.CALLS:
                    relationship.target_id = call_aliases.get(
                        relationship.target_id,
                        relationship.target_id,
                    )
            normalized_relationships.append(relationship)

        return normalized_entities, normalized_relationships

    @staticmethod
    def _build_scoped_entity_maps(
        parse_result: ParseResult,
        parser_file_ids: set[str],
        file_entity_id: str,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Build stable per-file IDs for parser-local entities and local call aliases."""
        local_entities = [
            entity for entity in parse_result.entities
            if entity.entity_type != EntityType.FILE
        ]
        if not local_entities:
            return {}, {}

        entity_lookup = {entity.id: entity for entity in local_entities}
        structural_relationships = {
            RelationshipType.CONTAINS,
            RelationshipType.DECLARES,
            RelationshipType.HAS_METHOD,
            RelationshipType.HAS_CONSTRUCTOR,
            RelationshipType.HAS_FIELD,
        }
        parent_ids: dict[str, str | None] = {}

        for relationship in parse_result.relationships:
            if relationship.relationship_type not in structural_relationships:
                continue
            if relationship.target_id not in entity_lookup:
                continue
            if relationship.source_id in parser_file_ids:
                parent_ids.setdefault(relationship.target_id, None)
            elif relationship.source_id in entity_lookup:
                parent_ids.setdefault(relationship.target_id, relationship.source_id)

        local_paths: dict[str, str] = {}
        used_paths: set[str] = set()
        visiting: set[str] = set()

        def assign_local_path(entity_id: str) -> str:
            if entity_id in local_paths:
                return local_paths[entity_id]

            entity = entity_lookup[entity_id]
            if entity_id in visiting:
                return entity.name

            visiting.add(entity_id)
            parent_id = parent_ids.get(entity_id)
            parent_path = ""
            if parent_id and parent_id in entity_lookup:
                parent_path = assign_local_path(parent_id)

            candidate = f"{parent_path}.{entity.name}" if parent_path else entity.name
            unique_candidate = candidate
            if unique_candidate in used_paths:
                unique_candidate = f"{candidate}@L{entity.line_number}"
                duplicate_index = 2
                while unique_candidate in used_paths:
                    unique_candidate = f"{candidate}@L{entity.line_number}_{duplicate_index}"
                    duplicate_index += 1

            local_paths[entity_id] = unique_candidate
            used_paths.add(unique_candidate)
            visiting.remove(entity_id)
            return unique_candidate

        for entity in local_entities:
            assign_local_path(entity.id)

        id_map = {
            entity_id: f"{file_entity_id}#{local_paths[entity_id]}"
            for entity_id in local_paths
        }
        call_aliases = {
            f"{entity.repository}::{local_paths[entity.id]}": id_map[entity.id]
            for entity in local_entities
        }
        return id_map, call_aliases

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
