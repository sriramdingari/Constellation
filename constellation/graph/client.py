"""Neo4j async graph client for the Constellation code knowledge graph."""

from neo4j import AsyncGraphDatabase

from constellation.config import Settings
from constellation.graph import schema, queries
from constellation.models import CodeEntity, CodeRelationship


class GraphClient:
    """Async wrapper around the Neo4j driver for code graph operations."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._driver = None

    async def connect(self):
        """Create the async driver and verify connectivity."""
        self._driver = AsyncGraphDatabase.driver(
            self._settings.neo4j_uri,
            auth=(self._settings.neo4j_user, self._settings.neo4j_password),
        )
        await self._driver.verify_connectivity()

    async def close(self):
        """Close the driver if connected."""
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def _ensure_driver(self):
        """Connect lazily when a graph operation is issued."""
        if not self._driver:
            await self.connect()

    @staticmethod
    async def _consume_result(result) -> list[dict]:
        """Collect all async Neo4j records into plain dicts."""
        return [record.data() async for record in result]

    @staticmethod
    def _chunked(items, batch_size: int):
        """Yield stable slices so large writes stay bounded inside a transaction."""
        size = max(1, batch_size)
        for index in range(0, len(items), size):
            yield items[index : index + size]

    async def _run_with_runner(self, runner, cypher: str, **params) -> list[dict]:
        """Execute a Cypher query on a session or transaction runner."""
        result = await runner.run(cypher, params)
        return await self._consume_result(result)

    async def query(self, cypher: str, **params) -> list[dict]:
        """Run a Cypher query and return result records as dicts.

        Auto-connects if no driver is present.
        """
        await self._ensure_driver()
        async with self._driver.session() as session:
            return await self._run_with_runner(session, cypher, **params)

    async def initialize_schema(self):
        """Create all constraints, composite indexes, and vector indexes."""
        embedding_dimensions = self._settings.resolved_embedding_dimensions()
        for constraint in schema.CONSTRAINTS:
            await self.query(constraint)
        for index in schema.COMPOSITE_INDEXES:
            await self.query(index)
        for label in schema.EMBEDDABLE_LABELS:
            index_name = f"vector_{label.lower()}_embedding"
            existing_dimensions = await self._get_vector_index_dimensions(index_name)
            if (
                existing_dimensions is not None
                and existing_dimensions != embedding_dimensions
            ):
                await self.query(queries.drop_index_query(index_name))
            q = queries.CREATE_VECTOR_INDEX.format(
                index_name=index_name, label=label
            )
            await self.query(q, dimensions=embedding_dimensions)

    async def upsert_entities(self, entities: list[CodeEntity]) -> int:
        """Upsert a batch of code entities into the graph.

        Entities are grouped by their entity_type label and a single
        UNWIND/MERGE query is issued per label.

        Returns the total count reported by the database.
        """
        await self._ensure_driver()
        async with self._driver.session() as session:
            return await self._upsert_entities_with_runner(session, entities)

    async def _upsert_entities_with_runner(self, runner, entities: list[CodeEntity]) -> int:
        """Upsert a batch of entities using a session or transaction runner."""
        if not entities:
            return 0

        by_label: dict[str, list[CodeEntity]] = {}
        for entity in entities:
            label = entity.entity_type.value
            by_label.setdefault(label, []).append(entity)

        total = 0
        for label, ents in by_label.items():
            q = queries.upsert_entities_query(label)
            for batch in self._chunked(ents, self._settings.entity_batch_size):
                entity_dicts = []
                for e in batch:
                    props: dict = {
                        "id": e.id,
                        "name": e.name,
                        "repository": e.repository,
                        "file_path": e.file_path,
                        "line_number": e.line_number,
                        "language": e.language,
                    }
                    if e.line_end is not None:
                        props["line_end"] = e.line_end
                    if e.code is not None:
                        props["code"] = e.code
                    if e.signature is not None:
                        props["signature"] = e.signature
                    if e.return_type is not None:
                        props["return_type"] = e.return_type
                    if e.docstring is not None:
                        props["docstring"] = e.docstring
                    if e.modifiers:
                        props["modifiers"] = e.modifiers
                    if e.stereotypes:
                        props["stereotypes"] = e.stereotypes
                    if e.content_hash is not None:
                        props["content_hash"] = e.content_hash
                    if e.embedding is not None:
                        props["embedding"] = e.embedding
                    for key, value in e.properties.items():
                        props.setdefault(key, value)
                    entity_dicts.append({"id": e.id, "properties": props})
                result = await self._run_with_runner(runner, q, entities=entity_dicts)
                if result:
                    total += result[0].get("count", 0)
        return total

    async def create_relationships(
        self, relationships: list[CodeRelationship]
    ) -> int:
        """Create/merge a batch of relationships in the graph.

        Relationships are grouped by type and a single UNWIND/MERGE query
        is issued per type.

        Returns the total count reported by the database.
        """
        await self._ensure_driver()
        async with self._driver.session() as session:
            return await self._create_relationships_with_runner(session, relationships)

    async def _create_relationships_with_runner(
        self,
        runner,
        relationships: list[CodeRelationship],
    ) -> int:
        """Create/merge relationships using a session or transaction runner."""
        if not relationships:
            return 0

        by_type: dict[str, list[CodeRelationship]] = {}
        for rel in relationships:
            rel_type = rel.relationship_type.value
            by_type.setdefault(rel_type, []).append(rel)

        total = 0
        for rel_type, rels in by_type.items():
            q = queries.create_relationships_query(rel_type)
            for batch in self._chunked(rels, self._settings.entity_batch_size):
                rel_dicts = [
                    {
                        "source_id": r.source_id,
                        "target_id": r.target_id,
                        "properties": r.properties,
                    }
                    for r in batch
                ]
                result = await self._run_with_runner(
                    runner,
                    q,
                    relationships=rel_dicts,
                )
                if result:
                    total += result[0].get("count", 0)
        return total

    async def get_file_hashes(self, repository: str) -> dict[str, str]:
        """Return a mapping of file_path to content_hash for a repository."""
        records = await self.query(
            queries.GET_FILE_HASHES, repository=repository
        )
        return {r["file_path"]: r["content_hash"] for r in records}

    async def prepare_file_reindex(
        self,
        repository: str,
        file_path: str,
        current_entity_ids: set[str],
    ) -> bool:
        """Clear stale relationships/nodes for a file before reindexing it.

        Returns ``True`` when the file already had indexed graph state.
        """
        await self._ensure_driver()
        async with self._driver.session() as session:
            return await self._prepare_file_reindex_with_runner(
                session,
                repository,
                file_path,
                current_entity_ids,
            )

    async def _prepare_file_reindex_with_runner(
        self,
        runner,
        repository: str,
        file_path: str,
        current_entity_ids: set[str],
    ) -> bool:
        """Clear stale relationships/nodes for a file using a runner."""
        records = await self._run_with_runner(
            runner,
            queries.GET_FILE_ENTITY_SNAPSHOT,
            repository=repository,
            file_path=file_path,
        )
        if not records:
            return False

        await self._run_with_runner(
            runner,
            queries.DELETE_FILE_OUTBOUND_RELATIONSHIPS,
            repository=repository,
            file_path=file_path,
        )

        stale_ids = [
            record["id"]
            for record in records
            if "Package" not in record.get("labels", [])
            and record["id"] not in current_entity_ids
        ]
        if stale_ids:
            await self._run_with_runner(
                runner,
                queries.DELETE_ENTITIES_BY_IDS,
                repository=repository,
                entity_ids=stale_ids,
            )

        return True

    async def delete_stale_files(
        self, repository: str, file_paths: list[str]
    ):
        """Delete files (and their descendant nodes) that are no longer present."""
        await self._ensure_driver()
        async with self._driver.session() as session:
            await self._delete_stale_files_with_runner(session, repository, file_paths)

    async def _delete_stale_files_with_runner(
        self,
        runner,
        repository: str,
        file_paths: list[str],
        *,
        cleanup_orphans: bool = True,
    ):
        """Delete stale files using a session or transaction runner."""
        if not file_paths:
            return
        await self._run_with_runner(
            runner,
            queries.DELETE_STALE_FILES,
            repository=repository,
            file_paths=file_paths,
        )
        if cleanup_orphans:
            await self._cleanup_orphan_packages_with_runner(runner, repository)
            await self._cleanup_orphan_references_with_runner(runner, repository)

    async def cleanup_orphan_packages(self, repository: str):
        """Delete orphan package nodes until no more remain."""
        await self._ensure_driver()
        async with self._driver.session() as session:
            await self._cleanup_orphan_packages_with_runner(session, repository)

    async def _cleanup_orphan_packages_with_runner(self, runner, repository: str):
        """Delete orphan package nodes until no more remain."""
        while True:
            records = await self._run_with_runner(
                runner,
                queries.DELETE_ORPHAN_PACKAGES,
                repository=repository,
            )
            deleted = records[0].get("count", 0) if records else 0
            if deleted == 0:
                return

    async def _cleanup_orphan_references_with_runner(self, runner, repository: str):
        """Delete reference nodes that no longer have any callers."""
        await self._run_with_runner(
            runner,
            queries.DELETE_ORPHAN_REFERENCES,
            repository=repository,
        )

    async def count_repository_entities(self, repository: str) -> int:
        """Return the number of indexed nodes owned by a repository."""
        await self._ensure_driver()
        async with self._driver.session() as session:
            return await self._count_repository_entities_with_runner(session, repository)

    async def _count_repository_entities_with_runner(
        self,
        runner,
        repository: str,
    ) -> int:
        """Return the number of indexed nodes owned by a repository."""
        records = await self._run_with_runner(
            runner,
            queries.COUNT_REPOSITORY_ENTITIES,
            repository=repository,
        )
        return records[0].get("count", 0) if records else 0

    async def upsert_repository(
        self,
        name: str,
        source: str,
        commit_sha: str | None,
        entity_count: int,
    ):
        """Create or update a Repository node."""
        await self._ensure_driver()
        async with self._driver.session() as session:
            await self._upsert_repository_with_runner(
                session,
                name=name,
                source=source,
                commit_sha=commit_sha,
                entity_count=entity_count,
            )

    async def _upsert_repository_with_runner(
        self,
        runner,
        *,
        name: str,
        source: str,
        commit_sha: str | None,
        entity_count: int,
    ):
        """Create or update a Repository node using a session or transaction."""
        await self._run_with_runner(
            runner,
            queries.UPSERT_REPOSITORY,
            name=name,
            source=source,
            commit_sha=commit_sha,
            entity_count=entity_count,
        )

    async def apply_indexing_changes(
        self,
        *,
        repository: str,
        source: str,
        commit_sha: str | None,
        reindex_preparations: list[tuple[str, set[str]]],
        entities: list[CodeEntity],
        relationships: list[CodeRelationship],
        stale_file_paths: list[str],
    ) -> tuple[int, int, int]:
        """Apply one indexing run atomically.

        This keeps file reconciliation, entity upserts, relationship creation,
        stale-file deletion, and repository metadata updates in a single
        database transaction.
        """
        await self._ensure_driver()
        async with self._driver.session() as session:
            tx = await session.begin_transaction()
            try:
                changed_existing_files = False
                for file_path, current_entity_ids in reindex_preparations:
                    changed_existing_files = await self._prepare_file_reindex_with_runner(
                        tx,
                        repository,
                        file_path,
                        current_entity_ids,
                    ) or changed_existing_files

                entities_created = await self._upsert_entities_with_runner(tx, entities)

                if stale_file_paths:
                    await self._delete_stale_files_with_runner(
                        tx,
                        repository,
                        stale_file_paths,
                        cleanup_orphans=False,
                    )

                relationships_created = await self._create_relationships_with_runner(
                    tx,
                    relationships,
                )

                if stale_file_paths or changed_existing_files:
                    await self._cleanup_orphan_packages_with_runner(tx, repository)
                    await self._cleanup_orphan_references_with_runner(tx, repository)

                repository_entity_count = await self._count_repository_entities_with_runner(
                    tx,
                    repository,
                )
                await self._upsert_repository_with_runner(
                    tx,
                    name=repository,
                    source=source,
                    commit_sha=commit_sha,
                    entity_count=repository_entity_count,
                )

                await tx.commit()
                return (
                    entities_created,
                    relationships_created,
                    repository_entity_count,
                )
            except Exception:
                await tx.rollback()
                raise

    async def get_repository(self, name: str) -> dict | None:
        """Fetch a single Repository node by name, or None if not found."""
        records = await self.query(queries.GET_REPOSITORY, name=name)
        return records[0]["r"] if records else None

    async def list_repositories(self) -> list[dict]:
        """Return all Repository nodes ordered by name."""
        records = await self.query(queries.LIST_REPOSITORIES)
        return [r["r"] for r in records]

    async def delete_repository(self, name: str):
        """Delete a Repository node and all nodes owned by that repository."""
        await self.query(queries.DELETE_REPOSITORY, name=name)

    async def _get_vector_index_dimensions(self, index_name: str) -> int | None:
        """Return the configured dimensions for an existing vector index."""
        records = await self.query(
            queries.GET_VECTOR_INDEX,
            index_name=index_name,
        )
        if not records:
            return None

        options = records[0].get("options")
        if not isinstance(options, dict):
            return None

        index_config = options.get("indexConfig")
        if not isinstance(index_config, dict):
            return None

        dimensions = index_config.get("vector.dimensions")
        if dimensions is None:
            dimensions = index_config.get("`vector.dimensions`")
        if dimensions is None:
            return None

        try:
            return int(dimensions)
        except (TypeError, ValueError):
            return None
