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

    async def query(self, cypher: str, **params) -> list[dict]:
        """Run a Cypher query and return result records as dicts.

        Auto-connects if no driver is present.
        """
        if not self._driver:
            await self.connect()
        async with self._driver.session() as session:
            result = await session.run(cypher, params)
            return [record.data() for record in await result.fetch(1000)]

    async def initialize_schema(self):
        """Create all constraints, composite indexes, and vector indexes."""
        for constraint in schema.CONSTRAINTS:
            await self.query(constraint)
        for index in schema.COMPOSITE_INDEXES:
            await self.query(index)
        for label in schema.EMBEDDABLE_LABELS:
            index_name = f"vector_{label.lower()}_embedding"
            q = queries.CREATE_VECTOR_INDEX.format(
                index_name=index_name, label=label
            )
            await self.query(q, dimensions=self._settings.embedding_dimensions)

    async def upsert_entities(self, entities: list[CodeEntity]) -> int:
        """Upsert a batch of code entities into the graph.

        Entities are grouped by their entity_type label and a single
        UNWIND/MERGE query is issued per label.

        Returns the total count reported by the database.
        """
        if not entities:
            return 0

        by_label: dict[str, list[CodeEntity]] = {}
        for entity in entities:
            label = entity.entity_type.value
            by_label.setdefault(label, []).append(entity)

        total = 0
        for label, ents in by_label.items():
            q = queries.upsert_entities_query(label)
            entity_dicts = []
            for e in ents:
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
                entity_dicts.append({"id": e.id, "properties": props})
            result = await self.query(q, entities=entity_dicts)
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
        if not relationships:
            return 0

        by_type: dict[str, list[CodeRelationship]] = {}
        for rel in relationships:
            rel_type = rel.relationship_type.value
            by_type.setdefault(rel_type, []).append(rel)

        total = 0
        for rel_type, rels in by_type.items():
            q = queries.create_relationships_query(rel_type)
            rel_dicts = [
                {
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "properties": r.properties,
                }
                for r in rels
            ]
            result = await self.query(q, relationships=rel_dicts)
            if result:
                total += result[0].get("count", 0)
        return total

    async def get_file_hashes(self, repository: str) -> dict[str, str]:
        """Return a mapping of file_path to content_hash for a repository."""
        records = await self.query(
            queries.GET_FILE_HASHES, repository=repository
        )
        return {r["file_path"]: r["content_hash"] for r in records}

    async def delete_stale_files(
        self, repository: str, file_paths: list[str]
    ):
        """Delete files (and their descendant nodes) that are no longer present."""
        if not file_paths:
            return
        await self.query(
            queries.DELETE_STALE_FILES,
            repository=repository,
            file_paths=file_paths,
        )

    async def upsert_repository(
        self,
        name: str,
        source: str,
        commit_sha: str | None,
        entity_count: int,
    ):
        """Create or update a Repository node."""
        await self.query(
            queries.UPSERT_REPOSITORY,
            name=name,
            source=source,
            commit_sha=commit_sha,
            entity_count=entity_count,
        )

    async def get_repository(self, name: str) -> dict | None:
        """Fetch a single Repository node by name, or None if not found."""
        records = await self.query(queries.GET_REPOSITORY, name=name)
        return records[0]["r"] if records else None

    async def list_repositories(self) -> list[dict]:
        """Return all Repository nodes ordered by name."""
        records = await self.query(queries.LIST_REPOSITORIES)
        return [r["r"] for r in records]

    async def delete_repository(self, name: str):
        """Delete a Repository node and all its descendants."""
        await self.query(queries.DELETE_REPOSITORY, name=name)
