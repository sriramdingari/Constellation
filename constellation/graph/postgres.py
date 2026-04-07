"""PostgreSQL implementation of WriteBackend using asyncpg."""

from __future__ import annotations

import json
import logging
from typing import Any

import asyncpg

from constellation.graph.base import WriteBackend
from constellation.models import CodeEntity, CodeRelationship, EntityType

logger = logging.getLogger(__name__)

# DDL executed by initialize_schema() — all statements are idempotent
_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS code_repos (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL UNIQUE,
    source          TEXT NOT NULL,
    commit_sha      TEXT,
    entity_count    INTEGER DEFAULT 0,
    last_indexed_at TIMESTAMPTZ DEFAULT now(),
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS code_symbols (
    id              TEXT PRIMARY KEY,
    repo_id         UUID REFERENCES code_repos(id) ON DELETE CASCADE,
    repository      TEXT NOT NULL,
    file_path       TEXT NOT NULL,
    symbol_name     TEXT NOT NULL,
    symbol_type     TEXT NOT NULL,
    language        TEXT NOT NULL,
    line_start      INTEGER,
    line_end        INTEGER,
    signature       TEXT,
    code            TEXT,
    docstring       TEXT,
    return_type     TEXT,
    modifiers       TEXT[],
    stereotypes     TEXT[],
    properties      JSONB DEFAULT '{}',
    content_hash    TEXT,
    is_test         BOOLEAN GENERATED ALWAYS AS ('test' = ANY(stereotypes)) STORED,
    is_endpoint     BOOLEAN GENERATED ALWAYS AS ('endpoint' = ANY(stereotypes)) STORED,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS code_references (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_id          UUID REFERENCES code_repos(id) ON DELETE CASCADE,
    source_symbol_id TEXT REFERENCES code_symbols(id) ON DELETE CASCADE,
    target_symbol_id TEXT REFERENCES code_symbols(id) ON DELETE CASCADE,
    ref_type         TEXT NOT NULL,
    properties       JSONB DEFAULT '{}',
    UNIQUE (source_symbol_id, target_symbol_id, ref_type)
);

CREATE INDEX IF NOT EXISTS idx_symbols_repo   ON code_symbols(repository);
CREATE INDEX IF NOT EXISTS idx_symbols_type   ON code_symbols(repository, symbol_type);
CREATE INDEX IF NOT EXISTS idx_symbols_name   ON code_symbols(symbol_name);
CREATE INDEX IF NOT EXISTS idx_symbols_file   ON code_symbols(repository, file_path);
CREATE INDEX IF NOT EXISTS idx_symbols_hash   ON code_symbols(repository, file_path, content_hash);
CREATE INDEX IF NOT EXISTS idx_refs_source    ON code_references(source_symbol_id);
CREATE INDEX IF NOT EXISTS idx_refs_target    ON code_references(target_symbol_id);
CREATE INDEX IF NOT EXISTS idx_refs_type      ON code_references(ref_type);
"""

_HNSW_INDEX = """
CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON code_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
"""

_EMBEDDABLE_TYPES = {
    EntityType.METHOD, EntityType.CLASS,
    EntityType.INTERFACE, EntityType.CONSTRUCTOR,
}


class PostgresWriteBackend(WriteBackend):
    """PostgreSQL + pgvector implementation of WriteBackend."""

    def __init__(self, dsn: str, embedding_dimensions: int = 1536) -> None:
        self._dsn = dsn
        self._embedding_dimensions = embedding_dimensions
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        from pgvector.asyncpg import register_vector

        # Ensure the pgvector extension exists BEFORE creating the pool,
        # otherwise register_vector (used as pool init) fails with
        # "unknown type: public.vector" on a fresh database.
        bootstrap = await asyncpg.connect(self._dsn)
        try:
            await bootstrap.execute("CREATE EXTENSION IF NOT EXISTS vector")
        finally:
            await bootstrap.close()

        self._pool = await asyncpg.create_pool(
            self._dsn, min_size=1, max_size=10, init=register_vector
        )

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    def _require_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PostgresWriteBackend: call connect() before using the backend")
        return self._pool

    async def _existing_embedding_dimensions(self, conn: asyncpg.Connection) -> int | None:
        """Return the configured vector dimensions for code_embeddings.embedding,
        or None if the table does not exist yet.

        Uses to_regclass() which returns NULL for missing tables, so the join
        silently produces zero rows → fetchval returns None. pgvector stores
        the configured dimension directly in pg_attribute.atttypmod (verified
        in pgvector source: vector_typmod_in returns *tl with no offset).
        """
        return await conn.fetchval("""
            SELECT atttypmod
            FROM pg_attribute
            WHERE attrelid = to_regclass('public.code_embeddings')
              AND attname = 'embedding'
        """)

    async def initialize_schema(self) -> None:
        """Run idempotent DDL. Safe to call on every startup.

        Handles embedding dimension drift: if code_embeddings exists with a
        different vector dimension than configured, drop and recreate it.
        """
        pool = self._require_pool()
        async with pool.acquire() as conn:
            # Execute the entire DDL block as one call — asyncpg supports
            # multi-statement strings natively. _DDL creates everything
            # except code_embeddings (which has a configurable dimension).
            await conn.execute(_DDL)

            # Check existing code_embeddings dimensions, drop+recreate if mismatched
            existing_dim = await self._existing_embedding_dimensions(conn)
            if existing_dim is not None and existing_dim != self._embedding_dimensions:
                logger.warning(
                    "Embedding dimension drift: existing=%d, configured=%d. "
                    "Dropping and recreating code_embeddings.",
                    existing_dim, self._embedding_dimensions,
                )
                await conn.execute("DROP TABLE IF EXISTS code_embeddings CASCADE")

            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS code_embeddings (
                    symbol_id    TEXT PRIMARY KEY REFERENCES code_symbols(id) ON DELETE CASCADE,
                    embedding    vector({self._embedding_dimensions}),
                    content_hash TEXT
                )
            """)
            await conn.execute(_HNSW_INDEX)

    # ── Repository ───────────────────────────────────────────────────────

    async def upsert_repository(
        self, name: str, source: str,
        commit_sha: str | None, entity_count: int,
    ) -> None:
        pool = self._require_pool()
        await pool.execute("""
            INSERT INTO code_repos (name, source, commit_sha, entity_count, last_indexed_at)
            VALUES ($1, $2, $3, $4, now())
            ON CONFLICT (name) DO UPDATE SET
                source          = EXCLUDED.source,
                commit_sha      = EXCLUDED.commit_sha,
                entity_count    = EXCLUDED.entity_count,
                last_indexed_at = now()
        """, name, source, commit_sha, entity_count)

    @staticmethod
    def _normalize_repo_row(row: dict) -> dict:
        """Normalize Postgres row to match the API's RepositoryInfo schema."""
        result = dict(row)
        # Rename commit_sha → last_commit_sha (API expectation)
        if "commit_sha" in result:
            result["last_commit_sha"] = result.pop("commit_sha")
        # Convert datetime to ISO string (Pydantic expects str)
        if "last_indexed_at" in result and result["last_indexed_at"] is not None:
            result["last_indexed_at"] = result["last_indexed_at"].isoformat()
        if "created_at" in result and result["created_at"] is not None:
            result["created_at"] = result["created_at"].isoformat()
        return result

    async def get_repository(self, name: str) -> dict | None:
        pool = self._require_pool()
        row = await pool.fetchrow(
            "SELECT * FROM code_repos WHERE name = $1", name
        )
        return self._normalize_repo_row(dict(row)) if row else None

    async def list_repositories(self) -> list[dict]:
        pool = self._require_pool()
        rows = await pool.fetch(
            "SELECT * FROM code_repos ORDER BY name"
        )
        return [self._normalize_repo_row(dict(r)) for r in rows]

    async def delete_repository(self, name: str) -> None:
        pool = self._require_pool()
        # code_symbols has CASCADE to code_references and code_embeddings.
        # Delete symbols first since repo_id FK is not populated.
        await pool.execute(
            "DELETE FROM code_symbols WHERE repository = $1", name
        )
        await pool.execute(
            "DELETE FROM code_repos WHERE name = $1", name
        )

    # ── Change detection ─────────────────────────────────────────────────

    async def get_file_hashes(self, repository: str) -> dict[str, str]:
        pool = self._require_pool()
        rows = await pool.fetch(
            "SELECT file_path, content_hash FROM code_symbols "
            "WHERE repository = $1 AND symbol_type = 'File' AND content_hash IS NOT NULL",
            repository,
        )
        return {r["file_path"]: r["content_hash"] for r in rows}

    # ── Indexing (atomic) ────────────────────────────────────────────────

    async def apply_indexing_changes(
        self, *,
        repository: str,
        source: str,
        commit_sha: str | None,
        reindex_preparations: list[tuple[str, set[str]]],
        entities: list[CodeEntity],
        relationships: list[CodeRelationship],
        stale_file_paths: list[str],
    ) -> tuple[int, int, int]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                # 1. Delete stale files (files that no longer exist on disk).
                # Excludes packages — they're shared across files.
                if stale_file_paths:
                    await conn.execute(
                        "DELETE FROM code_symbols WHERE repository = $1 "
                        "AND file_path = ANY($2) AND symbol_type != 'Package'",
                        repository, stale_file_paths,
                    )

                # 2. For each reindexed file, snapshot existing entity IDs and
                # delete only the orphans (existing - current). Don't delete
                # entities that are being re-upserted — that lets the upsert
                # take the UPDATE path so entities_created reflects only true
                # new entities. Mirrors Neo4j's _prepare_file_reindex.
                #
                # Also delete all outbound relationships sourced from any
                # entity in the reindex set BEFORE the upsert. Without this,
                # stale relationships (e.g. an EXTENDS edge that was removed
                # in a refactor) would survive forever — the new
                # _upsert_relationships uses ON CONFLICT DO NOTHING and
                # cannot remove rows. Mirrors Neo4j's
                # DELETE_FILE_OUTBOUND_RELATIONSHIPS in _prepare_file_reindex.
                for file_path, current_entity_ids in reindex_preparations:
                    existing_rows = await conn.fetch(
                        "SELECT id FROM code_symbols WHERE repository = $1 "
                        "AND file_path = $2 AND symbol_type != 'Package'",
                        repository, file_path,
                    )
                    existing_ids = {r["id"] for r in existing_rows}

                    # Delete outbound relationships for all entities in this file
                    # (both surviving and being-removed). They'll be recreated by
                    # _upsert_relationships from the new parsed set.
                    if existing_ids:
                        await conn.execute(
                            "DELETE FROM code_references WHERE source_symbol_id = ANY($1)",
                            list(existing_ids),
                        )

                    # Delete only the orphan entities (in snapshot, not in current)
                    stale_ids = list(existing_ids - current_entity_ids)
                    if stale_ids:
                        await conn.execute(
                            "DELETE FROM code_symbols WHERE id = ANY($1)",
                            stale_ids,
                        )

                # 3. Upsert all entities (RETURNING xmax tracks true creations)
                entities_created = await self._upsert_entities(conn, entities)

                # 4. Upsert all relationships (skips unresolved endpoints via EXISTS)
                rels_created = await self._upsert_relationships(conn, relationships)

                # 5. Upsert embeddings for embeddable entities
                await self._upsert_embeddings(conn, entities)

                # 6. Cleanup orphan packages — loop until stable.
                # Nested namespaces require multiple passes: leaf must be
                # deleted before its parent becomes orphan-eligible.
                while True:
                    result = await conn.execute("""
                        DELETE FROM code_symbols
                        WHERE repository = $1 AND symbol_type = 'Package'
                        AND id NOT IN (
                            SELECT DISTINCT target_symbol_id FROM code_references
                            WHERE ref_type = 'IN_PACKAGE' AND target_symbol_id IS NOT NULL
                        )
                        AND NOT EXISTS (
                            SELECT 1 FROM code_symbols child
                            WHERE child.repository = $1
                            AND child.symbol_type = 'Package'
                            AND starts_with(child.id, code_symbols.id || '.')
                        )
                    """, repository)
                    deleted = int(result.split()[-1]) if result.startswith("DELETE") else 0
                    if deleted == 0:
                        break

                # 7. Upsert repository metadata
                total = await conn.fetchval(
                    "SELECT COUNT(*) FROM code_symbols WHERE repository = $1", repository
                )
                await conn.execute("""
                    INSERT INTO code_repos (name, source, commit_sha, entity_count, last_indexed_at)
                    VALUES ($1, $2, $3, $4, now())
                    ON CONFLICT (name) DO UPDATE SET
                        source          = EXCLUDED.source,
                        commit_sha      = EXCLUDED.commit_sha,
                        entity_count    = EXCLUDED.entity_count,
                        last_indexed_at = now()
                """, repository, source, commit_sha, total)

                return entities_created, rels_created, total

    async def _upsert_entities(self, conn: asyncpg.Connection, entities: list[CodeEntity]) -> int:
        if not entities:
            return 0
        created = 0
        for entity in entities:
            # RETURNING (xmax = 0) AS inserted: xmax is 0 only on a true INSERT.
            # On UPDATE (i.e. ON CONFLICT fired), xmax is the txn id, so this
            # returns False. This gives us an exact "newly created" count
            # matching Neo4j's MERGE ON CREATE semantics.
            inserted = await conn.fetchval("""
                INSERT INTO code_symbols (
                    id, repository, file_path, symbol_name, symbol_type,
                    language, line_start, line_end, signature, code,
                    docstring, return_type, modifiers, stereotypes,
                    properties, content_hash
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15::jsonb,$16)
                ON CONFLICT (id) DO UPDATE SET
                    file_path    = EXCLUDED.file_path,
                    symbol_name  = EXCLUDED.symbol_name,
                    symbol_type  = EXCLUDED.symbol_type,
                    language     = EXCLUDED.language,
                    line_start   = EXCLUDED.line_start,
                    line_end     = EXCLUDED.line_end,
                    signature    = EXCLUDED.signature,
                    code         = EXCLUDED.code,
                    docstring    = EXCLUDED.docstring,
                    return_type  = EXCLUDED.return_type,
                    modifiers    = EXCLUDED.modifiers,
                    stereotypes  = EXCLUDED.stereotypes,
                    properties   = EXCLUDED.properties,
                    content_hash = EXCLUDED.content_hash,
                    updated_at   = now()
                RETURNING (xmax = 0) AS inserted
            """,
                entity.id, entity.repository, entity.file_path, entity.name,
                entity.entity_type.value, entity.language,
                entity.line_number, entity.line_end,
                entity.signature, entity.code, entity.docstring, entity.return_type,
                entity.modifiers or [], entity.stereotypes or [],
                json.dumps(entity.properties or {}), entity.content_hash,
            )
            if inserted:
                created += 1
        return created

    async def _upsert_relationships(self, conn: asyncpg.Connection, relationships: list[CodeRelationship]) -> int:
        if not relationships:
            return 0
        created = 0
        for rel in relationships:
            # Use INSERT ... SELECT ... WHERE EXISTS to silently skip edges
            # whose endpoints are not in code_symbols. This mirrors Neo4j's
            # MATCH (source) MATCH (target) MERGE behavior — unresolved targets
            # (e.g. Java external superclasses, C# external::Base types,
            # deleted-file references) are dropped, not raised.
            result = await conn.execute("""
                INSERT INTO code_references (source_symbol_id, target_symbol_id, ref_type, properties)
                SELECT $1, $2, $3, $4::jsonb
                WHERE EXISTS (SELECT 1 FROM code_symbols WHERE id = $1)
                  AND EXISTS (SELECT 1 FROM code_symbols WHERE id = $2)
                ON CONFLICT (source_symbol_id, target_symbol_id, ref_type) DO NOTHING
            """, rel.source_id, rel.target_id, rel.relationship_type.value, json.dumps(rel.properties or {}))
            if result.endswith(" 1"):
                created += 1
        return created

    async def _upsert_embeddings(self, conn: asyncpg.Connection, entities: list[CodeEntity]) -> None:
        embeddable = [e for e in entities if e.entity_type in _EMBEDDABLE_TYPES and e.embedding]
        for entity in embeddable:
            await conn.execute("""
                INSERT INTO code_embeddings (symbol_id, embedding, content_hash)
                VALUES ($1, $2, $3)
                ON CONFLICT (symbol_id) DO UPDATE SET
                    embedding    = EXCLUDED.embedding,
                    content_hash = EXCLUDED.content_hash
            """, entity.id, entity.embedding, entity.content_hash)

    # ── Stats ────────────────────────────────────────────────────────────

    async def count_repository_entities(self, repository: str) -> int:
        pool = self._require_pool()
        return await pool.fetchval(
            "SELECT COUNT(*) FROM code_symbols WHERE repository = $1", repository
        )
