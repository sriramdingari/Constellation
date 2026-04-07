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

    def __init__(
        self,
        dsn: str,
        embedding_dimensions: int = 1536,
        *,
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        self._dsn = dsn
        self._embedding_dimensions = embedding_dimensions
        self._embedding_model = embedding_model
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

    async def _existing_embedding_metadata(self, conn: asyncpg.Connection) -> dict | None:
        """Return the stored embedding_metadata row, or None if empty.

        The caller is responsible for ensuring the table exists first
        (initialize_schema creates it via CREATE TABLE IF NOT EXISTS
        before this helper is called).
        """
        row = await conn.fetchrow(
            "SELECT dimensions, model FROM embedding_metadata WHERE id = 1"
        )
        return dict(row) if row else None

    async def initialize_schema(self) -> None:
        """Run idempotent DDL. Safe to call on every startup.

        Handles embedding corpus drift via a singleton embedding_metadata
        table. Any mismatch on dimensions OR model between configured and
        stored values triggers a drop+recreate of code_embeddings and
        clears File content hashes so the next indexing run regenerates
        embeddings.
        """
        pool = self._require_pool()
        async with pool.acquire() as conn:
            # 1. Base schema (everything except code_embeddings, which is
            # dimension-parameterized and created separately below)
            await conn.execute(_DDL)

            # 2. Embedding metadata singleton table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS embedding_metadata (
                    id         INTEGER PRIMARY KEY CHECK (id = 1),
                    dimensions INTEGER NOT NULL,
                    model      TEXT NOT NULL,
                    updated_at TIMESTAMPTZ DEFAULT now()
                )
            """)

            # 3. Check stored metadata against configured
            stored = await self._existing_embedding_metadata(conn)
            configured_dim = self._embedding_dimensions
            configured_model = self._embedding_model

            if stored is None:
                # Migration / fresh-install path.
                # If code_embeddings already exists (and possibly has rows
                # from a previous deployment that didn't track model
                # metadata), we can't know what model those embeddings came
                # from. Conservative choice: treat the migration boundary
                # as drift and clear the corpus, forcing a full re-embed on
                # the next indexing run. This is the same-dim edge case the
                # new metadata table fixes going forward; for existing
                # deployments we pay it once.
                pre_existing_embeddings = await conn.fetchval(
                    "SELECT to_regclass('code_embeddings')"
                )
                if pre_existing_embeddings is not None:
                    logger.warning(
                        "Migration: embedding_metadata was empty but "
                        "code_embeddings already exists. Treating as drift "
                        "and clearing the corpus so the next indexing run "
                        "regenerates embeddings with the configured model=%s.",
                        configured_model,
                    )
                    await conn.execute("DROP TABLE IF EXISTS code_embeddings CASCADE")
                    await conn.execute(
                        "UPDATE code_symbols SET content_hash = NULL "
                        "WHERE symbol_type = 'File'"
                    )
                # Insert the configured values as the new baseline.
                # ON CONFLICT DO NOTHING guards against a race where two
                # processes start simultaneously, both see stored=None, and
                # both try to INSERT row id=1 — without the guard, the second
                # would fail with a PK violation.
                await conn.execute("""
                    INSERT INTO embedding_metadata (id, dimensions, model)
                    VALUES (1, $1, $2)
                    ON CONFLICT (id) DO NOTHING
                """, configured_dim, configured_model)
            else:
                if stored["dimensions"] != configured_dim or stored["model"] != configured_model:
                    logger.warning(
                        "Embedding corpus drift: stored=(dim=%d, model=%s), "
                        "configured=(dim=%d, model=%s). Dropping code_embeddings "
                        "and clearing File content hashes so the next indexing "
                        "run regenerates embeddings for every file.",
                        stored["dimensions"], stored["model"],
                        configured_dim, configured_model,
                    )
                    await conn.execute("DROP TABLE IF EXISTS code_embeddings CASCADE")
                    await conn.execute(
                        "UPDATE code_symbols SET content_hash = NULL "
                        "WHERE symbol_type = 'File'"
                    )
                    await conn.execute("""
                        UPDATE embedding_metadata
                        SET dimensions = $1, model = $2, updated_at = now()
                        WHERE id = 1
                    """, configured_dim, configured_model)

            # 4. Create code_embeddings (if it doesn't exist, or was just dropped)
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS code_embeddings (
                    symbol_id    TEXT PRIMARY KEY REFERENCES code_symbols(id) ON DELETE CASCADE,
                    embedding    vector({configured_dim}),
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

        # Deduplicate by id (last-writer-wins) before counting and batching.
        # The pipeline shouldn't emit duplicates, but defensive dedup keeps
        # `created` accurate and makes the upsert idempotent against accidental
        # duplicates in the input list.
        deduped: dict[str, CodeEntity] = {}
        for e in entities:
            deduped[e.id] = e
        unique_entities = list(deduped.values())

        # Pre-fetch: which of these IDs already exist? Used to compute the
        # accurate `created` count (len(unique) - len(existing_ids)), since
        # executemany does not preserve RETURNING output reliably.
        entity_ids = list(deduped.keys())
        existing_rows = await conn.fetch(
            "SELECT id FROM code_symbols WHERE id = ANY($1)",
            entity_ids,
        )
        existing_ids = {r["id"] for r in existing_rows}
        created = len(unique_entities) - len(existing_ids)

        # Batch upsert via executemany — pipelined into a single network write,
        # avoiding N separate round-trips. asyncpg uses the extended query
        # protocol (BIND+EXECUTE per row) but pipelines all rows together.
        await conn.executemany("""
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
        """, [
            (
                e.id, e.repository, e.file_path, e.name,
                e.entity_type.value, e.language,
                e.line_number, e.line_end,
                e.signature, e.code, e.docstring, e.return_type,
                e.modifiers or [], e.stereotypes or [],
                json.dumps(e.properties or {}), e.content_hash,
            )
            for e in unique_entities
        ])

        return created

    async def _upsert_relationships(self, conn: asyncpg.Connection, relationships: list[CodeRelationship]) -> int:
        if not relationships:
            return 0

        # Deduplicate by (source, target, ref_type) — the table's UNIQUE
        # constraint means duplicates can't coexist anyway, so dedup upfront
        # keeps the before/after count arithmetic accurate.
        deduped: dict[tuple[str, str, str], CodeRelationship] = {}
        for r in relationships:
            key = (r.source_id, r.target_id, r.relationship_type.value)
            deduped[key] = r
        unique_rels = list(deduped.values())

        # Pre-count existing edges so we can report an accurate `created` count.
        # executemany doesn't preserve RETURNING output, so we compute it by
        # counting existing rows before the insert and subtracting from after.
        sources = [r.source_id for r in unique_rels]
        targets = [r.target_id for r in unique_rels]
        ref_types = [r.relationship_type.value for r in unique_rels]

        before_count = await conn.fetchval("""
            SELECT COUNT(*) FROM code_references
            WHERE (source_symbol_id, target_symbol_id, ref_type) IN (
                SELECT * FROM unnest($1::text[], $2::text[], $3::text[])
            )
        """, sources, targets, ref_types)

        # Batch upsert with EXISTS guards. Pipelined into a single write.
        # INSERT...SELECT...WHERE EXISTS silently skips edges whose endpoints
        # aren't in code_symbols (mirrors Neo4j's MATCH...MATCH...MERGE — Java
        # external superclasses, C# external::Base types, deleted-file refs
        # all drop silently instead of aborting the transaction).
        await conn.executemany("""
            INSERT INTO code_references (source_symbol_id, target_symbol_id, ref_type, properties)
            SELECT $1, $2, $3, $4::jsonb
            WHERE EXISTS (SELECT 1 FROM code_symbols WHERE id = $1)
              AND EXISTS (SELECT 1 FROM code_symbols WHERE id = $2)
            ON CONFLICT (source_symbol_id, target_symbol_id, ref_type) DO NOTHING
        """, [
            (r.source_id, r.target_id, r.relationship_type.value, json.dumps(r.properties or {}))
            for r in unique_rels
        ])

        after_count = await conn.fetchval("""
            SELECT COUNT(*) FROM code_references
            WHERE (source_symbol_id, target_symbol_id, ref_type) IN (
                SELECT * FROM unnest($1::text[], $2::text[], $3::text[])
            )
        """, sources, targets, ref_types)
        return after_count - before_count

    async def _upsert_embeddings(self, conn: asyncpg.Connection, entities: list[CodeEntity]) -> None:
        embeddable = [e for e in entities if e.entity_type in _EMBEDDABLE_TYPES and e.embedding]
        if not embeddable:
            return
        await conn.executemany("""
            INSERT INTO code_embeddings (symbol_id, embedding, content_hash)
            VALUES ($1, $2, $3)
            ON CONFLICT (symbol_id) DO UPDATE SET
                embedding    = EXCLUDED.embedding,
                content_hash = EXCLUDED.content_hash
        """, [
            (e.id, e.embedding, e.content_hash)
            for e in embeddable
        ])

    # ── Stats ────────────────────────────────────────────────────────────

    async def count_repository_entities(self, repository: str) -> int:
        pool = self._require_pool()
        return await pool.fetchval(
            "SELECT COUNT(*) FROM code_symbols WHERE repository = $1", repository
        )
