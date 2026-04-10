"""Abstract base class for Constellation write backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from constellation.models import CodeEntity, CodeRelationship


class WriteBackend(ABC):
    """Interface for all Constellation storage write backends.

    Implementations: Neo4jWriteBackend, PostgresWriteBackend.
    """

    # -- Lifecycle ------------------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """Open the database connection."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the database connection."""
        ...

    @abstractmethod
    async def initialize_schema(self) -> None:
        """Create tables/constraints/indexes. Must be idempotent (IF NOT EXISTS).
        For Postgres: also runs CREATE EXTENSION IF NOT EXISTS vector.
        """
        ...

    # -- Repository -----------------------------------------------------------

    @abstractmethod
    async def upsert_repository(
        self,
        name: str,
        source: str,
        commit_sha: str | None,
        entity_count: int,
    ) -> None:
        """Insert or update repository metadata."""
        ...

    @abstractmethod
    async def get_repository(self, name: str) -> dict | None:
        """Return repository metadata dict or None if not found."""
        ...

    @abstractmethod
    async def list_repositories(self) -> list[dict]:
        """Return all indexed repositories."""
        ...

    @abstractmethod
    async def delete_repository(self, name: str) -> None:
        """Delete a repository and all its entities (cascade)."""
        ...

    # -- Change detection -----------------------------------------------------

    @abstractmethod
    async def get_file_hashes(self, repository: str) -> dict[str, str]:
        """Return {file_path: content_hash} for all files in the repository."""
        ...

    # -- Indexing (atomic) ----------------------------------------------------

    @abstractmethod
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
        """Atomically write an indexing run.

        Returns (entities_created, relationships_created, total_entity_count).
        """
        ...

    async def apply_spooled_indexing_changes(
        self,
        *,
        spool_dir: Path,
    ) -> tuple[int, int, int]:
        """Atomically replay pre-prepared spool files.

        Only backends that support chunked spool-and-replay need to
        override this.  The default raises NotImplementedError so callers
        that check storage_backend before calling are safe, and any
        future backend that forgets to implement it fails loudly.

        Returns (entities_created, relationships_created, total_entity_count).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support spooled indexing"
        )

    # -- Stats ----------------------------------------------------------------

    @abstractmethod
    async def count_repository_entities(self, repository: str) -> int:
        """Return total entity count for a repository."""
        ...
