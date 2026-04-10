from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from constellation.models import CodeEntity, CodeRelationship

# NOTE: CodeEntity and CodeRelationship are Pydantic BaseModel subclasses,
# NOT Python dataclasses. Use .model_dump() for serialization, not asdict().


@dataclass(frozen=True)
class SpoolChunkPaths:
    chunk_dir: Path
    entities_path: Path
    relationships_path: Path
    preparations_path: Path

    @classmethod
    def for_chunk(cls, spool_dir: Path, chunk_index: int) -> "SpoolChunkPaths":
        chunk_dir = spool_dir / f"chunk-{chunk_index:05d}"
        return cls(
            chunk_dir=chunk_dir,
            entities_path=chunk_dir / "entities.jsonl",
            relationships_path=chunk_dir / "relationships.jsonl",
            preparations_path=chunk_dir / "reindex_preparations.json",
        )


@dataclass
class ChunkPreparation:
    chunk_index: int
    files: list[tuple[str, str, bool]]
    reindex_preparations: list[tuple[str, set[str]]]
    entities: list[CodeEntity]
    relationships: list[CodeRelationship]


@dataclass
class RunManifest:
    run_id: str
    repository: str
    source: str
    commit_sha: str | None
    stale_file_paths: list[str]
    chunk_indices: list[int]
    files_total: int
    files_processed: int
    files_skipped: int


def create_spool_dir(root: Path, run_id: str) -> Path:
    spool_dir = root / run_id
    spool_dir.mkdir(parents=True, exist_ok=False)
    return spool_dir


def cleanup_spool_dir(spool_dir: Path) -> None:
    shutil.rmtree(spool_dir, ignore_errors=True)


def _entity_to_dict(entity: CodeEntity) -> dict:
    return entity.model_dump()


def _relationship_to_dict(relationship: CodeRelationship) -> dict:
    return relationship.model_dump()


def _dict_to_entity(payload: dict) -> CodeEntity:
    return CodeEntity(**payload)


def _dict_to_relationship(payload: dict) -> CodeRelationship:
    return CodeRelationship(**payload)


def write_chunk_preparation(paths: SpoolChunkPaths, chunk: ChunkPreparation) -> None:
    paths.chunk_dir.mkdir(parents=True, exist_ok=False)
    with paths.entities_path.open("w", encoding="utf-8") as fh:
        for entity in chunk.entities:
            fh.write(json.dumps(_entity_to_dict(entity)) + "\n")
    with paths.relationships_path.open("w", encoding="utf-8") as fh:
        for relationship in chunk.relationships:
            fh.write(json.dumps(_relationship_to_dict(relationship)) + "\n")
    serializable_preparations = [
        [file_path, sorted(entity_ids)]
        for file_path, entity_ids in chunk.reindex_preparations
    ]
    paths.preparations_path.write_text(
        json.dumps(
            {
                "chunk_index": chunk.chunk_index,
                "files": chunk.files,
                "reindex_preparations": serializable_preparations,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def load_chunk_preparation(paths: SpoolChunkPaths) -> ChunkPreparation:
    entities = [
        _dict_to_entity(json.loads(line))
        for line in paths.entities_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    relationships = [
        _dict_to_relationship(json.loads(line))
        for line in paths.relationships_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    payload = json.loads(paths.preparations_path.read_text(encoding="utf-8"))
    reindex_preparations = [
        (file_path, set(entity_ids))
        for file_path, entity_ids in payload["reindex_preparations"]
    ]
    return ChunkPreparation(
        chunk_index=payload["chunk_index"],
        files=[tuple(item) for item in payload["files"]],
        reindex_preparations=reindex_preparations,
        entities=entities,
        relationships=relationships,
    )


def write_run_manifest(path: Path, manifest: RunManifest) -> None:
    path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8")


def load_run_manifest(path: Path) -> RunManifest:
    return RunManifest(**json.loads(path.read_text(encoding="utf-8")))
