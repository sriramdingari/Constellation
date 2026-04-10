from pathlib import Path

from constellation.indexer.spool import (
    ChunkPreparation,
    RunManifest,
    SpoolChunkPaths,
    create_spool_dir,
    load_chunk_preparation,
    load_run_manifest,
    write_chunk_preparation,
    write_run_manifest,
)
from constellation.models import CodeEntity, CodeRelationship, EntityType, RelationshipType


def test_chunk_and_manifest_round_trip(tmp_path: Path):
    spool_dir = create_spool_dir(tmp_path, run_id="run-123")
    chunk_paths = SpoolChunkPaths.for_chunk(spool_dir, 1)

    entity = CodeEntity(
        id="repo::src/a.py",
        name="a.py",
        entity_type=EntityType.FILE,
        repository="repo",
        file_path="src/a.py",
        line_number=1,
        language="python",
        content_hash="hash-a",
    )
    relationship = CodeRelationship(
        source_id="repo::src/a.py",
        target_id="repo::src/a.py#Foo",
        relationship_type=RelationshipType.CONTAINS,
    )
    chunk = ChunkPreparation(
        chunk_index=1,
        files=[("src/a.py", "hash-a", True)],
        reindex_preparations=[("src/a.py", {"repo::src/a.py", "repo::src/a.py#Foo"})],
        entities=[entity],
        relationships=[relationship],
    )
    manifest = RunManifest(
        run_id="run-123",
        repository="repo",
        source="/tmp/repo",
        commit_sha="abc123",
        stale_file_paths=["src/old.py"],
        chunk_indices=[1],
        files_total=1,
        files_processed=1,
        files_skipped=0,
    )

    write_chunk_preparation(chunk_paths, chunk)
    write_run_manifest(spool_dir / "run_manifest.json", manifest)

    loaded_chunk = load_chunk_preparation(chunk_paths)
    loaded_manifest = load_run_manifest(spool_dir / "run_manifest.json")

    assert loaded_chunk.chunk_index == 1
    assert loaded_chunk.files == [("src/a.py", "hash-a", True)]
    assert loaded_chunk.reindex_preparations == [("src/a.py", {"repo::src/a.py", "repo::src/a.py#Foo"})]
    assert [e.id for e in loaded_chunk.entities] == ["repo::src/a.py"]
    assert [r.relationship_type for r in loaded_chunk.relationships] == [RelationshipType.CONTAINS]
    assert loaded_manifest.repository == "repo"
    assert loaded_manifest.chunk_indices == [1]
