"""Tests that WriteBackend ABC enforces the correct interface."""
import pytest
from constellation.graph.base import WriteBackend


def test_write_backend_cannot_be_instantiated_directly():
    """WriteBackend is abstract — direct instantiation must fail."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        WriteBackend()  # type: ignore


def test_concrete_write_backend_missing_method_raises():
    """A class that doesn't implement all abstract methods can't be instantiated."""
    class Incomplete(WriteBackend):
        pass  # implements nothing

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()


def test_concrete_write_backend_all_methods_implemented():
    """A class that implements all abstract methods can be instantiated."""
    class FakeBackend(WriteBackend):
        async def connect(self): pass
        async def close(self): pass
        async def initialize_schema(self): pass
        async def upsert_repository(self, name, source, commit_sha, entity_count): pass
        async def get_repository(self, name): return None
        async def list_repositories(self): return []
        async def delete_repository(self, name): pass
        async def get_file_hashes(self, repository): return {}
        async def apply_indexing_changes(self, *, repository, source, commit_sha,
                                          reindex_preparations, entities,
                                          relationships, stale_file_paths): return (0, 0, 0)
        async def count_repository_entities(self, repository): return 0

    backend = FakeBackend()
    assert backend is not None
