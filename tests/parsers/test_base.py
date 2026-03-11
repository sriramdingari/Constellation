"""Tests for constellation.parsers.base module."""

from pathlib import Path

import pytest

from constellation.models import CodeEntity, CodeRelationship, EntityType, RelationshipType
from constellation.parsers.base import BaseParser, ParseResult


# --- Stub parser for testing BaseParser ABC ---

class StubParser(BaseParser):
    """Concrete implementation of BaseParser for testing."""

    @property
    def language(self) -> str:
        return "python"

    @property
    def file_extensions(self) -> list[str]:
        return [".py", ".pyi"]

    def parse_file(self, file_path: Path, repository: str) -> ParseResult:
        result = ParseResult(file_path=str(file_path), language=self.language)
        return result


# --- Fixtures ---

def _make_entity(**overrides) -> CodeEntity:
    defaults = dict(
        id="ent-1",
        name="MyClass",
        entity_type=EntityType.CLASS,
        repository="repo",
        file_path="src/main.py",
        line_number=1,
        language="python",
    )
    defaults.update(overrides)
    return CodeEntity(**defaults)


def _make_relationship(**overrides) -> CodeRelationship:
    defaults = dict(
        source_id="ent-1",
        target_id="ent-2",
        relationship_type=RelationshipType.CALLS,
    )
    defaults.update(overrides)
    return CodeRelationship(**defaults)


# --- ParseResult tests ---


class TestParseResultDefaults:
    """ParseResult should initialise with sensible defaults."""

    def test_entities_default_empty(self):
        result = ParseResult(file_path="a.py", language="python")
        assert result.entities == []

    def test_relationships_default_empty(self):
        result = ParseResult(file_path="a.py", language="python")
        assert result.relationships == []

    def test_errors_default_empty(self):
        result = ParseResult(file_path="a.py", language="python")
        assert result.errors == []

    def test_success_true_when_no_errors(self):
        result = ParseResult(file_path="a.py", language="python")
        assert result.success is True


class TestParseResultSuccess:
    """success property tracks error state."""

    def test_success_false_when_errors_added(self):
        result = ParseResult(file_path="a.py", language="python")
        result.add_error("something broke")
        assert result.success is False

    def test_success_false_with_multiple_errors(self):
        result = ParseResult(file_path="a.py", language="python")
        result.add_error("error 1")
        result.add_error("error 2")
        assert result.success is False
        assert len(result.errors) == 2


class TestParseResultAddEntity:
    """add_entity appends to entities list."""

    def test_add_single_entity(self):
        result = ParseResult(file_path="a.py", language="python")
        entity = _make_entity()
        result.add_entity(entity)
        assert len(result.entities) == 1
        assert result.entities[0] is entity

    def test_add_multiple_entities(self):
        result = ParseResult(file_path="a.py", language="python")
        e1 = _make_entity(id="e1")
        e2 = _make_entity(id="e2")
        result.add_entity(e1)
        result.add_entity(e2)
        assert len(result.entities) == 2


class TestParseResultAddRelationship:
    """add_relationship appends to relationships list."""

    def test_add_single_relationship(self):
        result = ParseResult(file_path="a.py", language="python")
        rel = _make_relationship()
        result.add_relationship(rel)
        assert len(result.relationships) == 1
        assert result.relationships[0] is rel

    def test_add_multiple_relationships(self):
        result = ParseResult(file_path="a.py", language="python")
        r1 = _make_relationship(source_id="a", target_id="b")
        r2 = _make_relationship(source_id="c", target_id="d")
        result.add_relationship(r1)
        result.add_relationship(r2)
        assert len(result.relationships) == 2


class TestParseResultAddError:
    """add_error appends to errors list."""

    def test_add_error(self):
        result = ParseResult(file_path="a.py", language="python")
        result.add_error("kaboom")
        assert result.errors == ["kaboom"]

    def test_add_multiple_errors(self):
        result = ParseResult(file_path="a.py", language="python")
        result.add_error("err1")
        result.add_error("err2")
        result.add_error("err3")
        assert result.errors == ["err1", "err2", "err3"]


class TestParseResultInstanceIsolation:
    """Lists must be independent between instances (no shared mutable defaults)."""

    def test_entities_independent(self):
        r1 = ParseResult(file_path="a.py", language="python")
        r2 = ParseResult(file_path="b.py", language="python")
        r1.add_entity(_make_entity(id="only-in-r1"))
        assert r1.entities != r2.entities
        assert len(r2.entities) == 0

    def test_relationships_independent(self):
        r1 = ParseResult(file_path="a.py", language="python")
        r2 = ParseResult(file_path="b.py", language="python")
        r1.add_relationship(_make_relationship())
        assert len(r2.relationships) == 0

    def test_errors_independent(self):
        r1 = ParseResult(file_path="a.py", language="python")
        r2 = ParseResult(file_path="b.py", language="python")
        r1.add_error("only-in-r1")
        assert len(r2.errors) == 0


# --- BaseParser tests (via StubParser) ---


class TestBaseParserCanParse:
    """can_parse should match on file extensions."""

    def test_matching_extension(self):
        parser = StubParser()
        assert parser.can_parse(Path("src/main.py")) is True

    def test_matching_alternate_extension(self):
        parser = StubParser()
        assert parser.can_parse(Path("stubs/types.pyi")) is True

    def test_non_matching_extension(self):
        parser = StubParser()
        assert parser.can_parse(Path("src/main.java")) is False

    def test_no_extension(self):
        parser = StubParser()
        assert parser.can_parse(Path("Makefile")) is False


class TestBaseParserProperties:
    """language and file_extensions are abstract properties."""

    def test_language(self):
        parser = StubParser()
        assert parser.language == "python"

    def test_file_extensions(self):
        parser = StubParser()
        assert ".py" in parser.file_extensions
        assert ".pyi" in parser.file_extensions


class TestBaseParserParseFile:
    """parse_file returns a ParseResult."""

    def test_returns_parse_result(self):
        parser = StubParser()
        result = parser.parse_file(Path("src/main.py"), repository="my-repo")
        assert isinstance(result, ParseResult)

    def test_parse_result_has_correct_file_path(self):
        parser = StubParser()
        result = parser.parse_file(Path("src/main.py"), repository="my-repo")
        assert result.file_path == "src/main.py"

    def test_parse_result_has_correct_language(self):
        parser = StubParser()
        result = parser.parse_file(Path("src/main.py"), repository="my-repo")
        assert result.language == "python"
