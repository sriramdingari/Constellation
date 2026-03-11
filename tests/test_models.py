import pytest
from pydantic import ValidationError

from constellation.models import (
    CodeEntity, CodeRelationship, EntityType, RelationshipType,
)


class TestEntityType:
    def test_all_values_exist(self):
        assert EntityType.FILE == "File"
        assert EntityType.PACKAGE == "Package"
        assert EntityType.CLASS == "Class"
        assert EntityType.INTERFACE == "Interface"
        assert EntityType.METHOD == "Method"
        assert EntityType.CONSTRUCTOR == "Constructor"
        assert EntityType.FIELD == "Field"
        assert EntityType.HOOK == "Hook"
        assert EntityType.REFERENCE == "Reference"

    def test_total_count(self):
        assert len(EntityType) == 9


class TestRelationshipType:
    def test_structural(self):
        assert RelationshipType.CONTAINS == "CONTAINS"
        assert RelationshipType.IN_PACKAGE == "IN_PACKAGE"
        assert RelationshipType.HAS_METHOD == "HAS_METHOD"
        assert RelationshipType.HAS_CONSTRUCTOR == "HAS_CONSTRUCTOR"
        assert RelationshipType.HAS_FIELD == "HAS_FIELD"
        assert RelationshipType.DECLARES == "DECLARES"

    def test_inheritance(self):
        assert RelationshipType.EXTENDS == "EXTENDS"
        assert RelationshipType.IMPLEMENTS == "IMPLEMENTS"
        assert RelationshipType.OVERRIDES == "OVERRIDES"

    def test_call_graph(self):
        assert RelationshipType.CALLS == "CALLS"
        assert RelationshipType.USES_TYPE == "USES_TYPE"

    def test_import(self):
        assert RelationshipType.IMPORTS == "IMPORTS"
        assert RelationshipType.EXPORTS == "EXPORTS"

    def test_framework(self):
        assert RelationshipType.DEPENDS_ON == "DEPENDS_ON"
        assert RelationshipType.PRODUCES_BEAN == "PRODUCES_BEAN"
        assert RelationshipType.THROWS == "THROWS"

    def test_react(self):
        assert RelationshipType.RENDERS == "RENDERS"
        assert RelationshipType.USES_HOOK == "USES_HOOK"
        assert RelationshipType.PROVIDES_CONTEXT == "PROVIDES_CONTEXT"
        assert RelationshipType.CONSUMES_CONTEXT == "CONSUMES_CONTEXT"

    def test_total_count(self):
        assert len(RelationshipType) == 20


class TestCodeEntity:
    def test_minimal_creation(self):
        entity = CodeEntity(
            id="repo::Foo", name="Foo", entity_type=EntityType.CLASS,
            repository="repo", file_path="Foo.java", line_number=1, language="java",
        )
        assert entity.id == "repo::Foo"
        assert entity.name == "Foo"
        assert entity.entity_type == EntityType.CLASS

    def test_optional_fields_default_none(self):
        entity = CodeEntity(
            id="repo::Foo", name="Foo", entity_type=EntityType.CLASS,
            repository="repo", file_path="Foo.java", line_number=1, language="java",
        )
        assert entity.line_end is None
        assert entity.code is None
        assert entity.signature is None
        assert entity.return_type is None
        assert entity.docstring is None
        assert entity.content_hash is None
        assert entity.embedding is None

    def test_list_fields_default_empty(self):
        entity = CodeEntity(
            id="repo::Foo", name="Foo", entity_type=EntityType.CLASS,
            repository="repo", file_path="Foo.java", line_number=1, language="java",
        )
        assert entity.modifiers == []
        assert entity.stereotypes == []
        assert entity.properties == {}

    def test_full_entity_creation(self):
        entity = CodeEntity(
            id="repo::com.example.Foo.bar(String,int)",
            name="bar", entity_type=EntityType.METHOD,
            repository="repo", file_path="src/Foo.java",
            line_number=10, line_end=20, language="java",
            code="public int bar(String s, int n) { return n; }",
            signature="public int bar(String s, int n)",
            return_type="int", docstring="Does bar things",
            modifiers=["public"], stereotypes=["endpoint"],
            properties={"http_method": "GET"},
            embedding=[0.1, 0.2, 0.3],
        )
        assert entity.line_end == 20
        assert entity.return_type == "int"
        assert entity.embedding == [0.1, 0.2, 0.3]
        assert entity.properties["http_method"] == "GET"

    def test_file_entity_with_content_hash(self):
        entity = CodeEntity(
            id="repo::src/main.py", name="main.py", entity_type=EntityType.FILE,
            repository="repo", file_path="src/main.py", line_number=0,
            language="python", content_hash="abc123",
        )
        assert entity.content_hash == "abc123"

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            CodeEntity(id="repo::Foo", name="Foo")

    def test_list_defaults_are_independent(self):
        e1 = CodeEntity(
            id="repo::A", name="A", entity_type=EntityType.CLASS,
            repository="r", file_path="a.py", line_number=1, language="python",
        )
        e2 = CodeEntity(
            id="repo::B", name="B", entity_type=EntityType.CLASS,
            repository="r", file_path="b.py", line_number=1, language="python",
        )
        e1.modifiers.append("public")
        assert e2.modifiers == []


class TestCodeRelationship:
    def test_creation(self):
        rel = CodeRelationship(
            source_id="repo::Foo.bar()",
            target_id="repo::Baz.qux()",
            relationship_type=RelationshipType.CALLS,
        )
        assert rel.source_id == "repo::Foo.bar()"
        assert rel.target_id == "repo::Baz.qux()"
        assert rel.relationship_type == RelationshipType.CALLS

    def test_properties_default_empty(self):
        rel = CodeRelationship(
            source_id="a", target_id="b",
            relationship_type=RelationshipType.CONTAINS,
        )
        assert rel.properties == {}

    def test_with_properties(self):
        rel = CodeRelationship(
            source_id="a", target_id="b",
            relationship_type=RelationshipType.CALLS,
            properties={"http_method": "GET", "route": "/users"},
        )
        assert rel.properties["http_method"] == "GET"

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            CodeRelationship(source_id="a", target_id="b")
