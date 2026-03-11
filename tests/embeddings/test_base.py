import pytest

from constellation.models import CodeEntity, EntityType
from constellation.embeddings.base import (
    EMBEDDABLE_TYPES,
    MAX_EMBEDDING_TEXT_LENGTH,
    BaseEmbeddingProvider,
    is_embeddable,
    prepare_embedding_text,
)


class TestIsEmbeddable:
    def test_method_is_embeddable(self):
        assert is_embeddable(EntityType.METHOD) is True

    def test_class_is_embeddable(self):
        assert is_embeddable(EntityType.CLASS) is True

    def test_interface_is_embeddable(self):
        assert is_embeddable(EntityType.INTERFACE) is True

    def test_constructor_is_embeddable(self):
        assert is_embeddable(EntityType.CONSTRUCTOR) is True

    def test_file_not_embeddable(self):
        assert is_embeddable(EntityType.FILE) is False

    def test_package_not_embeddable(self):
        assert is_embeddable(EntityType.PACKAGE) is False

    def test_field_not_embeddable(self):
        assert is_embeddable(EntityType.FIELD) is False

    def test_embeddable_types_matches_expected_set(self):
        expected = {EntityType.METHOD, EntityType.CLASS, EntityType.INTERFACE, EntityType.CONSTRUCTOR}
        assert EMBEDDABLE_TYPES == expected


def _make_entity(**kwargs) -> CodeEntity:
    defaults = dict(
        id="repo::com.example.Foo.bar(String)",
        name="bar",
        entity_type=EntityType.METHOD,
        repository="repo",
        file_path="src/Foo.java",
        line_number=10,
        language="java",
    )
    defaults.update(kwargs)
    return CodeEntity(**defaults)


class TestPrepareEmbeddingText:
    def test_includes_entity_type_label(self):
        entity = _make_entity()
        text = prepare_embedding_text(entity)
        assert "[Method]" in text

    def test_includes_entity_id(self):
        entity = _make_entity(id="repo::com.example.Foo.bar(String)")
        text = prepare_embedding_text(entity)
        assert "repo::com.example.Foo.bar(String)" in text

    def test_includes_signature_when_present(self):
        entity = _make_entity(signature="public int bar(String s)")
        text = prepare_embedding_text(entity)
        assert "Signature: public int bar(String s)" in text

    def test_includes_stereotypes_when_present(self):
        entity = _make_entity(stereotypes=["endpoint", "getter"])
        text = prepare_embedding_text(entity)
        assert "Stereotypes: endpoint, getter" in text

    def test_includes_docstring_when_present(self):
        entity = _make_entity(docstring="Does bar things")
        text = prepare_embedding_text(entity)
        assert "Docstring: Does bar things" in text

    def test_includes_code_when_present(self):
        entity = _make_entity(code="public int bar(String s) { return 0; }")
        text = prepare_embedding_text(entity)
        assert "Code: public int bar(String s) { return 0; }" in text

    def test_omits_none_fields(self):
        entity = _make_entity(signature=None, docstring=None, code=None)
        text = prepare_embedding_text(entity)
        assert "Signature:" not in text
        assert "Docstring:" not in text
        assert "Code:" not in text

    def test_truncates_text_longer_than_max(self):
        long_code = "x" * (MAX_EMBEDDING_TEXT_LENGTH + 500)
        entity = _make_entity(code=long_code)
        text = prepare_embedding_text(entity)
        assert len(text) == MAX_EMBEDDING_TEXT_LENGTH

    def test_class_entity_shows_class_label(self):
        entity = _make_entity(entity_type=EntityType.CLASS, name="Foo", id="repo::Foo")
        text = prepare_embedding_text(entity)
        assert "[Class]" in text

    def test_constructor_entity_shows_constructor_label(self):
        entity = _make_entity(entity_type=EntityType.CONSTRUCTOR, name="Foo", id="repo::Foo.<init>()")
        text = prepare_embedding_text(entity)
        assert "[Constructor]" in text

    def test_empty_stereotypes_list_omits_line(self):
        entity = _make_entity(stereotypes=[])
        text = prepare_embedding_text(entity)
        assert "Stereotypes:" not in text
