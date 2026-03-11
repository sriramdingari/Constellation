"""Tests for constellation.parsers.javascript module."""

from pathlib import Path

import pytest

from constellation.models import EntityType, RelationshipType
from constellation.parsers.javascript import JavaScriptParser


FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_TSX = FIXTURES / "sample_javascript.tsx"
SAMPLE_EDGE = FIXTURES / "sample_javascript_edge_cases.ts"
REPOSITORY = "test-repo"


@pytest.fixture(scope="module")
def parser():
    return JavaScriptParser()


@pytest.fixture(scope="module")
def tsx_result(parser):
    return parser.parse_file(SAMPLE_TSX, repository=REPOSITORY)


@pytest.fixture(scope="module")
def edge_result(parser):
    return parser.parse_file(SAMPLE_EDGE, repository=REPOSITORY)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _entities_by_type(result, entity_type):
    return [e for e in result.entities if e.entity_type == entity_type]


def _entity_by_name(result, name):
    for e in result.entities:
        if e.name == name:
            return e
    return None


def _rels_by_type(result, rel_type):
    return [r for r in result.relationships if r.relationship_type == rel_type]


def _rels_from(result, source_id, rel_type):
    return [
        r for r in result.relationships
        if r.source_id == source_id and r.relationship_type == rel_type
    ]


# ===========================================================================
# Parser properties
# ===========================================================================


class TestParserProperties:
    """Parser should report correct language and extensions."""

    def test_language_is_javascript(self, parser):
        assert parser.language == "javascript"

    def test_extensions_include_js(self, parser):
        assert ".js" in parser.file_extensions

    def test_extensions_include_jsx(self, parser):
        assert ".jsx" in parser.file_extensions

    def test_extensions_include_ts(self, parser):
        assert ".ts" in parser.file_extensions

    def test_extensions_include_tsx(self, parser):
        assert ".tsx" in parser.file_extensions

    def test_can_parse_ts_file(self, parser):
        assert parser.can_parse(Path("app.ts")) is True

    def test_can_parse_tsx_file(self, parser):
        assert parser.can_parse(Path("app.tsx")) is True

    def test_cannot_parse_python(self, parser):
        assert parser.can_parse(Path("app.py")) is False


# ===========================================================================
# TSX fixture: class extraction
# ===========================================================================


class TestClassExtraction:
    """Classes should be extracted as CLASS entities."""

    def test_class_found(self, tsx_result):
        classes = _entities_by_type(tsx_result, EntityType.CLASS)
        names = [c.name for c in classes]
        assert "UserService" in names

    def test_class_entity_id_format(self, tsx_result):
        entity = _entity_by_name(tsx_result, "UserService")
        assert entity is not None
        assert entity.id.startswith(f"{REPOSITORY}::")

    def test_class_has_line_number(self, tsx_result):
        entity = _entity_by_name(tsx_result, "UserService")
        assert entity is not None
        assert entity.line_number > 0


# ===========================================================================
# TSX fixture: TypeScript interface -> INTERFACE entity
# ===========================================================================


class TestInterfaceExtraction:
    """TypeScript interfaces should become INTERFACE entities."""

    def test_interface_found(self, tsx_result):
        interfaces = _entities_by_type(tsx_result, EntityType.INTERFACE)
        names = [i.name for i in interfaces]
        assert "UserProps" in names

    def test_interface_entity_type(self, tsx_result):
        entity = _entity_by_name(tsx_result, "UserProps")
        assert entity is not None
        assert entity.entity_type == EntityType.INTERFACE

    def test_interface_entity_id_format(self, tsx_result):
        entity = _entity_by_name(tsx_result, "UserProps")
        assert entity is not None
        assert REPOSITORY in entity.id


# ===========================================================================
# Edge cases: TypeScript namespace -> PACKAGE entity
# ===========================================================================


class TestNamespaceExtraction:
    """TypeScript namespace should become a PACKAGE entity."""

    def test_namespace_found(self, edge_result):
        packages = _entities_by_type(edge_result, EntityType.PACKAGE)
        names = [p.name for p in packages]
        assert "Validators" in names

    def test_namespace_entity_type(self, edge_result):
        entity = _entity_by_name(edge_result, "Validators")
        assert entity is not None
        assert entity.entity_type == EntityType.PACKAGE


# ===========================================================================
# TSX fixture: method extraction from classes
# ===========================================================================


class TestMethodExtraction:
    """Methods inside classes should be extracted."""

    def test_methods_found(self, tsx_result):
        methods = _entities_by_type(tsx_result, EntityType.METHOD)
        names = [m.name for m in methods]
        assert "fetchUser" in names
        assert "deleteUser" in names

    def test_method_entity_id_includes_class(self, tsx_result):
        entity = _entity_by_name(tsx_result, "fetchUser")
        assert entity is not None
        # Entity ID should include the class name
        assert "UserService" in entity.id

    def test_method_entity_id_format(self, tsx_result):
        entity = _entity_by_name(tsx_result, "fetchUser")
        assert entity is not None
        # Format: repository::module.Class.method
        assert entity.id.startswith(f"{REPOSITORY}::")
        assert "UserService.fetchUser" in entity.id


# ===========================================================================
# TSX fixture: top-level function extraction
# ===========================================================================


class TestTopLevelFunctionExtraction:
    """Top-level functions should be extracted as METHOD entities."""

    def test_function_found(self, tsx_result):
        entity = _entity_by_name(tsx_result, "formatUserName")
        assert entity is not None
        assert entity.entity_type == EntityType.METHOD

    def test_function_return_type(self, tsx_result):
        entity = _entity_by_name(tsx_result, "formatUserName")
        assert entity is not None
        assert entity.return_type is not None
        assert "string" in entity.return_type


# ===========================================================================
# TSX fixture: arrow function const extraction -> METHOD
# ===========================================================================


class TestArrowFunctionExtraction:
    """Arrow functions assigned to const should be extracted as METHOD entities."""

    def test_arrow_function_found(self, tsx_result):
        entity = _entity_by_name(tsx_result, "UserCard")
        assert entity is not None
        assert entity.entity_type == EntityType.METHOD

    def test_arrow_function_in_edge_cases(self, edge_result):
        entity = _entity_by_name(edge_result, "calculateTotal")
        assert entity is not None
        assert entity.entity_type == EntityType.METHOD

    def test_arrow_function_return_type(self, edge_result):
        entity = _entity_by_name(edge_result, "calculateTotal")
        assert entity is not None
        assert entity.return_type is not None
        assert "number" in entity.return_type


# ===========================================================================
# Edge cases: async modifier detection
# ===========================================================================


class TestAsyncModifier:
    """Async methods and functions should have 'async' modifier."""

    def test_async_class_method(self, tsx_result):
        entity = _entity_by_name(tsx_result, "fetchUser")
        assert entity is not None
        assert "async" in entity.modifiers

    def test_async_arrow_function(self, edge_result):
        entity = _entity_by_name(edge_result, "fetchData")
        assert entity is not None
        assert "async" in entity.modifiers

    def test_async_method_in_exported_class(self, edge_result):
        entity = _entity_by_name(edge_result, "get")
        assert entity is not None
        assert "async" in entity.modifiers


# ===========================================================================
# TSX fixture: EXPORTS relationships
# ===========================================================================


class TestExportsRelationships:
    """EXPORTS relationships should be created for exported symbols."""

    def test_named_export_relationship(self, tsx_result):
        exports = _rels_by_type(tsx_result, RelationshipType.EXPORTS)
        target_ids = [r.target_id for r in exports]
        # formatUserName is named-exported
        matching = [t for t in target_ids if "formatUserName" in t]
        assert len(matching) > 0

    def test_default_export_relationship(self, tsx_result):
        exports = _rels_by_type(tsx_result, RelationshipType.EXPORTS)
        target_ids = [r.target_id for r in exports]
        # UserCard is default-exported
        matching = [t for t in target_ids if "UserCard" in t]
        assert len(matching) > 0

    def test_export_class(self, edge_result):
        exports = _rels_by_type(edge_result, RelationshipType.EXPORTS)
        target_ids = [r.target_id for r in exports]
        matching = [t for t in target_ids if "ApiClient" in t]
        assert len(matching) > 0

    def test_export_default_function(self, edge_result):
        exports = _rels_by_type(edge_result, RelationshipType.EXPORTS)
        target_ids = [r.target_id for r in exports]
        matching = [t for t in target_ids if "createClient" in t]
        assert len(matching) > 0


# ===========================================================================
# TSX fixture: USES_HOOK relationships
# ===========================================================================


class TestUsesHookRelationships:
    """USES_HOOK relationships should be created for React hook calls."""

    def test_usestate_hook(self, tsx_result):
        hooks = _rels_by_type(tsx_result, RelationshipType.USES_HOOK)
        target_ids = [r.target_id for r in hooks]
        matching = [t for t in target_ids if "useState" in t]
        assert len(matching) > 0

    def test_usecontext_hook(self, tsx_result):
        hooks = _rels_by_type(tsx_result, RelationshipType.USES_HOOK)
        target_ids = [r.target_id for r in hooks]
        matching = [t for t in target_ids if "useContext" in t]
        assert len(matching) > 0

    def test_hook_source_is_component(self, tsx_result):
        hooks = _rels_by_type(tsx_result, RelationshipType.USES_HOOK)
        # Hooks should originate from the UserCard component
        source_ids = [r.source_id for r in hooks]
        matching = [s for s in source_ids if "UserCard" in s]
        assert len(matching) > 0


# ===========================================================================
# Edge cases: test stereotype detection
# ===========================================================================


class TestTestStereotype:
    """describe/it/test calls should mark functions with test stereotype."""

    def test_describe_stereotype(self, edge_result):
        # The file has describe() calls -- look for test stereotypes
        methods = _entities_by_type(edge_result, EntityType.METHOD)
        test_methods = [m for m in methods if "test" in m.stereotypes]
        # There should be at least one method with test stereotype
        assert len(test_methods) > 0


# ===========================================================================
# Edge cases: constructor extraction
# ===========================================================================


class TestConstructorExtraction:
    """Constructors should be extracted as CONSTRUCTOR entities."""

    def test_constructor_found(self, edge_result):
        constructors = _entities_by_type(edge_result, EntityType.CONSTRUCTOR)
        assert len(constructors) > 0

    def test_constructor_in_class(self, edge_result):
        constructors = _entities_by_type(edge_result, EntityType.CONSTRUCTOR)
        # Should have constructor from ApiClient
        ids = [c.id for c in constructors]
        matching = [i for i in ids if "ApiClient" in i]
        assert len(matching) > 0

    def test_constructor_in_tsx_class(self, tsx_result):
        constructors = _entities_by_type(tsx_result, EntityType.CONSTRUCTOR)
        assert len(constructors) > 0
        ids = [c.id for c in constructors]
        matching = [i for i in ids if "UserService" in i]
        assert len(matching) > 0


# ===========================================================================
# CONTAINS relationships (File -> Class, File -> top-level Method)
# ===========================================================================


class TestContainsRelationships:
    """File entity should CONTAINS classes and top-level methods."""

    def test_file_contains_class(self, tsx_result):
        contains = _rels_by_type(tsx_result, RelationshipType.CONTAINS)
        file_entity = _entities_by_type(tsx_result, EntityType.FILE)
        assert len(file_entity) > 0
        file_id = file_entity[0].id
        class_entity = _entity_by_name(tsx_result, "UserService")
        assert class_entity is not None
        matching = [
            r for r in contains
            if r.source_id == file_id and r.target_id == class_entity.id
        ]
        assert len(matching) > 0

    def test_file_contains_top_level_function(self, tsx_result):
        contains = _rels_by_type(tsx_result, RelationshipType.CONTAINS)
        file_entity = _entities_by_type(tsx_result, EntityType.FILE)
        assert len(file_entity) > 0
        file_id = file_entity[0].id
        func_entity = _entity_by_name(tsx_result, "formatUserName")
        assert func_entity is not None
        matching = [
            r for r in contains
            if r.source_id == file_id and r.target_id == func_entity.id
        ]
        assert len(matching) > 0


# ===========================================================================
# HAS_METHOD relationships
# ===========================================================================


class TestHasMethodRelationships:
    """Class entities should have HAS_METHOD relationships to their methods."""

    def test_class_has_method(self, tsx_result):
        has_methods = _rels_by_type(tsx_result, RelationshipType.HAS_METHOD)
        class_entity = _entity_by_name(tsx_result, "UserService")
        assert class_entity is not None
        method_entity = _entity_by_name(tsx_result, "fetchUser")
        assert method_entity is not None
        matching = [
            r for r in has_methods
            if r.source_id == class_entity.id and r.target_id == method_entity.id
        ]
        assert len(matching) > 0

    def test_edge_class_has_method(self, edge_result):
        has_methods = _rels_by_type(edge_result, RelationshipType.HAS_METHOD)
        class_entity = _entity_by_name(edge_result, "ApiClient")
        assert class_entity is not None
        method_entity = _entity_by_name(edge_result, "get")
        assert method_entity is not None
        matching = [
            r for r in has_methods
            if r.source_id == class_entity.id and r.target_id == method_entity.id
        ]
        assert len(matching) > 0


# ===========================================================================
# HAS_CONSTRUCTOR relationships
# ===========================================================================


class TestHasConstructorRelationships:
    """Class entities should have HAS_CONSTRUCTOR relationships."""

    def test_class_has_constructor(self, edge_result):
        has_constructors = _rels_by_type(edge_result, RelationshipType.HAS_CONSTRUCTOR)
        class_entity = _entity_by_name(edge_result, "ApiClient")
        assert class_entity is not None
        matching = [
            r for r in has_constructors
            if r.source_id == class_entity.id
        ]
        assert len(matching) > 0


# ===========================================================================
# Entity ID format
# ===========================================================================


class TestEntityIdFormat:
    """Entity IDs should follow the format: {repository}::{module}.{Class}.{method}"""

    def test_class_id_format(self, tsx_result):
        entity = _entity_by_name(tsx_result, "UserService")
        assert entity is not None
        # Should be: test-repo::sample_javascript.UserService
        assert entity.id == f"{REPOSITORY}::sample_javascript.UserService"

    def test_method_id_format(self, tsx_result):
        entity = _entity_by_name(tsx_result, "fetchUser")
        assert entity is not None
        # Should be: test-repo::sample_javascript.UserService.fetchUser
        assert entity.id == f"{REPOSITORY}::sample_javascript.UserService.fetchUser"

    def test_top_level_function_id_format(self, tsx_result):
        entity = _entity_by_name(tsx_result, "formatUserName")
        assert entity is not None
        # Should be: test-repo::sample_javascript.formatUserName
        assert entity.id == f"{REPOSITORY}::sample_javascript.formatUserName"

    def test_interface_id_format(self, tsx_result):
        entity = _entity_by_name(tsx_result, "UserProps")
        assert entity is not None
        assert entity.id == f"{REPOSITORY}::sample_javascript.UserProps"

    def test_constructor_id_format(self, edge_result):
        constructors = _entities_by_type(edge_result, EntityType.CONSTRUCTOR)
        api_ctor = [c for c in constructors if "ApiClient" in c.id]
        assert len(api_ctor) > 0
        assert api_ctor[0].id == f"{REPOSITORY}::sample_javascript_edge_cases.ApiClient.constructor"


# ===========================================================================
# Parse success
# ===========================================================================


class TestParseSuccess:
    """Parsing should succeed without errors."""

    def test_tsx_no_errors(self, tsx_result):
        assert tsx_result.success is True
        assert len(tsx_result.errors) == 0

    def test_edge_no_errors(self, edge_result):
        assert edge_result.success is True
        assert len(edge_result.errors) == 0

    def test_tsx_has_file_entity(self, tsx_result):
        files = _entities_by_type(tsx_result, EntityType.FILE)
        assert len(files) == 1

    def test_edge_has_file_entity(self, edge_result):
        files = _entities_by_type(edge_result, EntityType.FILE)
        assert len(files) == 1
