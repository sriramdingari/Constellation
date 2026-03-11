"""Tests for constellation.parsers.java module."""

from pathlib import Path

import pytest

from constellation.models import EntityType, RelationshipType
from constellation.parsers.java import JavaParser

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "SampleJava.java"
EDGE_CASES = FIXTURES / "SampleJavaEdgeCases.java"
REPO = "test-repo"


@pytest.fixture
def parser():
    return JavaParser()


@pytest.fixture
def result(parser):
    return parser.parse_file(SAMPLE, repository=REPO)


@pytest.fixture
def edge_result(parser):
    return parser.parse_file(EDGE_CASES, repository=REPO)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entities(result, etype=None):
    """Return entities, optionally filtered by EntityType."""
    if etype is None:
        return result.entities
    return [e for e in result.entities if e.entity_type == etype]


def _entity(result, name, etype=None):
    """Return the first entity matching *name* (and optionally *etype*)."""
    for e in result.entities:
        if e.name == name and (etype is None or e.entity_type == etype):
            return e
    return None


def _rels(result, rtype=None):
    """Return relationships, optionally filtered by RelationshipType."""
    if rtype is None:
        return result.relationships
    return [r for r in result.relationships if r.relationship_type == rtype]


def _rel_pairs(result, rtype):
    """Return (source_id, target_id) pairs for a given relationship type."""
    return [(r.source_id, r.target_id) for r in _rels(result, rtype)]


# ===========================================================================
# Parser Properties
# ===========================================================================


class TestParserProperties:
    def test_language(self, parser):
        assert parser.language == "java"

    def test_file_extensions(self, parser):
        assert parser.file_extensions == [".java"]

    def test_can_parse_java(self, parser):
        assert parser.can_parse(Path("Foo.java")) is True

    def test_cannot_parse_python(self, parser):
        assert parser.can_parse(Path("Foo.py")) is False


# ===========================================================================
# Successful Parsing
# ===========================================================================


class TestParseSuccess:
    def test_success_flag(self, result):
        assert result.success is True

    def test_language_set(self, result):
        assert result.language == "java"

    def test_file_path_set(self, result):
        assert result.file_path == str(SAMPLE)


# ===========================================================================
# Package Extraction
# ===========================================================================


class TestPackageExtraction:
    def test_package_entity_created(self, result):
        pkgs = _entities(result, EntityType.PACKAGE)
        assert len(pkgs) == 1

    def test_package_name(self, result):
        pkg = _entity(result, "com.example", EntityType.PACKAGE)
        assert pkg is not None

    def test_package_line_number(self, result):
        pkg = _entity(result, "com.example", EntityType.PACKAGE)
        assert pkg.line_number >= 1

    def test_in_package_relationship(self, result):
        in_pkg = _rels(result, RelationshipType.IN_PACKAGE)
        assert len(in_pkg) > 0


# ===========================================================================
# Class Extraction
# ===========================================================================


class TestClassExtraction:
    def test_classes_found(self, result):
        classes = _entities(result, EntityType.CLASS)
        names = {c.name for c in classes}
        assert "UserService" in names
        assert "UserController" in names

    def test_class_line_number(self, result):
        cls = _entity(result, "UserService", EntityType.CLASS)
        assert cls.line_number >= 1

    def test_class_line_end(self, result):
        cls = _entity(result, "UserService", EntityType.CLASS)
        assert cls.line_end is not None
        assert cls.line_end > cls.line_number

    def test_class_id_format(self, result):
        cls = _entity(result, "UserService", EntityType.CLASS)
        assert cls.id == f"{REPO}::com.example.UserService"

    def test_class_modifiers(self, result):
        cls = _entity(result, "UserService", EntityType.CLASS)
        assert "public" in cls.modifiers

    def test_class_docstring(self, result):
        cls = _entity(result, "UserService", EntityType.CLASS)
        assert cls.docstring is not None
        assert "managing users" in cls.docstring


# ===========================================================================
# Spring Stereotypes
# ===========================================================================


class TestSpringStereotypes:
    def test_service_stereotype(self, result):
        cls = _entity(result, "UserService", EntityType.CLASS)
        assert "service" in cls.stereotypes

    def test_rest_controller_stereotype(self, result):
        cls = _entity(result, "UserController", EntityType.CLASS)
        assert "rest_controller" in cls.stereotypes

    def test_repository_stereotype(self, edge_result):
        cls = _entity(edge_result, "OrderRepository", EntityType.CLASS)
        assert "repository" in cls.stereotypes

    def test_component_stereotype(self, edge_result):
        cls = _entity(edge_result, "OrderValidator", EntityType.CLASS)
        assert "component" in cls.stereotypes


# ===========================================================================
# Interface Extraction
# ===========================================================================


class TestInterfaceExtraction:
    def test_interface_entity(self, result):
        iface = _entity(result, "Auditable", EntityType.INTERFACE)
        assert iface is not None

    def test_interface_id_format(self, result):
        iface = _entity(result, "Auditable", EntityType.INTERFACE)
        assert iface.id == f"{REPO}::com.example.Auditable"

    def test_interface_has_method(self, result):
        iface = _entity(result, "Auditable", EntityType.INTERFACE)
        has_method_rels = [
            r for r in _rels(result, RelationshipType.HAS_METHOD)
            if r.source_id == iface.id
        ]
        assert len(has_method_rels) == 2  # audit() and getAuditLog()

    def test_interface_extends_interface(self, edge_result):
        extends = _rels(edge_result, RelationshipType.EXTENDS)
        searchable = _entity(edge_result, "Searchable", EntityType.INTERFACE)
        extends_from_searchable = [
            r for r in extends if r.source_id == searchable.id
        ]
        assert len(extends_from_searchable) == 1
        assert "Auditable" in extends_from_searchable[0].target_id


# ===========================================================================
# Endpoint Stereotypes
# ===========================================================================


class TestEndpointStereotypes:
    def test_get_mapping_stereotype(self, result):
        method = _entity(result, "getUsers", EntityType.METHOD)
        assert "endpoint" in method.stereotypes

    def test_post_mapping_stereotype(self, edge_result):
        method = _entity(edge_result, "createOrder", EntityType.METHOD)
        assert "endpoint" in method.stereotypes

    def test_put_mapping_stereotype(self, edge_result):
        method = _entity(edge_result, "updateOrder", EntityType.METHOD)
        assert "endpoint" in method.stereotypes

    def test_delete_mapping_stereotype(self, edge_result):
        method = _entity(edge_result, "deleteOrder", EntityType.METHOD)
        assert "endpoint" in method.stereotypes

    def test_request_mapping_stereotype(self, edge_result):
        method = _entity(edge_result, "health", EntityType.METHOD)
        assert "endpoint" in method.stereotypes


# ===========================================================================
# Test Stereotypes
# ===========================================================================


class TestTestStereotypes:
    def test_parameterized_test_stereotype(self, edge_result):
        method = _entity(edge_result, "testParam", EntityType.METHOD)
        assert "test" in method.stereotypes


# ===========================================================================
# Modifier Detection
# ===========================================================================


class TestModifiers:
    def test_public_modifier(self, result):
        method = _entity(result, "findById", EntityType.METHOD)
        assert "public" in method.modifiers

    def test_protected_modifier(self, result):
        method = _entity(result, "validate", EntityType.METHOD)
        assert "protected" in method.modifiers

    def test_private_modifier(self, result):
        field = _entity(result, "userRepository", EntityType.FIELD)
        assert "private" in field.modifiers

    def test_final_modifier(self, result):
        field = _entity(result, "userRepository", EntityType.FIELD)
        assert "final" in field.modifiers

    def test_static_modifier(self, edge_result):
        field = _entity(edge_result, "serialVersionUID", EntityType.FIELD)
        assert "static" in field.modifiers

    def test_abstract_modifier(self, edge_result):
        cls = _entity(edge_result, "BaseEntity", EntityType.CLASS)
        assert "abstract" in cls.modifiers

    def test_abstract_method(self, edge_result):
        method = _entity(edge_result, "save", EntityType.METHOD)
        assert "abstract" in method.modifiers


# ===========================================================================
# Constructor Extraction
# ===========================================================================


class TestConstructorExtraction:
    def test_constructor_entity(self, result):
        constructors = _entities(result, EntityType.CONSTRUCTOR)
        names = {c.name for c in constructors}
        assert "UserService" in names

    def test_constructor_id_includes_params(self, result):
        ctor = _entity(result, "UserService", EntityType.CONSTRUCTOR)
        assert ctor.id == f"{REPO}::com.example.UserService.UserService(UserRepository)"

    def test_has_constructor_relationship(self, result):
        cls = _entity(result, "UserService", EntityType.CLASS)
        ctor = _entity(result, "UserService", EntityType.CONSTRUCTOR)
        has_ctor = _rels(result, RelationshipType.HAS_CONSTRUCTOR)
        pairs = [(r.source_id, r.target_id) for r in has_ctor]
        assert (cls.id, ctor.id) in pairs


# ===========================================================================
# Field Extraction
# ===========================================================================


class TestFieldExtraction:
    def test_field_entity(self, result):
        fields = _entities(result, EntityType.FIELD)
        names = {f.name for f in fields}
        assert "userRepository" in names
        assert "serviceName" in names

    def test_has_field_relationship(self, result):
        cls = _entity(result, "UserService", EntityType.CLASS)
        field = _entity(result, "userRepository", EntityType.FIELD)
        has_field = _rels(result, RelationshipType.HAS_FIELD)
        pairs = [(r.source_id, r.target_id) for r in has_field]
        assert (cls.id, field.id) in pairs


# ===========================================================================
# Method Entity ID with Parameter Types (Overloaded Methods)
# ===========================================================================


class TestOverloadedMethodIds:
    def test_method_ids_unique_for_overloads(self, edge_result):
        methods = [
            e for e in _entities(edge_result, EntityType.METHOD)
            if e.name == "processOrder"
        ]
        assert len(methods) == 2
        ids = {m.id for m in methods}
        assert len(ids) == 2  # IDs must be unique

    def test_overloaded_id_includes_param_types(self, edge_result):
        methods = [
            e for e in _entities(edge_result, EntityType.METHOD)
            if e.name == "processOrder"
        ]
        ids = {m.id for m in methods}
        assert f"{REPO}::com.example.edge.OrderService.processOrder(String)" in ids
        assert f"{REPO}::com.example.edge.OrderService.processOrder(String,int)" in ids


# ===========================================================================
# EXTENDS and IMPLEMENTS Relationships
# ===========================================================================


class TestExtendsImplements:
    def test_extends_relationship(self, edge_result):
        extends = _rels(edge_result, RelationshipType.EXTENDS)
        order_service = _entity(edge_result, "OrderService", EntityType.CLASS)
        extends_from_os = [r for r in extends if r.source_id == order_service.id]
        assert len(extends_from_os) == 1
        assert "BaseEntity" in extends_from_os[0].target_id

    def test_implements_relationship(self, edge_result):
        impls = _rels(edge_result, RelationshipType.IMPLEMENTS)
        order_service = _entity(edge_result, "OrderService", EntityType.CLASS)
        impls_from_os = [r for r in impls if r.source_id == order_service.id]
        assert len(impls_from_os) == 1
        assert "Auditable" in impls_from_os[0].target_id


# ===========================================================================
# DECLARES Relationship (Inner/Nested Classes)
# ===========================================================================


class TestDeclaresRelationship:
    def test_inner_class_declares(self, edge_result):
        declares = _rels(edge_result, RelationshipType.DECLARES)
        container = _entity(edge_result, "Container", EntityType.CLASS)
        inner = _entity(edge_result, "InnerConfig", EntityType.CLASS)
        assert container is not None
        assert inner is not None
        pairs = [(r.source_id, r.target_id) for r in declares]
        assert (container.id, inner.id) in pairs


# ===========================================================================
# CONTAINS Relationship (File -> Class)
# ===========================================================================


class TestContainsRelationship:
    def test_file_contains_class(self, result):
        contains = _rels(result, RelationshipType.CONTAINS)
        cls = _entity(result, "UserService", EntityType.CLASS)
        file_entity = _entity(result, SAMPLE.name, EntityType.FILE)
        pairs = [(r.source_id, r.target_id) for r in contains]
        assert (file_entity.id, cls.id) in pairs


# ===========================================================================
# HAS_METHOD / HAS_CONSTRUCTOR / HAS_FIELD Relationships
# ===========================================================================


class TestHasRelationships:
    def test_has_method(self, result):
        cls = _entity(result, "UserService", EntityType.CLASS)
        method = _entity(result, "findById", EntityType.METHOD)
        has_method = _rels(result, RelationshipType.HAS_METHOD)
        pairs = [(r.source_id, r.target_id) for r in has_method]
        assert (cls.id, method.id) in pairs

    def test_has_constructor(self, result):
        cls = _entity(result, "UserService", EntityType.CLASS)
        ctor = _entity(result, "UserService", EntityType.CONSTRUCTOR)
        has_ctor = _rels(result, RelationshipType.HAS_CONSTRUCTOR)
        pairs = [(r.source_id, r.target_id) for r in has_ctor]
        assert (cls.id, ctor.id) in pairs

    def test_has_field(self, result):
        cls = _entity(result, "UserService", EntityType.CLASS)
        field = _entity(result, "userRepository", EntityType.FIELD)
        has_field = _rels(result, RelationshipType.HAS_FIELD)
        pairs = [(r.source_id, r.target_id) for r in has_field]
        assert (cls.id, field.id) in pairs


# ===========================================================================
# CALLS Relationships
# ===========================================================================


class TestCallsRelationships:
    def test_method_calls_another(self, edge_result):
        calls = _rels(edge_result, RelationshipType.CALLS)
        # processOrder(String) calls save() and audit()
        process_method = None
        for e in _entities(edge_result, EntityType.METHOD):
            if e.name == "processOrder" and "String)" in e.id and ",int)" not in e.id:
                process_method = e
                break
        assert process_method is not None
        call_targets = [r.target_id for r in calls if r.source_id == process_method.id]
        assert len(call_targets) >= 2  # save() and audit()


# ===========================================================================
# Docstring / Javadoc Capture
# ===========================================================================


class TestDocstring:
    def test_class_docstring(self, result):
        cls = _entity(result, "UserService", EntityType.CLASS)
        assert cls.docstring is not None
        assert "managing users" in cls.docstring

    def test_method_docstring(self, result):
        method = _entity(result, "findById", EntityType.METHOD)
        assert method.docstring is not None
        assert "user" in method.docstring.lower()

    def test_controller_docstring(self, result):
        ctrl = _entity(result, "UserController", EntityType.CLASS)
        assert ctrl.docstring is not None
        assert "REST" in ctrl.docstring


# ===========================================================================
# Enum Handling
# ===========================================================================


class TestEnumHandling:
    def test_enum_entity_type(self, edge_result):
        enum = _entity(edge_result, "OrderStatus", EntityType.CLASS)
        assert enum is not None

    def test_enum_stereotype(self, edge_result):
        enum = _entity(edge_result, "OrderStatus", EntityType.CLASS)
        assert "enum" in enum.stereotypes

    def test_enum_has_method(self, edge_result):
        enum = _entity(edge_result, "OrderStatus", EntityType.CLASS)
        method = _entity(edge_result, "isTerminal", EntityType.METHOD)
        assert method is not None
        has_method = _rels(edge_result, RelationshipType.HAS_METHOD)
        pairs = [(r.source_id, r.target_id) for r in has_method]
        assert (enum.id, method.id) in pairs


# ===========================================================================
# Empty File Handling
# ===========================================================================


class TestEmptyFile:
    def test_empty_file(self, parser, tmp_path):
        empty_file = tmp_path / "Empty.java"
        empty_file.write_text("")
        result = parser.parse_file(empty_file, repository=REPO)
        assert result.success is True
        # Only the FILE entity should exist
        assert len(_entities(result, EntityType.FILE)) == 1
        assert len(_entities(result, EntityType.CLASS)) == 0

    def test_nonexistent_file(self, parser):
        result = parser.parse_file(Path("/nonexistent/Foo.java"), repository=REPO)
        assert result.success is False
        assert len(result.errors) > 0


# ===========================================================================
# Line Numbers are 1-indexed
# ===========================================================================


class TestLineNumbers:
    def test_class_line_is_1_indexed(self, result):
        cls = _entity(result, "UserService", EntityType.CLASS)
        # The class is not on line 0
        assert cls.line_number >= 1

    def test_method_line_is_1_indexed(self, result):
        method = _entity(result, "findById", EntityType.METHOD)
        assert method.line_number >= 1

    def test_field_line_is_1_indexed(self, result):
        field = _entity(result, "userRepository", EntityType.FIELD)
        assert field.line_number >= 1

    def test_package_line_is_1_indexed(self, result):
        pkg = _entity(result, "com.example", EntityType.PACKAGE)
        assert pkg.line_number == 1
