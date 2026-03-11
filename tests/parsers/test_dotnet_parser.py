"""Tests for constellation.parsers.dotnet module (C#/.NET parser)."""

from pathlib import Path

import pytest

from constellation.models import CodeEntity, CodeRelationship, EntityType, RelationshipType
from constellation.parsers.dotnet import DotNetParser
from constellation.parsers.base import ParseResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_FILE = FIXTURES_DIR / "SampleCSharp.cs"
EDGE_CASES_FILE = FIXTURES_DIR / "SampleCSharpEdgeCases.cs"
REPOSITORY = "test-repo"


@pytest.fixture
def parser():
    return DotNetParser()


@pytest.fixture
def sample_result(parser):
    return parser.parse_file(SAMPLE_FILE, repository=REPOSITORY)


@pytest.fixture
def edge_cases_result(parser):
    return parser.parse_file(EDGE_CASES_FILE, repository=REPOSITORY)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_entity(result: ParseResult, name: str, entity_type: EntityType | None = None) -> CodeEntity | None:
    for e in result.entities:
        if e.name == name and (entity_type is None or e.entity_type == entity_type):
            return e
    return None


def _find_entities(result: ParseResult, entity_type: EntityType) -> list[CodeEntity]:
    return [e for e in result.entities if e.entity_type == entity_type]


def _find_relationship(
    result: ParseResult,
    rel_type: RelationshipType,
    source_name: str | None = None,
    target_name: str | None = None,
) -> CodeRelationship | None:
    for r in result.relationships:
        if r.relationship_type != rel_type:
            continue
        if source_name and source_name not in r.source_id:
            continue
        if target_name and target_name not in r.target_id:
            continue
        return r
    return None


def _find_relationships(result: ParseResult, rel_type: RelationshipType) -> list[CodeRelationship]:
    return [r for r in result.relationships if r.relationship_type == rel_type]


# ===========================================================================
# Parser Properties
# ===========================================================================


class TestParserProperties:
    """DotNetParser exposes correct language and extensions."""

    def test_language_is_csharp(self, parser):
        assert parser.language == "csharp"

    def test_file_extensions(self, parser):
        assert parser.file_extensions == [".cs"]

    def test_can_parse_cs_file(self, parser):
        assert parser.can_parse(Path("Foo.cs")) is True

    def test_cannot_parse_java_file(self, parser):
        assert parser.can_parse(Path("Foo.java")) is False


# ===========================================================================
# Namespace -> PACKAGE Entity
# ===========================================================================


class TestNamespaceExtraction:
    """Namespaces become PACKAGE entities."""

    def test_namespace_entity_found(self, sample_result):
        entity = _find_entity(sample_result, "Services", EntityType.PACKAGE)
        assert entity is not None

    def test_namespace_line_number(self, sample_result):
        entity = _find_entity(sample_result, "Services", EntityType.PACKAGE)
        assert entity.line_number >= 1

    def test_in_package_relationship(self, sample_result):
        """Classes in a namespace have IN_PACKAGE relationship."""
        rel = _find_relationship(
            sample_result,
            RelationshipType.IN_PACKAGE,
            source_name="UserService",
        )
        assert rel is not None


# ===========================================================================
# Class Extraction
# ===========================================================================


class TestClassExtraction:
    """Classes are extracted as CLASS entities."""

    def test_class_found(self, sample_result):
        entity = _find_entity(sample_result, "UserService", EntityType.CLASS)
        assert entity is not None

    def test_class_line_number(self, sample_result):
        entity = _find_entity(sample_result, "UserService", EntityType.CLASS)
        assert entity.line_number >= 1

    def test_class_line_end_greater_than_start(self, sample_result):
        entity = _find_entity(sample_result, "UserService", EntityType.CLASS)
        assert entity.line_end is not None
        assert entity.line_end > entity.line_number

    def test_class_docstring(self, sample_result):
        entity = _find_entity(sample_result, "UserService", EntityType.CLASS)
        assert entity.docstring is not None
        assert "managing users" in entity.docstring

    def test_class_modifiers(self, sample_result):
        entity = _find_entity(sample_result, "UserService", EntityType.CLASS)
        assert "public" in entity.modifiers

    def test_class_repository(self, sample_result):
        entity = _find_entity(sample_result, "UserService", EntityType.CLASS)
        assert entity.repository == REPOSITORY

    def test_class_language(self, sample_result):
        entity = _find_entity(sample_result, "UserService", EntityType.CLASS)
        assert entity.language == "csharp"


# ===========================================================================
# Interface Extraction
# ===========================================================================


class TestInterfaceExtraction:
    """Interfaces are extracted as INTERFACE entities."""

    def test_interface_found(self, sample_result):
        entity = _find_entity(sample_result, "IAuditable", EntityType.INTERFACE)
        assert entity is not None

    def test_interface_modifiers(self, sample_result):
        entity = _find_entity(sample_result, "IAuditable", EntityType.INTERFACE)
        assert "public" in entity.modifiers

    def test_interface_line_number(self, sample_result):
        entity = _find_entity(sample_result, "IAuditable", EntityType.INTERFACE)
        assert entity.line_number >= 1


# ===========================================================================
# Enum Extraction
# ===========================================================================


class TestEnumExtraction:
    """Enums are extracted as CLASS entities with 'enum' stereotype."""

    def test_enum_found(self, sample_result):
        entity = _find_entity(sample_result, "OrderStatus", EntityType.CLASS)
        assert entity is not None

    def test_enum_stereotype(self, sample_result):
        entity = _find_entity(sample_result, "OrderStatus", EntityType.CLASS)
        assert "enum" in entity.stereotypes

    def test_enum_modifiers(self, sample_result):
        entity = _find_entity(sample_result, "OrderStatus", EntityType.CLASS)
        assert "public" in entity.modifiers


# ===========================================================================
# Method Extraction
# ===========================================================================


class TestMethodExtraction:
    """Methods are extracted with correct metadata."""

    def test_method_found(self, sample_result):
        entity = _find_entity(sample_result, "FindById", EntityType.METHOD)
        assert entity is not None

    def test_method_signature(self, sample_result):
        entity = _find_entity(sample_result, "FindById", EntityType.METHOD)
        assert entity.signature is not None
        assert "FindById" in entity.signature

    def test_method_return_type(self, sample_result):
        entity = _find_entity(sample_result, "FindById", EntityType.METHOD)
        assert entity.return_type is not None
        assert "User" in entity.return_type

    def test_method_docstring(self, sample_result):
        entity = _find_entity(sample_result, "FindById", EntityType.METHOD)
        assert entity.docstring is not None
        assert "Finds" in entity.docstring

    def test_method_modifiers_public(self, sample_result):
        entity = _find_entity(sample_result, "FindById", EntityType.METHOD)
        assert "public" in entity.modifiers

    def test_method_modifiers_protected(self, sample_result):
        entity = _find_entity(sample_result, "Validate", EntityType.METHOD)
        assert entity is not None
        assert "protected" in entity.modifiers

    def test_method_line_end_greater_than_start(self, sample_result):
        entity = _find_entity(sample_result, "FindById", EntityType.METHOD)
        assert entity.line_end is not None
        assert entity.line_end > entity.line_number

    def test_method_code(self, sample_result):
        entity = _find_entity(sample_result, "FindById", EntityType.METHOD)
        assert entity.code is not None
        assert "FindById" in entity.code


# ===========================================================================
# Constructor Extraction
# ===========================================================================


class TestConstructorExtraction:
    """Constructors are extracted as CONSTRUCTOR entities."""

    def test_constructor_found(self, sample_result):
        entity = _find_entity(sample_result, "UserService", EntityType.CONSTRUCTOR)
        assert entity is not None

    def test_constructor_signature(self, sample_result):
        entity = _find_entity(sample_result, "UserService", EntityType.CONSTRUCTOR)
        assert entity.signature is not None
        assert "UserService" in entity.signature

    def test_has_constructor_relationship(self, sample_result):
        rel = _find_relationship(
            sample_result,
            RelationshipType.HAS_CONSTRUCTOR,
            source_name="UserService",
        )
        assert rel is not None


# ===========================================================================
# Field Extraction
# ===========================================================================


class TestFieldExtraction:
    """Fields are extracted as FIELD entities."""

    def test_fields_extracted(self, sample_result):
        fields = _find_entities(sample_result, EntityType.FIELD)
        field_names = [f.name for f in fields]
        assert "_userRepository" in field_names
        assert "_serviceName" in field_names

    def test_field_modifiers_private(self, sample_result):
        entity = _find_entity(sample_result, "_userRepository", EntityType.FIELD)
        assert entity is not None
        assert "private" in entity.modifiers

    def test_has_field_relationship(self, sample_result):
        rel = _find_relationship(
            sample_result,
            RelationshipType.HAS_FIELD,
            source_name="UserService",
            target_name="_userRepository",
        )
        assert rel is not None


# ===========================================================================
# Modifier Detection
# ===========================================================================


class TestModifierDetection:
    """Modifiers are correctly extracted from declarations."""

    def test_public_modifier(self, sample_result):
        entity = _find_entity(sample_result, "UserService", EntityType.CLASS)
        assert "public" in entity.modifiers

    def test_private_modifier(self, sample_result):
        entity = _find_entity(sample_result, "_userRepository", EntityType.FIELD)
        assert "private" in entity.modifiers

    def test_protected_modifier(self, sample_result):
        entity = _find_entity(sample_result, "Validate", EntityType.METHOD)
        assert "protected" in entity.modifiers

    def test_abstract_modifier(self, edge_cases_result):
        entity = _find_entity(edge_cases_result, "BaseEntity", EntityType.CLASS)
        assert entity is not None
        assert "abstract" in entity.modifiers

    def test_static_modifier(self, edge_cases_result):
        entity = _find_entity(edge_cases_result, "MathHelper", EntityType.CLASS)
        assert entity is not None
        assert "static" in entity.modifiers

    def test_static_method_modifier(self, edge_cases_result):
        entity = _find_entity(edge_cases_result, "Add", EntityType.METHOD)
        assert entity is not None
        assert "static" in entity.modifiers

    def test_async_modifier(self, edge_cases_result):
        entity = _find_entity(edge_cases_result, "ProcessOrderAsync", EntityType.METHOD)
        assert entity is not None
        assert "async" in entity.modifiers

    def test_virtual_modifier(self, edge_cases_result):
        entity = _find_entity(edge_cases_result, "ProcessOrder", EntityType.METHOD)
        assert entity is not None
        assert "virtual" in entity.modifiers

    def test_override_modifier(self, edge_cases_result):
        saves = [e for e in edge_cases_result.entities if e.name == "Save" and e.entity_type == EntityType.METHOD]
        override_save = [e for e in saves if "override" in e.modifiers]
        assert len(override_save) == 1

    def test_internal_modifier(self, edge_cases_result):
        entity = _find_entity(edge_cases_result, "InternalHelper", EntityType.CLASS)
        assert entity is not None
        assert "internal" in entity.modifiers

    def test_abstract_method_modifier(self, edge_cases_result):
        # The abstract Save() declared in BaseEntity
        # Find the abstract Save method (in BaseEntity, before OrderService overrides it)
        methods = [
            e for e in edge_cases_result.entities
            if e.name == "Save" and e.entity_type == EntityType.METHOD and "abstract" in e.modifiers
        ]
        assert len(methods) >= 1


# ===========================================================================
# Return Type Capture
# ===========================================================================


class TestReturnTypeCapture:
    """Return types are correctly captured."""

    def test_simple_return_type(self, sample_result):
        entity = _find_entity(sample_result, "FindById", EntityType.METHOD)
        assert entity.return_type is not None
        assert "User" in entity.return_type

    def test_void_return_type(self, sample_result):
        entity = _find_entity(sample_result, "Validate", EntityType.METHOD)
        assert entity.return_type is not None
        assert "void" in entity.return_type

    def test_generic_return_type(self, edge_cases_result):
        entity = _find_entity(edge_cases_result, "ProcessOrderAsync", EntityType.METHOD)
        assert entity.return_type is not None
        assert "Task" in entity.return_type

    def test_static_method_return_type(self, edge_cases_result):
        entity = _find_entity(edge_cases_result, "Add", EntityType.METHOD)
        assert entity.return_type is not None
        assert "int" in entity.return_type


# ===========================================================================
# EXTENDS Relationships
# ===========================================================================


class TestExtendsRelationships:
    """EXTENDS relationships are created for class inheritance."""

    def test_extends_relationship(self, edge_cases_result):
        rel = _find_relationship(
            edge_cases_result,
            RelationshipType.EXTENDS,
            source_name="OrderService",
            target_name="BaseEntity",
        )
        assert rel is not None


# ===========================================================================
# IMPLEMENTS Relationships
# ===========================================================================


class TestImplementsRelationships:
    """IMPLEMENTS relationships are created for interface implementations."""

    def test_implements_relationship(self, edge_cases_result):
        rel = _find_relationship(
            edge_cases_result,
            RelationshipType.IMPLEMENTS,
            source_name="OrderService",
            target_name="IAuditable",
        )
        assert rel is not None


# ===========================================================================
# CONTAINS Relationships
# ===========================================================================


class TestContainsRelationships:
    """File CONTAINS Class relationships."""

    def test_file_contains_class(self, sample_result):
        rel = _find_relationship(
            sample_result,
            RelationshipType.CONTAINS,
            target_name="UserService",
        )
        assert rel is not None

    def test_file_contains_interface(self, sample_result):
        rel = _find_relationship(
            sample_result,
            RelationshipType.CONTAINS,
            target_name="IAuditable",
        )
        assert rel is not None

    def test_file_contains_enum(self, sample_result):
        rel = _find_relationship(
            sample_result,
            RelationshipType.CONTAINS,
            target_name="OrderStatus",
        )
        assert rel is not None


# ===========================================================================
# HAS_METHOD Relationships
# ===========================================================================


class TestHasMethodRelationships:
    """Class HAS_METHOD for methods."""

    def test_class_has_method(self, sample_result):
        rel = _find_relationship(
            sample_result,
            RelationshipType.HAS_METHOD,
            source_name="UserService",
            target_name="FindById",
        )
        assert rel is not None

    def test_class_has_protected_method(self, sample_result):
        rel = _find_relationship(
            sample_result,
            RelationshipType.HAS_METHOD,
            source_name="UserService",
            target_name="Validate",
        )
        assert rel is not None


# ===========================================================================
# HAS_CONSTRUCTOR Relationships
# ===========================================================================


class TestHasConstructorRelationships:
    """Class HAS_CONSTRUCTOR for constructors."""

    def test_class_has_constructor(self, sample_result):
        rel = _find_relationship(
            sample_result,
            RelationshipType.HAS_CONSTRUCTOR,
            source_name="UserService",
        )
        assert rel is not None


# ===========================================================================
# HAS_FIELD Relationships
# ===========================================================================


class TestHasFieldRelationships:
    """Class HAS_FIELD for field declarations."""

    def test_class_has_field(self, sample_result):
        rel = _find_relationship(
            sample_result,
            RelationshipType.HAS_FIELD,
            source_name="UserService",
        )
        assert rel is not None

    def test_multiple_fields(self, sample_result):
        rels = [
            r for r in sample_result.relationships
            if r.relationship_type == RelationshipType.HAS_FIELD
            and "UserService" in r.source_id
        ]
        assert len(rels) >= 2  # _userRepository, _serviceName


# ===========================================================================
# DECLARES Relationships (Nested Classes)
# ===========================================================================


class TestDeclaresRelationships:
    """Outer class DECLARES nested class."""

    def test_declares_nested_class(self, edge_cases_result):
        rel = _find_relationship(
            edge_cases_result,
            RelationshipType.DECLARES,
            source_name="Container",
            target_name="InnerConfig",
        )
        assert rel is not None


# ===========================================================================
# Test Stereotype Detection
# ===========================================================================


class TestTestStereotypeDetection:
    """Test methods with [TestMethod], [Fact], [Test] get 'test' stereotype."""

    def test_testmethod_attribute(self, edge_cases_result):
        entity = _find_entity(edge_cases_result, "TestSaveOrder", EntityType.METHOD)
        assert entity is not None
        assert "test" in entity.stereotypes

    def test_fact_attribute(self, edge_cases_result):
        entity = _find_entity(edge_cases_result, "ShouldProcessOrder", EntityType.METHOD)
        assert entity is not None
        assert "test" in entity.stereotypes

    def test_test_attribute(self, edge_cases_result):
        entity = _find_entity(edge_cases_result, "VerifyAuditLog", EntityType.METHOD)
        assert entity is not None
        assert "test" in entity.stereotypes


# ===========================================================================
# Docstring (XML Doc Comments) Capture
# ===========================================================================


class TestDocstringCapture:
    """XML doc comments (/// <summary>) are captured as docstrings."""

    def test_class_docstring(self, sample_result):
        entity = _find_entity(sample_result, "UserService", EntityType.CLASS)
        assert entity.docstring is not None
        assert "managing users" in entity.docstring

    def test_method_docstring(self, sample_result):
        entity = _find_entity(sample_result, "FindById", EntityType.METHOD)
        assert entity.docstring is not None
        assert "Finds" in entity.docstring

    def test_abstract_class_docstring(self, edge_cases_result):
        entity = _find_entity(edge_cases_result, "BaseEntity", EntityType.CLASS)
        assert entity.docstring is not None
        assert "Abstract" in entity.docstring

    def test_async_method_docstring(self, edge_cases_result):
        entity = _find_entity(edge_cases_result, "ProcessOrderAsync", EntityType.METHOD)
        assert entity.docstring is not None
        assert "asynchronously" in entity.docstring


# ===========================================================================
# Entity ID Format
# ===========================================================================


class TestEntityIdFormat:
    """Entity IDs follow the format {repository}::{namespace}.{Class}.{method}."""

    def test_class_entity_id(self, sample_result):
        entity = _find_entity(sample_result, "UserService", EntityType.CLASS)
        assert entity.id.startswith(f"{REPOSITORY}::")
        assert "UserService" in entity.id

    def test_method_entity_id(self, sample_result):
        entity = _find_entity(sample_result, "FindById", EntityType.METHOD)
        assert entity.id.startswith(f"{REPOSITORY}::")
        assert "UserService" in entity.id
        assert "FindById" in entity.id

    def test_constructor_entity_id(self, sample_result):
        entity = _find_entity(sample_result, "UserService", EntityType.CONSTRUCTOR)
        assert entity.id.startswith(f"{REPOSITORY}::")
        assert "UserService" in entity.id

    def test_namespace_in_entity_id(self, sample_result):
        entity = _find_entity(sample_result, "UserService", EntityType.CLASS)
        assert "SampleApp.Services" in entity.id


# ===========================================================================
# Line Numbers
# ===========================================================================


class TestLineNumbers:
    """Line numbers are 1-indexed and line_end > line_number for multi-line entities."""

    def test_class_line_number_one_indexed(self, sample_result):
        entity = _find_entity(sample_result, "UserService", EntityType.CLASS)
        assert entity.line_number >= 1

    def test_method_line_number_one_indexed(self, sample_result):
        entity = _find_entity(sample_result, "FindById", EntityType.METHOD)
        assert entity.line_number >= 1

    def test_class_line_end_greater(self, sample_result):
        entity = _find_entity(sample_result, "UserService", EntityType.CLASS)
        assert entity.line_end is not None
        assert entity.line_end > entity.line_number

    def test_method_line_end_greater(self, sample_result):
        entity = _find_entity(sample_result, "FindById", EntityType.METHOD)
        assert entity.line_end is not None
        assert entity.line_end > entity.line_number

    def test_interface_line_number(self, sample_result):
        entity = _find_entity(sample_result, "IAuditable", EntityType.INTERFACE)
        assert entity.line_number >= 1

    def test_enum_line_number(self, sample_result):
        entity = _find_entity(sample_result, "OrderStatus", EntityType.CLASS)
        assert entity.line_number >= 1


# ===========================================================================
# Interface Method Extraction
# ===========================================================================


class TestInterfaceMethodExtraction:
    """Interface methods are extracted with HAS_METHOD."""

    def test_interface_has_method(self, sample_result):
        rel = _find_relationship(
            sample_result,
            RelationshipType.HAS_METHOD,
            source_name="IAuditable",
            target_name="Audit",
        )
        assert rel is not None

    def test_interface_method_return_type(self, sample_result):
        entity = _find_entity(sample_result, "GetAuditLog", EntityType.METHOD)
        assert entity is not None
        assert entity.return_type is not None
        assert "string" in entity.return_type


# ===========================================================================
# Abstract Class and Method
# ===========================================================================


class TestAbstractClassAndMethod:
    """Abstract classes and methods have correct modifiers."""

    def test_abstract_class(self, edge_cases_result):
        entity = _find_entity(edge_cases_result, "BaseEntity", EntityType.CLASS)
        assert entity is not None
        assert "abstract" in entity.modifiers
        assert "public" in entity.modifiers

    def test_abstract_method(self, edge_cases_result):
        methods = [
            e for e in edge_cases_result.entities
            if e.name == "Save" and e.entity_type == EntityType.METHOD and "abstract" in e.modifiers
        ]
        assert len(methods) >= 1


# ===========================================================================
# Static Class
# ===========================================================================


class TestStaticClass:
    """Static classes have 'static' modifier."""

    def test_static_class(self, edge_cases_result):
        entity = _find_entity(edge_cases_result, "MathHelper", EntityType.CLASS)
        assert entity is not None
        assert "static" in entity.modifiers

    def test_static_class_methods(self, edge_cases_result):
        entity = _find_entity(edge_cases_result, "Add", EntityType.METHOD)
        assert entity is not None
        assert "static" in entity.modifiers


# ===========================================================================
# Async Method
# ===========================================================================


class TestAsyncMethod:
    """Async methods have 'async' modifier and correct return type."""

    def test_async_method_modifier(self, edge_cases_result):
        entity = _find_entity(edge_cases_result, "ProcessOrderAsync", EntityType.METHOD)
        assert entity is not None
        assert "async" in entity.modifiers

    def test_async_method_return_type(self, edge_cases_result):
        entity = _find_entity(edge_cases_result, "ProcessOrderAsync", EntityType.METHOD)
        assert entity.return_type is not None
        assert "Task" in entity.return_type


# ===========================================================================
# Readonly Modifier on Field
# ===========================================================================


class TestReadonlyModifier:
    """Fields with 'readonly' modifier are detected."""

    def test_readonly_field(self, sample_result):
        entity = _find_entity(sample_result, "_userRepository", EntityType.FIELD)
        assert entity is not None
        assert "readonly" in entity.modifiers
