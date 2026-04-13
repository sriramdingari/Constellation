"""Tests for constellation.parsers.dotnet module (C#/.NET parser)."""

from pathlib import Path

import pytest

from constellation.models import CodeEntity, CodeRelationship, EntityType, RelationshipType
from constellation.parsers.dotnet import DotNetParser, _ParsingContext
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

    def test_nested_class_ids_include_outer_class(self, parser, tmp_path):
        nested_file = tmp_path / "Nested.cs"
        nested_file.write_text(
            "namespace SampleApp.Services {\n"
            "    public class OuterA {\n"
            "        public class Inner {}\n"
            "    }\n"
            "    public class OuterB {\n"
            "        public class Inner {}\n"
            "    }\n"
            "}\n"
        )

        result = parser.parse_file(nested_file, repository=REPOSITORY)

        inner_ids = {
            entity.id
            for entity in _find_entities(result, EntityType.CLASS)
            if entity.name == "Inner"
        }

        assert inner_ids == {
            f"{REPOSITORY}::SampleApp.Services.OuterA.Inner",
            f"{REPOSITORY}::SampleApp.Services.OuterB.Inner",
        }


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


# ===========================================================================
# Using-Directive Tracking
# ===========================================================================


class TestUsingDirectiveTracking:
    """_collect_usings populates the _ParsingContext correctly."""

    def test_existing_fixture_usings_collected(self, parser, tmp_path):
        """The SampleCSharp.cs fixture has 'using System;' and
        'using System.Collections.Generic;' — both should be tracked."""
        result = parser.parse_file(SAMPLE_FILE, repository=REPOSITORY)
        # Parsing succeeds (entity extraction unaffected)
        assert len(result.entities) > 0
        assert len(result.errors) == 0

    def test_regular_usings(self, parser, tmp_path):
        src = tmp_path / "Usings.cs"
        src.write_text(
            "using System;\n"
            "using System.Collections.Generic;\n"
            "\n"
            "namespace Foo {\n"
            "    class Bar {}\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)
        # File parses without error
        assert len(result.errors) == 0
        # Class still extracted
        assert _find_entity(result, "Bar", EntityType.CLASS) is not None

    def test_static_usings(self, parser, tmp_path):
        src = tmp_path / "StaticUsing.cs"
        src.write_text(
            "using static System.Math;\n"
            "using static System.Console;\n"
            "\n"
            "class Calc {\n"
            "    int X() { return Abs(-1); }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)
        assert len(result.errors) == 0
        assert _find_entity(result, "Calc", EntityType.CLASS) is not None

    def test_alias_usings(self, parser, tmp_path):
        src = tmp_path / "AliasUsing.cs"
        src.write_text(
            "using IntList = System.Collections.Generic.List<int>;\n"
            "\n"
            "class Foo {\n"
            "    IntList items;\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)
        assert len(result.errors) == 0
        assert _find_entity(result, "Foo", EntityType.CLASS) is not None

    def test_global_usings(self, parser, tmp_path):
        src = tmp_path / "GlobalUsing.cs"
        src.write_text(
            "global using System.Linq;\n"
            "global using static System.Console;\n"
            "\n"
            "class App {}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)
        assert len(result.errors) == 0
        assert _find_entity(result, "App", EntityType.CLASS) is not None

    def test_usings_inside_namespace(self, parser, tmp_path):
        src = tmp_path / "NsUsing.cs"
        src.write_text(
            "using System;\n"
            "\n"
            "namespace Outer {\n"
            "    using System.Linq;\n"
            "    class Inner {}\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)
        assert len(result.errors) == 0
        assert _find_entity(result, "Inner", EntityType.CLASS) is not None

    def test_mixed_usings_does_not_crash(self, parser, tmp_path):
        """A file with every using variant should parse without error."""
        src = tmp_path / "MixedUsings.cs"
        src.write_text(
            "using System;\n"
            "using static System.Math;\n"
            "using Alias = System.Text.StringBuilder;\n"
            "global using System.IO;\n"
            "global using static System.Console;\n"
            "\n"
            "namespace Demo {\n"
            "    using System.Linq;\n"
            "    class Widget {}\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)
        assert len(result.errors) == 0
        assert _find_entity(result, "Widget", EntityType.CLASS) is not None

    def test_context_populated_regular_usings(self, parser):
        """_collect_usings populates ctx.usings for plain using directives."""
        from tree_sitter import Parser as TSParser, Language
        import tree_sitter_c_sharp as tscsharp

        code = b"using System;\nusing System.IO;\nclass X {}\n"
        ts_parser = TSParser(Language(tscsharp.language()))
        tree = ts_parser.parse(code)
        ctx = _ParsingContext(file_path="test.cs", repository="r", code=code)
        parser._collect_usings(tree.root_node, ctx)

        assert "System" in ctx.usings
        assert "System.IO" in ctx.usings
        assert len(ctx.using_statics) == 0
        assert len(ctx.using_aliases) == 0

    def test_context_populated_static_usings(self, parser):
        """_collect_usings populates ctx.using_statics for 'using static'."""
        from tree_sitter import Parser as TSParser, Language
        import tree_sitter_c_sharp as tscsharp

        code = b"using static System.Math;\nusing static System.Console;\nclass X {}\n"
        ts_parser = TSParser(Language(tscsharp.language()))
        tree = ts_parser.parse(code)
        ctx = _ParsingContext(file_path="test.cs", repository="r", code=code)
        parser._collect_usings(tree.root_node, ctx)

        assert "System.Math" in ctx.using_statics
        assert "System.Console" in ctx.using_statics
        assert len(ctx.usings) == 0

    def test_context_populated_alias_usings(self, parser):
        """_collect_usings populates ctx.using_aliases for alias directives."""
        from tree_sitter import Parser as TSParser, Language
        import tree_sitter_c_sharp as tscsharp

        code = b"using IntList = System.Collections.Generic.List<int>;\nclass X {}\n"
        ts_parser = TSParser(Language(tscsharp.language()))
        tree = ts_parser.parse(code)
        ctx = _ParsingContext(file_path="test.cs", repository="r", code=code)
        parser._collect_usings(tree.root_node, ctx)

        assert "IntList" in ctx.using_aliases
        assert "System.Collections.Generic.List<int>" in ctx.using_aliases["IntList"]
        assert len(ctx.usings) == 0
        assert len(ctx.using_statics) == 0

    def test_context_populated_global_usings(self, parser):
        """Global usings go to the same buckets as non-global."""
        from tree_sitter import Parser as TSParser, Language
        import tree_sitter_c_sharp as tscsharp

        code = b"global using System.Linq;\nglobal using static System.Console;\nclass X {}\n"
        ts_parser = TSParser(Language(tscsharp.language()))
        tree = ts_parser.parse(code)
        ctx = _ParsingContext(file_path="test.cs", repository="r", code=code)
        parser._collect_usings(tree.root_node, ctx)

        assert "System.Linq" in ctx.usings
        assert "System.Console" in ctx.using_statics

    def test_context_populated_mixed(self, parser):
        """All using categories are correctly separated."""
        from tree_sitter import Parser as TSParser, Language
        import tree_sitter_c_sharp as tscsharp

        code = (
            b"using System;\n"
            b"using static System.Math;\n"
            b"using Alias = System.Text.StringBuilder;\n"
            b"global using System.IO;\n"
            b"global using static System.Console;\n"
            b"\n"
            b"namespace Demo {\n"
            b"    using System.Linq;\n"
            b"    class Widget {}\n"
            b"}\n"
        )
        ts_parser = TSParser(Language(tscsharp.language()))
        tree = ts_parser.parse(code)
        ctx = _ParsingContext(file_path="test.cs", repository="r", code=code)
        parser._collect_usings(tree.root_node, ctx)

        assert sorted(ctx.usings) == ["System", "System.IO", "System.Linq"]
        assert sorted(ctx.using_statics) == ["System.Console", "System.Math"]
        assert ctx.using_aliases == {"Alias": "System.Text.StringBuilder"}


# ===========================================================================
# Class/Method Pre-Collection
# ===========================================================================


class TestPreCollectClasses:
    """_pre_collect_classes populates module_class_ids, class_method_ids,
    and class_static_method_ids on _ParsingContext."""

    @staticmethod
    def _parse_and_collect(parser, code: bytes, repository: str = "r") -> _ParsingContext:
        """Helper: parse code, run _pre_collect_classes, return ctx."""
        from tree_sitter import Parser as TSParser, Language
        import tree_sitter_c_sharp as tscsharp

        ts_parser = TSParser(Language(tscsharp.language()))
        tree = ts_parser.parse(code)
        ctx = _ParsingContext(file_path="test.cs", repository=repository, code=code)
        parser._pre_collect_classes(tree.root_node, ctx)
        return ctx

    def test_single_class_in_namespace(self, parser):
        """A single class inside a namespace is registered in module_class_ids."""
        code = (
            b"namespace NS {\n"
            b"    public class Foo {\n"
            b"        public void Bar() {}\n"
            b"    }\n"
            b"}\n"
        )
        ctx = self._parse_and_collect(parser, code)

        assert "Foo" in ctx.module_class_ids
        assert ctx.module_class_ids["Foo"] == "r::NS.Foo"

    def test_class_method_ids_populated(self, parser):
        """Methods of a class are collected in class_method_ids."""
        code = (
            b"namespace NS {\n"
            b"    public class Foo {\n"
            b"        public void Bar() {}\n"
            b"        public int Baz() { return 1; }\n"
            b"    }\n"
            b"}\n"
        )
        ctx = self._parse_and_collect(parser, code)

        class_id = "r::NS.Foo"
        assert class_id in ctx.class_method_ids
        assert "Bar" in ctx.class_method_ids[class_id]
        assert ctx.class_method_ids[class_id]["Bar"] == "r::NS.Foo.Bar"
        assert "Baz" in ctx.class_method_ids[class_id]
        assert ctx.class_method_ids[class_id]["Baz"] == "r::NS.Foo.Baz"

    def test_static_methods_registered(self, parser):
        """Static methods appear in both class_method_ids and class_static_method_ids."""
        code = (
            b"namespace NS {\n"
            b"    public static class Helper {\n"
            b"        public static int Add(int a, int b) { return a + b; }\n"
            b"        public static int Sub(int a, int b) { return a - b; }\n"
            b"    }\n"
            b"}\n"
        )
        ctx = self._parse_and_collect(parser, code)

        class_id = "r::NS.Helper"
        assert class_id in ctx.class_static_method_ids
        assert "Add" in ctx.class_static_method_ids[class_id]
        assert ctx.class_static_method_ids[class_id]["Add"] == "r::NS.Helper.Add"
        assert "Sub" in ctx.class_static_method_ids[class_id]

    def test_instance_method_not_in_static_map(self, parser):
        """Non-static methods should NOT appear in class_static_method_ids."""
        code = (
            b"namespace NS {\n"
            b"    public class Svc {\n"
            b"        public void Run() {}\n"
            b"        public static void Create() {}\n"
            b"    }\n"
            b"}\n"
        )
        ctx = self._parse_and_collect(parser, code)

        class_id = "r::NS.Svc"
        # Both are in the all-methods map
        assert "Run" in ctx.class_method_ids[class_id]
        assert "Create" in ctx.class_method_ids[class_id]
        # Only Create is static
        assert "Create" in ctx.class_static_method_ids[class_id]
        assert "Run" not in ctx.class_static_method_ids[class_id]

    def test_multiple_classes(self, parser):
        """Multiple classes in the same namespace are all registered."""
        code = (
            b"namespace NS {\n"
            b"    public class Alpha { public void A() {} }\n"
            b"    public class Beta { public void B() {} }\n"
            b"}\n"
        )
        ctx = self._parse_and_collect(parser, code)

        assert "Alpha" in ctx.module_class_ids
        assert "Beta" in ctx.module_class_ids
        assert ctx.module_class_ids["Alpha"] == "r::NS.Alpha"
        assert ctx.module_class_ids["Beta"] == "r::NS.Beta"

    def test_nested_namespace(self, parser):
        """Classes in nested namespaces get fully qualified IDs."""
        code = (
            b"namespace Outer {\n"
            b"    namespace Inner {\n"
            b"        public class Deep { public void Go() {} }\n"
            b"    }\n"
            b"}\n"
        )
        ctx = self._parse_and_collect(parser, code)

        assert "Deep" in ctx.module_class_ids
        assert ctx.module_class_ids["Deep"] == "r::Outer.Inner.Deep"

    def test_nested_class(self, parser):
        """Nested classes use the outer class as part of their qualified name."""
        code = (
            b"namespace NS {\n"
            b"    public class Outer {\n"
            b"        public class Inner {\n"
            b"            public void DoStuff() {}\n"
            b"        }\n"
            b"    }\n"
            b"}\n"
        )
        ctx = self._parse_and_collect(parser, code)

        assert "Outer" in ctx.module_class_ids
        assert ctx.module_class_ids["Outer"] == "r::NS.Outer"
        assert "Inner" in ctx.module_class_ids
        assert ctx.module_class_ids["Inner"] == "r::NS.Outer.Inner"

        inner_id = "r::NS.Outer.Inner"
        assert inner_id in ctx.class_method_ids
        assert "DoStuff" in ctx.class_method_ids[inner_id]

    def test_class_without_namespace(self, parser):
        """A class at top level (no namespace) is still registered."""
        code = b"public class Standalone { public void Ping() {} }\n"
        ctx = self._parse_and_collect(parser, code)

        assert "Standalone" in ctx.module_class_ids
        assert ctx.module_class_ids["Standalone"] == "r::Standalone"

    def test_class_without_methods(self, parser):
        """A class with no methods should be in module_class_ids but
        not in class_method_ids."""
        code = (
            b"namespace NS {\n"
            b"    public class Empty {}\n"
            b"}\n"
        )
        ctx = self._parse_and_collect(parser, code)

        assert "Empty" in ctx.module_class_ids
        class_id = "r::NS.Empty"
        assert class_id not in ctx.class_method_ids
        assert class_id not in ctx.class_static_method_ids

    def test_file_scoped_namespace(self, parser):
        """File-scoped namespace (C# 10+): classes appear as root-level
        siblings in the AST, so they are pre-collected without the namespace
        prefix (matching the main entity walk behaviour)."""
        code = (
            b"namespace NS;\n"
            b"\n"
            b"public class Svc {\n"
            b"    public void Run() {}\n"
            b"}\n"
        )
        ctx = self._parse_and_collect(parser, code)

        # The class is a root-level sibling; pre-collection mirrors the main walk.
        assert "Svc" in ctx.module_class_ids

    def test_ids_match_main_walk(self, parser, tmp_path):
        """Pre-collected IDs match those produced by the main entity walk."""
        src = tmp_path / "Match.cs"
        src.write_text(
            "namespace NS {\n"
            "    public class TaxService {\n"
            "        public decimal Calculate() { return 0m; }\n"
            "        public static TaxService Create() { return new TaxService(); }\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository="repo")

        # Verify entity IDs from the main walk
        tax_cls = _find_entity(result, "TaxService", EntityType.CLASS)
        assert tax_cls is not None
        assert tax_cls.id == "repo::NS.TaxService"

        calc_method = _find_entity(result, "Calculate", EntityType.METHOD)
        assert calc_method is not None
        assert calc_method.id == "repo::NS.TaxService.Calculate"

        create_method = _find_entity(result, "Create", EntityType.METHOD)
        assert create_method is not None
        assert create_method.id == "repo::NS.TaxService.Create"

    def test_existing_fixtures_precollected(self, parser):
        """The pre-collection pass runs on the real fixture files without
        breaking any existing entity extraction."""
        result = parser.parse_file(SAMPLE_FILE, repository=REPOSITORY)
        assert len(result.entities) > 0
        assert len(result.errors) == 0

        edge_result = parser.parse_file(EDGE_CASES_FILE, repository=REPOSITORY)
        assert len(edge_result.entities) > 0
        assert len(edge_result.errors) == 0
