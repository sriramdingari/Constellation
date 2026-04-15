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


# ===========================================================================
# Calls and References (Task 3)
# ===========================================================================


class TestCallsAndReferences:
    """Call extraction: same-class, this/base resolution, and references."""

    def test_same_class_method_call_resolves(self, parser, tmp_path):
        """A bare call to Validate() in a class that defines Validate
        resolves to the Validate method entity via a CALLS edge."""
        src = tmp_path / "SameClass.cs"
        src.write_text(
            "namespace NS {\n"
            "    public class Svc {\n"
            "        public void Run() {\n"
            "            Validate();\n"
            "        }\n"
            "        public void Validate() {}\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)

        run_method = _find_entity(result, "Run", EntityType.METHOD)
        validate_method = _find_entity(result, "Validate", EntityType.METHOD)
        assert run_method is not None
        assert validate_method is not None

        calls = _find_relationships(result, RelationshipType.CALLS)
        resolved = [r for r in calls if r.source_id == run_method.id and r.target_id == validate_method.id]
        assert len(resolved) == 1

    def test_this_method_call_resolves(self, parser, tmp_path):
        """this.Validate() resolves to the same-class method."""
        src = tmp_path / "ThisCall.cs"
        src.write_text(
            "namespace NS {\n"
            "    public class Svc {\n"
            "        public void Run() {\n"
            "            this.Validate();\n"
            "        }\n"
            "        public void Validate() {}\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)

        run_method = _find_entity(result, "Run", EntityType.METHOD)
        validate_method = _find_entity(result, "Validate", EntityType.METHOD)
        assert run_method is not None
        assert validate_method is not None

        calls = _find_relationships(result, RelationshipType.CALLS)
        resolved = [r for r in calls if r.source_id == run_method.id and r.target_id == validate_method.id]
        assert len(resolved) == 1

    def test_base_method_call_stays_unresolved(self, parser, tmp_path):
        """base.Save() stays unresolved — produces a Reference entity."""
        src = tmp_path / "BaseCall.cs"
        src.write_text(
            "namespace NS {\n"
            "    public class Child : BaseEntity {\n"
            "        public void DoWork() {\n"
            "            base.Save();\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)

        refs = _find_entities(result, EntityType.REFERENCE)
        base_refs = [r for r in refs if "Save" in r.name]
        assert len(base_refs) == 1

        # There should be a CALLS edge from DoWork to the reference
        do_work = _find_entity(result, "DoWork", EntityType.METHOD)
        assert do_work is not None
        calls = _find_relationships(result, RelationshipType.CALLS)
        call = [r for r in calls if r.source_id == do_work.id and r.target_id == base_refs[0].id]
        assert len(call) == 1

    def test_unresolved_external_call_produces_reference(self, parser, tmp_path):
        """A call to an external unknown method creates a REFERENCE entity."""
        src = tmp_path / "External.cs"
        src.write_text(
            "namespace NS {\n"
            "    public class Svc {\n"
            "        public void Run() {\n"
            "            ExternalLib.DoSomething();\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)

        refs = _find_entities(result, EntityType.REFERENCE)
        assert len(refs) >= 1
        ext_refs = [r for r in refs if "DoSomething" in r.name]
        assert len(ext_refs) == 1

        # CALLS edge exists from Run to the reference
        run_method = _find_entity(result, "Run", EntityType.METHOD)
        assert run_method is not None
        calls = _find_relationships(result, RelationshipType.CALLS)
        call = [r for r in calls if r.source_id == run_method.id and r.target_id == ext_refs[0].id]
        assert len(call) == 1

    def test_same_name_unresolved_in_different_methods_get_distinct_ids(self, parser, tmp_path):
        """Two methods calling the same unresolved name get distinct reference IDs."""
        src = tmp_path / "DistinctRefs.cs"
        src.write_text(
            "namespace NS {\n"
            "    public class Svc {\n"
            "        public void Alpha() {\n"
            "            Logger.Log();\n"
            "        }\n"
            "        public void Beta() {\n"
            "            Logger.Log();\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)

        refs = _find_entities(result, EntityType.REFERENCE)
        log_refs = [r for r in refs if "Log" in r.name]
        assert len(log_refs) == 2
        assert log_refs[0].id != log_refs[1].id

    def test_nameof_does_not_produce_calls_edge(self, parser, tmp_path):
        """nameof(Foo) is a compile-time operator and must NOT produce a CALLS edge."""
        src = tmp_path / "NameOf.cs"
        src.write_text(
            "namespace NS {\n"
            "    public class Svc {\n"
            "        public void Run() {\n"
            "            var n = nameof(Run);\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)

        calls = _find_relationships(result, RelationshipType.CALLS)
        # No CALLS edges should come from Run method
        run_method = _find_entity(result, "Run", EntityType.METHOD)
        assert run_method is not None
        run_calls = [r for r in calls if r.source_id == run_method.id]
        assert len(run_calls) == 0

    def test_typeof_does_not_produce_calls_edge(self, parser, tmp_path):
        """typeof(Foo) is a compile-time operator and must NOT produce a CALLS edge."""
        src = tmp_path / "TypeOf.cs"
        src.write_text(
            "namespace NS {\n"
            "    public class Svc {\n"
            "        public void Run() {\n"
            "            var t = typeof(Svc);\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)

        calls = _find_relationships(result, RelationshipType.CALLS)
        run_method = _find_entity(result, "Run", EntityType.METHOD)
        assert run_method is not None
        run_calls = [r for r in calls if r.source_id == run_method.id]
        assert len(run_calls) == 0

    def test_calls_inside_await_are_found(self, parser, tmp_path):
        """Calls inside await expressions are extracted via recursive walk."""
        src = tmp_path / "AwaitCall.cs"
        src.write_text(
            "namespace NS {\n"
            "    public class Svc {\n"
            "        public async Task Run() {\n"
            "            await DoAsync();\n"
            "        }\n"
            "        public Task DoAsync() { return Task.CompletedTask; }\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)

        run_method = _find_entity(result, "Run", EntityType.METHOD)
        do_async = _find_entity(result, "DoAsync", EntityType.METHOD)
        assert run_method is not None
        assert do_async is not None

        calls = _find_relationships(result, RelationshipType.CALLS)
        resolved = [r for r in calls if r.source_id == run_method.id and r.target_id == do_async.id]
        assert len(resolved) == 1

        edge_result = parser.parse_file(EDGE_CASES_FILE, repository=REPOSITORY)
        assert len(edge_result.entities) > 0
        assert len(edge_result.errors) == 0

    # ----- Task 4: Receiver Typing and Cross-Namespace Resolution -----

    def test_local_receiver_typing_resolves_instance_call(self, parser, tmp_path):
        """var svc = new TaxService(); svc.Calculate() resolves when
        TaxService is defined in the same file."""
        src = tmp_path / "ReceiverType.cs"
        src.write_text(
            "namespace NS {\n"
            "    public class TaxService {\n"
            "        public decimal Calculate() { return 0m; }\n"
            "    }\n"
            "    public class Client {\n"
            "        public void Run() {\n"
            "            var svc = new TaxService();\n"
            "            svc.Calculate();\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)

        run_method = _find_entity(result, "Run", EntityType.METHOD)
        calc_method = _find_entity(result, "Calculate", EntityType.METHOD)
        assert run_method is not None
        assert calc_method is not None

        calls = _find_relationships(result, RelationshipType.CALLS)
        resolved = [r for r in calls if r.source_id == run_method.id and r.target_id == calc_method.id]
        assert len(resolved) == 1, f"Expected resolved CALLS edge; got {[r.target_id for r in calls if r.source_id == run_method.id]}"

    def test_static_method_call_resolves(self, parser, tmp_path):
        """TaxService.StaticMethod() resolves to static method of same-file class."""
        src = tmp_path / "StaticCall.cs"
        src.write_text(
            "namespace NS {\n"
            "    public class TaxService {\n"
            "        public static decimal Calculate() { return 0m; }\n"
            "    }\n"
            "    public class Client {\n"
            "        public void Run() {\n"
            "            TaxService.Calculate();\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)

        run_method = _find_entity(result, "Run", EntityType.METHOD)
        calc_method = _find_entity(result, "Calculate", EntityType.METHOD)
        assert run_method is not None
        assert calc_method is not None

        calls = _find_relationships(result, RelationshipType.CALLS)
        resolved = [r for r in calls if r.source_id == run_method.id and r.target_id == calc_method.id]
        assert len(resolved) == 1, f"Expected resolved static CALLS edge; got {[r.target_id for r in calls if r.source_id == run_method.id]}"

    def test_using_alias_resolves_to_same_file_class(self, parser, tmp_path):
        """using Svc = NS.TaxService; then Svc.Calculate() resolves
        when the alias target short name matches a same-file class."""
        src = tmp_path / "AliasCall.cs"
        src.write_text(
            "using Svc = NS.TaxService;\n"
            "\n"
            "namespace NS {\n"
            "    public class TaxService {\n"
            "        public static decimal Calculate() { return 0m; }\n"
            "    }\n"
            "    public class Client {\n"
            "        public void Run() {\n"
            "            Svc.Calculate();\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)

        run_method = _find_entity(result, "Run", EntityType.METHOD)
        calc_method = _find_entity(result, "Calculate", EntityType.METHOD)
        assert run_method is not None
        assert calc_method is not None

        calls = _find_relationships(result, RelationshipType.CALLS)
        resolved = [r for r in calls if r.source_id == run_method.id and r.target_id == calc_method.id]
        assert len(resolved) == 1, f"Expected alias-resolved CALLS edge; got {[r.target_id for r in calls if r.source_id == run_method.id]}"

    def test_new_expression_resolved_emits_calls_edge(self, parser, tmp_path):
        """new TaxService() constructor call emits CALLS edge to class or constructor."""
        src = tmp_path / "NewCall.cs"
        src.write_text(
            "namespace NS {\n"
            "    public class TaxService {\n"
            "        public TaxService() {}\n"
            "        public decimal Calculate() { return 0m; }\n"
            "    }\n"
            "    public class Client {\n"
            "        public void Run() {\n"
            "            var svc = new TaxService();\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)

        run_method = _find_entity(result, "Run", EntityType.METHOD)
        tax_ctor = _find_entity(result, "TaxService", EntityType.CONSTRUCTOR)
        tax_cls = _find_entity(result, "TaxService", EntityType.CLASS)
        assert run_method is not None
        assert tax_cls is not None

        calls = _find_relationships(result, RelationshipType.CALLS)
        run_calls = [r for r in calls if r.source_id == run_method.id]
        # Should resolve to the constructor entity if it exists, otherwise class
        expected_target = tax_ctor.id if tax_ctor else tax_cls.id
        resolved = [r for r in run_calls if r.target_id == expected_target]
        assert len(resolved) == 1, f"Expected new-call CALLS edge to {expected_target}; got {[r.target_id for r in run_calls]}"

    def test_new_expression_unresolved_emits_reference(self, parser, tmp_path):
        """new ExternalClass() with unknown type emits Reference entity."""
        src = tmp_path / "NewExternal.cs"
        src.write_text(
            "namespace NS {\n"
            "    public class Client {\n"
            "        public void Run() {\n"
            "            var obj = new ExternalClass();\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)

        refs = _find_entities(result, EntityType.REFERENCE)
        new_refs = [r for r in refs if "ExternalClass" in r.name]
        assert len(new_refs) == 1, f"Expected Reference for new ExternalClass; got {[r.name for r in refs]}"

        # CALLS edge from Run to the reference
        run_method = _find_entity(result, "Run", EntityType.METHOD)
        assert run_method is not None
        calls = _find_relationships(result, RelationshipType.CALLS)
        call = [r for r in calls if r.source_id == run_method.id and r.target_id == new_refs[0].id]
        assert len(call) == 1

    def test_null_conditional_call_does_not_crash(self, parser, tmp_path):
        """handler?.Invoke() produces a CALLS edge (unresolved) and does not crash."""
        src = tmp_path / "NullConditional.cs"
        src.write_text(
            "namespace NS {\n"
            "    public class Svc {\n"
            "        public void Run() {\n"
            "            handler?.Invoke();\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)

        # Must not crash
        assert len(result.errors) == 0

        run_method = _find_entity(result, "Run", EntityType.METHOD)
        assert run_method is not None

        calls = _find_relationships(result, RelationshipType.CALLS)
        run_calls = [r for r in calls if r.source_id == run_method.id]
        # Should have at least one CALLS edge (likely unresolved reference)
        assert len(run_calls) >= 1, "Expected at least one CALLS edge from handler?.Invoke()"

    def test_same_name_classes_different_namespaces_resolve_correctly(self, parser, tmp_path):
        """Bug #1: When two classes share a short name in different namespaces,
        calls in the first class should NOT resolve to the second class."""
        source = tmp_path / "DualService.cs"
        source.write_text(
            "namespace A {\n"
            "  public class TaxService {\n"
            "    public void Calculate() {}\n"
            "  }\n"
            "  public class Client {\n"
            "    public void Run() {\n"
            "      var svc = new TaxService();\n"
            "      svc.Calculate();\n"
            "    }\n"
            "  }\n"
            "}\n"
            "namespace B {\n"
            "  public class TaxService {\n"
            "    public void Calculate() {}\n"
            "  }\n"
            "}\n"
        )

        result = parser.parse_file(source, repository=REPOSITORY)
        client_run = _find_entity(result, "Run", EntityType.METHOD)
        assert client_run is not None

        calls = _find_relationships(result, RelationshipType.CALLS)
        run_calls = [r for r in calls if r.source_id == client_run.id]

        # svc.Calculate() should resolve to A.TaxService.Calculate, NOT B.TaxService.Calculate
        a_calc_id = f"{REPOSITORY}::A.TaxService.Calculate"
        b_calc_id = f"{REPOSITORY}::B.TaxService.Calculate"
        call_targets = {r.target_id for r in run_calls}
        assert a_calc_id in call_targets, f"Expected A.TaxService.Calculate in {call_targets}"
        assert b_calc_id not in call_targets, f"B.TaxService.Calculate should not be in {call_targets}"

    def test_nested_block_instance_typing_resolves(self, parser, tmp_path):
        """Bug #2: Variable declarations inside if/for/while blocks should
        still be available for receiver typing within that block."""
        source = tmp_path / "NestedBlock.cs"
        source.write_text(
            "public class Worker {\n"
            "  public void DoWork() {}\n"
            "}\n"
            "public class Client {\n"
            "  public void Run() {\n"
            "    if (true) {\n"
            "      var w = new Worker();\n"
            "      w.DoWork();\n"
            "    }\n"
            "  }\n"
            "}\n"
        )

        result = parser.parse_file(source, repository=REPOSITORY)
        client_run = _find_entity(result, "Run", EntityType.METHOD)
        assert client_run is not None

        worker_dowork_id = f"{REPOSITORY}::Worker.DoWork"
        calls = _find_relationships(result, RelationshipType.CALLS)
        run_calls = [r for r in calls if r.source_id == client_run.id]
        call_targets = {r.target_id for r in run_calls}
        assert worker_dowork_id in call_targets, (
            f"Expected Worker.DoWork resolved, got targets: {call_targets}"
        )

    def test_class_level_test_attribute_propagates_to_methods(self, parser, tmp_path):
        """Bug #3: [TestClass] or [TestFixture] on a class should make ALL
        methods in that class get stereotypes=['test']."""
        source = tmp_path / "Verification.cs"
        source.write_text(
            "using Microsoft.VisualStudio.TestTools.UnitTesting;\n"
            "\n"
            "[TestClass]\n"
            "public class MyTests {\n"
            "  public void Helper() {}\n"
            "  [TestMethod]\n"
            "  public void TestSomething() {}\n"
            "}\n"
        )

        result = parser.parse_file(source, repository=REPOSITORY)
        helper = _find_entity(result, "Helper", EntityType.METHOD)
        assert helper is not None
        assert "test" in (helper.stereotypes or []), (
            f"Expected 'test' stereotype on Helper, got {helper.stereotypes}"
        )

    def test_switch_block_instance_typing_resolves(self, parser, tmp_path):
        """Instance typing inside switch blocks should resolve."""
        source = tmp_path / "SwitchBlock.cs"
        source.write_text(
            "public class Worker {\n"
            "  public void DoWork() {}\n"
            "}\n"
            "public class Client {\n"
            "  public void Run(int x) {\n"
            "    switch (x) {\n"
            "      case 1:\n"
            "        var w = new Worker();\n"
            "        w.DoWork();\n"
            "        break;\n"
            "    }\n"
            "  }\n"
            "}\n"
        )

        result = parser.parse_file(source, repository=REPOSITORY)
        client_run = _find_entity(result, "Run", EntityType.METHOD)
        assert client_run is not None

        worker_dowork_id = f"{REPOSITORY}::Worker.DoWork"
        calls = _find_relationships(result, RelationshipType.CALLS)
        run_calls = [r for r in calls if r.source_id == client_run.id]
        call_targets = {r.target_id for r in run_calls}
        assert worker_dowork_id in call_targets, (
            f"Expected Worker.DoWork resolved in switch block, got: {call_targets}"
        )

    def test_generic_member_call_not_dropped(self, parser, tmp_path):
        """Generic method calls on receivers must not be silently dropped."""
        source = tmp_path / "GenericCall.cs"
        source.write_text(
            "public class Converter {\n"
            "  public T Convert<T>(object input) { return default(T); }\n"
            "}\n"
            "public class Client {\n"
            "  public void Run() {\n"
            "    var c = new Converter();\n"
            "    c.Convert<int>(42);\n"
            "  }\n"
            "}\n"
        )

        result = parser.parse_file(source, repository=REPOSITORY)
        client_run = _find_entity(result, "Run", EntityType.METHOD)
        assert client_run is not None

        calls = _find_relationships(result, RelationshipType.CALLS)
        run_calls = [r for r in calls if r.source_id == client_run.id]
        # Must have at least a CALLS edge for c.Convert<int>() — either resolved or Reference
        convert_calls = [r for r in run_calls if "Convert" in r.target_id or "convert" in r.target_id.lower()]
        assert len(convert_calls) >= 1, (
            f"Generic member call c.Convert<int>() was dropped. All targets: "
            f"{[r.target_id for r in run_calls]}"
        )


# ===========================================================================
# Task 5: Extended Test Stereotype Detection (filename/path-based)
# ===========================================================================


class TestExtendedTestStereotypeDetection:
    """Methods get stereotype=['test'] based on filename/path patterns."""

    def test_file_named_FooTest_cs_gives_all_methods_test_stereotype(self, parser, tmp_path):
        """A method in a file named FooTest.cs gets stereotypes=['test'] even
        without any test-related attributes."""
        src = tmp_path / "FooTest.cs"
        src.write_text(
            "namespace App {\n"
            "    public class FooTest {\n"
            "        public void PlainHelper() {}\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)

        method = _find_entity(result, "PlainHelper", EntityType.METHOD)
        assert method is not None
        assert "test" in method.stereotypes, (
            f"Expected 'test' stereotype for method in FooTest.cs; got {method.stereotypes}"
        )

    def test_file_named_FooTests_cs_gives_test_stereotype(self, parser, tmp_path):
        """A file ending with Tests.cs also triggers test stereotype."""
        src = tmp_path / "FooTests.cs"
        src.write_text(
            "namespace App {\n"
            "    public class FooTests {\n"
            "        public void AnotherHelper() {}\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)

        method = _find_entity(result, "AnotherHelper", EntityType.METHOD)
        assert method is not None
        assert "test" in method.stereotypes

    def test_file_named_FooSpec_cs_gives_test_stereotype(self, parser, tmp_path):
        """A file ending with Spec.cs triggers test stereotype."""
        src = tmp_path / "FooSpec.cs"
        src.write_text(
            "namespace App {\n"
            "    public class FooSpec {\n"
            "        public void CheckBehavior() {}\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)

        method = _find_entity(result, "CheckBehavior", EntityType.METHOD)
        assert method is not None
        assert "test" in method.stereotypes

    def test_path_containing_dot_Tests_gives_test_stereotype(self, parser, tmp_path):
        """A method in a path containing '.Tests/' gets stereotypes=['test']."""
        test_dir = tmp_path / "MyProject.Tests" / "Unit"
        test_dir.mkdir(parents=True)
        src = test_dir / "OrderService.cs"
        src.write_text(
            "namespace MyProject.Tests.Unit {\n"
            "    public class OrderService {\n"
            "        public void PlainMethod() {}\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)

        method = _find_entity(result, "PlainMethod", EntityType.METHOD)
        assert method is not None
        assert "test" in method.stereotypes, (
            f"Expected 'test' stereotype for method in .Tests/ path; got {method.stereotypes}"
        )

    def test_path_containing_tests_dir_gives_test_stereotype(self, parser, tmp_path):
        """A method in a path containing '/tests/' gets stereotypes=['test']."""
        test_dir = tmp_path / "tests" / "unit"
        test_dir.mkdir(parents=True)
        src = test_dir / "Calc.cs"
        src.write_text(
            "namespace Tests.Unit {\n"
            "    public class Calc {\n"
            "        public void Add() {}\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)

        method = _find_entity(result, "Add", EntityType.METHOD)
        assert method is not None
        assert "test" in method.stereotypes

    def test_expanded_nunit_setup_attribute_gives_test_stereotype(self, parser, tmp_path):
        """NUnit [SetUp] attribute triggers test stereotype."""
        src = tmp_path / "Runner.cs"
        src.write_text(
            "namespace App {\n"
            "    public class Runner {\n"
            "        [SetUp]\n"
            "        public void Init() {}\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)

        method = _find_entity(result, "Init", EntityType.METHOD)
        assert method is not None
        assert "test" in method.stereotypes

    def test_expanded_mstest_testclass_attribute_gives_test_stereotype(self, parser, tmp_path):
        """MSTest [TestInitialize] attribute triggers test stereotype."""
        src = tmp_path / "Runner.cs"
        src.write_text(
            "namespace App {\n"
            "    public class Runner {\n"
            "        [TestInitialize]\n"
            "        public void Setup() {}\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)

        method = _find_entity(result, "Setup", EntityType.METHOD)
        assert method is not None
        assert "test" in method.stereotypes

    def test_no_duplicate_test_stereotype(self, parser, tmp_path):
        """A [Fact] method in a *Test.cs file should only get 'test' once."""
        src = tmp_path / "FooTest.cs"
        src.write_text(
            "namespace App {\n"
            "    public class FooTest {\n"
            "        [Fact]\n"
            "        public void MyTest() {}\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)

        method = _find_entity(result, "MyTest", EntityType.METHOD)
        assert method is not None
        assert method.stereotypes.count("test") == 1, (
            f"Expected exactly one 'test' stereotype; got {method.stereotypes}"
        )


# ===========================================================================
# Task 5: Endpoint Stereotype Detection
# ===========================================================================


class TestEndpointStereotypeDetection:
    """Methods in controller classes with HTTP attributes get stereotype=['endpoint']."""

    def test_httpget_in_class_inheriting_controllerbase(self, parser, tmp_path):
        """A method with [HttpGet] in a class inheriting ControllerBase
        gets stereotypes=['endpoint']."""
        src = tmp_path / "OrdersController.cs"
        src.write_text(
            "namespace App.Controllers {\n"
            "    public class OrdersController : ControllerBase {\n"
            "        [HttpGet]\n"
            "        public void GetOrders() {}\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)

        method = _find_entity(result, "GetOrders", EntityType.METHOD)
        assert method is not None
        assert "endpoint" in method.stereotypes, (
            f"Expected 'endpoint' stereotype; got {method.stereotypes}"
        )

    def test_httppost_in_class_with_apicontroller_attribute(self, parser, tmp_path):
        """A method with [HttpPost] in a class with [ApiController] attribute
        gets stereotypes=['endpoint']."""
        src = tmp_path / "UsersController.cs"
        src.write_text(
            "namespace App.Controllers {\n"
            "    [ApiController]\n"
            "    public class UsersController {\n"
            "        [HttpPost]\n"
            "        public void CreateUser() {}\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)

        method = _find_entity(result, "CreateUser", EntityType.METHOD)
        assert method is not None
        assert "endpoint" in method.stereotypes, (
            f"Expected 'endpoint' stereotype; got {method.stereotypes}"
        )

    def test_httpput_in_controller_inheriting_controller(self, parser, tmp_path):
        """A method with [HttpPut] in a class inheriting Controller
        gets stereotypes=['endpoint']."""
        src = tmp_path / "ItemsController.cs"
        src.write_text(
            "namespace App {\n"
            "    public class ItemsController : Controller {\n"
            "        [HttpPut]\n"
            "        public void UpdateItem() {}\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)

        method = _find_entity(result, "UpdateItem", EntityType.METHOD)
        assert method is not None
        assert "endpoint" in method.stereotypes

    def test_httpdelete_endpoint(self, parser, tmp_path):
        """[HttpDelete] also triggers endpoint stereotype."""
        src = tmp_path / "ProductsController.cs"
        src.write_text(
            "namespace App {\n"
            "    public class ProductsController : ControllerBase {\n"
            "        [HttpDelete]\n"
            "        public void DeleteProduct() {}\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)

        method = _find_entity(result, "DeleteProduct", EntityType.METHOD)
        assert method is not None
        assert "endpoint" in method.stereotypes

    def test_route_attribute_endpoint(self, parser, tmp_path):
        """[Route] attribute also triggers endpoint stereotype."""
        src = tmp_path / "HomeController.cs"
        src.write_text(
            "namespace App {\n"
            "    public class HomeController : Controller {\n"
            "        [Route(\"home\")]\n"
            "        public void Index() {}\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)

        method = _find_entity(result, "Index", EntityType.METHOD)
        assert method is not None
        assert "endpoint" in method.stereotypes

    def test_regular_method_in_non_controller_no_endpoint(self, parser, tmp_path):
        """A regular method in a non-controller class does NOT get 'endpoint'
        stereotype, even if it has an [HttpGet]-like attribute."""
        src = tmp_path / "Service.cs"
        src.write_text(
            "namespace App {\n"
            "    public class OrderService {\n"
            "        [HttpGet]\n"
            "        public void FetchOrders() {}\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)

        method = _find_entity(result, "FetchOrders", EntityType.METHOD)
        assert method is not None
        assert "endpoint" not in method.stereotypes, (
            f"Expected NO 'endpoint' for non-controller class; got {method.stereotypes}"
        )

    def test_method_without_http_attr_in_controller_no_endpoint(self, parser, tmp_path):
        """A method without an HTTP attribute in a controller class does NOT
        get 'endpoint' stereotype."""
        src = tmp_path / "FooController.cs"
        src.write_text(
            "namespace App {\n"
            "    public class FooController : ControllerBase {\n"
            "        public void HelperMethod() {}\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)

        method = _find_entity(result, "HelperMethod", EntityType.METHOD)
        assert method is not None
        assert "endpoint" not in method.stereotypes

    def test_combined_test_and_endpoint_stereotypes(self, parser, tmp_path):
        """A method can have both 'test' and 'endpoint' if conditions are met,
        although this is unusual in practice."""
        test_dir = tmp_path / "tests"
        test_dir.mkdir()
        src = test_dir / "ControllerTest.cs"
        src.write_text(
            "namespace App {\n"
            "    public class TestController : ControllerBase {\n"
            "        [HttpGet]\n"
            "        [Fact]\n"
            "        public void TestEndpoint() {}\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)

        method = _find_entity(result, "TestEndpoint", EntityType.METHOD)
        assert method is not None
        assert "test" in method.stereotypes
        assert "endpoint" in method.stereotypes

    def test_apicontroller_inheriting_apicontroller_base(self, parser, tmp_path):
        """A class inheriting ApiController counts as a controller."""
        src = tmp_path / "MyApi.cs"
        src.write_text(
            "namespace App {\n"
            "    public class MyApi : ApiController {\n"
            "        [HttpPatch]\n"
            "        public void PatchItem() {}\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse_file(src, repository=REPOSITORY)

        method = _find_entity(result, "PatchItem", EntityType.METHOD)
        assert method is not None
        assert "endpoint" in method.stereotypes
