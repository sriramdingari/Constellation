"""Tests for constellation.parsers.python_parser module."""

from pathlib import Path

import pytest

from constellation.models import CodeEntity, CodeRelationship, EntityType, RelationshipType
from constellation.parsers.python_parser import PythonParser
from constellation.parsers.base import ParseResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_FILE = FIXTURES_DIR / "sample_python.py"
EDGE_CASES_FILE = FIXTURES_DIR / "sample_python_edge_cases.py"
EMPTY_FILE = FIXTURES_DIR / "empty_file.py"
SYNTAX_ERROR_FILE = FIXTURES_DIR / "syntax_error.py"
REPOSITORY = "test-repo"


@pytest.fixture
def parser():
    return PythonParser()


@pytest.fixture
def sample_result(parser):
    return parser.parse_file(SAMPLE_FILE, repository=REPOSITORY)


@pytest.fixture
def edge_cases_result(parser):
    return parser.parse_file(EDGE_CASES_FILE, repository=REPOSITORY)


@pytest.fixture
def empty_result(parser):
    return parser.parse_file(EMPTY_FILE, repository=REPOSITORY)


@pytest.fixture
def syntax_error_result(parser):
    return parser.parse_file(SYNTAX_ERROR_FILE, repository=REPOSITORY)


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
    """PythonParser exposes correct language and extensions."""

    def test_language_is_python(self, parser):
        assert parser.language == "python"

    def test_file_extensions(self, parser):
        assert parser.file_extensions == [".py"]

    def test_can_parse_py_file(self, parser):
        assert parser.can_parse(Path("foo.py")) is True

    def test_cannot_parse_java_file(self, parser):
        assert parser.can_parse(Path("Foo.java")) is False


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

    def test_class_signature(self, sample_result):
        entity = _find_entity(sample_result, "UserService", EntityType.CLASS)
        assert entity.signature is not None
        assert "UserService" in entity.signature

    def test_class_repository(self, sample_result):
        entity = _find_entity(sample_result, "UserService", EntityType.CLASS)
        assert entity.repository == REPOSITORY

    def test_class_language(self, sample_result):
        entity = _find_entity(sample_result, "UserService", EntityType.CLASS)
        assert entity.language == "python"


# ===========================================================================
# Method Extraction
# ===========================================================================


class TestMethodExtraction:
    """Methods are extracted with correct metadata."""

    def test_method_found(self, sample_result):
        entity = _find_entity(sample_result, "get_user", EntityType.METHOD)
        assert entity is not None

    def test_method_signature(self, sample_result):
        entity = _find_entity(sample_result, "get_user", EntityType.METHOD)
        assert entity.signature is not None
        assert "get_user" in entity.signature

    def test_method_return_type(self, sample_result):
        entity = _find_entity(sample_result, "get_user", EntityType.METHOD)
        assert entity.return_type is not None
        # Should capture Optional[dict] or similar
        assert "dict" in entity.return_type or "Optional" in entity.return_type

    def test_method_docstring(self, sample_result):
        entity = _find_entity(sample_result, "get_user", EntityType.METHOD)
        assert entity.docstring is not None
        assert "Fetch" in entity.docstring

    def test_method_code(self, sample_result):
        entity = _find_entity(sample_result, "get_user", EntityType.METHOD)
        assert entity.code is not None
        assert "def get_user" in entity.code

    def test_method_line_end_greater_than_start(self, sample_result):
        entity = _find_entity(sample_result, "get_user", EntityType.METHOD)
        assert entity.line_end is not None
        assert entity.line_end > entity.line_number


# ===========================================================================
# Constructor Extraction
# ===========================================================================


class TestConstructorExtraction:
    """__init__ methods are extracted as CONSTRUCTOR entities."""

    def test_init_is_constructor(self, sample_result):
        entity = _find_entity(sample_result, "__init__", EntityType.CONSTRUCTOR)
        assert entity is not None

    def test_constructor_signature(self, sample_result):
        entity = _find_entity(sample_result, "__init__", EntityType.CONSTRUCTOR)
        assert entity.signature is not None
        assert "__init__" in entity.signature

    def test_has_constructor_relationship(self, sample_result):
        rel = _find_relationship(
            sample_result,
            RelationshipType.HAS_CONSTRUCTOR,
            source_name="UserService",
            target_name="__init__",
        )
        assert rel is not None


# ===========================================================================
# Top-level Function Extraction
# ===========================================================================


class TestTopLevelFunctionExtraction:
    """Top-level functions become METHOD entities with File CONTAINS relationship."""

    def test_function_found(self, sample_result):
        entity = _find_entity(sample_result, "process_data", EntityType.METHOD)
        assert entity is not None

    def test_function_return_type(self, sample_result):
        entity = _find_entity(sample_result, "process_data", EntityType.METHOD)
        assert entity.return_type is not None
        assert "int" in entity.return_type

    def test_function_docstring(self, sample_result):
        entity = _find_entity(sample_result, "process_data", EntityType.METHOD)
        assert entity.docstring is not None
        assert "Process" in entity.docstring

    def test_file_contains_function(self, sample_result):
        rel = _find_relationship(
            sample_result,
            RelationshipType.CONTAINS,
            target_name="process_data",
        )
        assert rel is not None


# ===========================================================================
# Field Extraction
# ===========================================================================


class TestFieldExtraction:
    """self.x assignments in __init__ produce FIELD entities."""

    def test_fields_extracted(self, edge_cases_result):
        fields = _find_entities(edge_cases_result, EntityType.FIELD)
        field_names = [f.name for f in fields]
        assert "precision" in field_names
        assert "history" in field_names
        assert "_cache" in field_names

    def test_has_field_relationship(self, edge_cases_result):
        rel = _find_relationship(
            edge_cases_result,
            RelationshipType.HAS_FIELD,
            source_name="Calculator",
            target_name="precision",
        )
        assert rel is not None


# ===========================================================================
# Async Detection
# ===========================================================================


class TestAsyncDetection:
    """Async methods have 'async' in their modifiers."""

    def test_async_method_detected(self, sample_result):
        entity = _find_entity(sample_result, "list_users", EntityType.METHOD)
        assert entity is not None
        assert "async" in entity.modifiers


# ===========================================================================
# Test Stereotype Detection
# ===========================================================================


class TestTestStereotypeDetection:
    """Test functions and classes receive 'test' stereotype."""

    def test_test_method_stereotype(self, sample_result):
        entity = _find_entity(sample_result, "test_get_user", EntityType.METHOD)
        assert entity is not None
        assert "test" in entity.stereotypes

    def test_test_class_stereotype(self, sample_result):
        entity = _find_entity(sample_result, "TestUserService", EntityType.CLASS)
        assert entity is not None
        assert "test" in entity.stereotypes


# ===========================================================================
# Django Model Stereotype
# ===========================================================================


class TestDjangoModelStereotype:
    """Classes inheriting models.Model get 'django_model' stereotype."""

    def test_django_model_stereotype(self, edge_cases_result):
        entity = _find_entity(edge_cases_result, "Product", EntityType.CLASS)
        assert entity is not None
        assert "django_model" in entity.stereotypes


# ===========================================================================
# Pydantic Model Stereotype
# ===========================================================================


class TestPydanticModelStereotype:
    """Classes inheriting BaseModel get 'pydantic_model' stereotype."""

    def test_pydantic_model_stereotype(self, edge_cases_result):
        entity = _find_entity(edge_cases_result, "UserSchema", EntityType.CLASS)
        assert entity is not None
        assert "pydantic_model" in entity.stereotypes


# ===========================================================================
# Celery Task Stereotype
# ===========================================================================


class TestCeleryTaskStereotype:
    """Functions decorated with @shared_task get 'celery_task' stereotype."""

    def test_celery_task_stereotype(self, edge_cases_result):
        entity = _find_entity(edge_cases_result, "send_email", EntityType.METHOD)
        assert entity is not None
        assert "celery_task" in entity.stereotypes


# ===========================================================================
# Endpoint Stereotype
# ===========================================================================


class TestEndpointStereotype:
    """Functions decorated with @api_view get 'endpoint' stereotype."""

    def test_endpoint_stereotype(self, edge_cases_result):
        entity = _find_entity(edge_cases_result, "list_products", EntityType.METHOD)
        assert entity is not None
        assert "endpoint" in entity.stereotypes


# ===========================================================================
# Enum Stereotype
# ===========================================================================


class TestEnumStereotype:
    """Classes inheriting Enum get 'enum' stereotype and entity_type CLASS."""

    def test_enum_stereotype(self, edge_cases_result):
        entity = _find_entity(edge_cases_result, "Color", EntityType.CLASS)
        assert entity is not None
        assert "enum" in entity.stereotypes

    def test_enum_is_class_entity(self, edge_cases_result):
        entity = _find_entity(edge_cases_result, "Color", EntityType.CLASS)
        assert entity is not None
        assert entity.entity_type == EntityType.CLASS


# ===========================================================================
# Docstring Capture
# ===========================================================================


class TestDocstringCapture:
    """Docstrings are properly captured and cleaned."""

    def test_class_docstring(self, sample_result):
        entity = _find_entity(sample_result, "UserService", EntityType.CLASS)
        assert entity.docstring is not None
        assert "Service for managing users" in entity.docstring

    def test_method_docstring(self, sample_result):
        entity = _find_entity(sample_result, "get_user", EntityType.METHOD)
        assert entity.docstring is not None
        assert "Fetch a single user" in entity.docstring

    def test_function_docstring(self, sample_result):
        entity = _find_entity(sample_result, "process_data", EntityType.METHOD)
        assert entity.docstring is not None
        assert "Process a list of data items" in entity.docstring


# ===========================================================================
# Return Type Annotation Capture
# ===========================================================================


class TestReturnTypeAnnotation:
    """Return type annotations are captured."""

    def test_simple_return_type(self, sample_result):
        entity = _find_entity(sample_result, "process_data", EntityType.METHOD)
        assert entity.return_type is not None
        assert "int" in entity.return_type

    def test_generic_return_type(self, sample_result):
        entity = _find_entity(sample_result, "list_users", EntityType.METHOD)
        assert entity.return_type is not None
        assert "List" in entity.return_type or "list" in entity.return_type


# ===========================================================================
# CONTAINS Relationships
# ===========================================================================


class TestContainsRelationships:
    """File CONTAINS Class and File CONTAINS top-level Method."""

    def test_file_contains_class(self, sample_result):
        rel = _find_relationship(
            sample_result,
            RelationshipType.CONTAINS,
            target_name="UserService",
        )
        assert rel is not None

    def test_file_contains_function(self, sample_result):
        rel = _find_relationship(
            sample_result,
            RelationshipType.CONTAINS,
            target_name="process_data",
        )
        assert rel is not None


# ===========================================================================
# HAS_METHOD Relationships
# ===========================================================================


class TestHasMethodRelationships:
    """Class HAS_METHOD for non-__init__ methods."""

    def test_class_has_method(self, sample_result):
        rel = _find_relationship(
            sample_result,
            RelationshipType.HAS_METHOD,
            source_name="UserService",
            target_name="get_user",
        )
        assert rel is not None

    def test_class_has_async_method(self, sample_result):
        rel = _find_relationship(
            sample_result,
            RelationshipType.HAS_METHOD,
            source_name="UserService",
            target_name="list_users",
        )
        assert rel is not None


# ===========================================================================
# HAS_CONSTRUCTOR Relationships
# ===========================================================================


class TestHasConstructorRelationships:
    """Class HAS_CONSTRUCTOR for __init__ methods."""

    def test_class_has_constructor(self, sample_result):
        rel = _find_relationship(
            sample_result,
            RelationshipType.HAS_CONSTRUCTOR,
            source_name="UserService",
            target_name="__init__",
        )
        assert rel is not None


# ===========================================================================
# HAS_FIELD Relationships
# ===========================================================================


class TestHasFieldRelationships:
    """Class HAS_FIELD for self.x assignments."""

    def test_class_has_field(self, edge_cases_result):
        rel = _find_relationship(
            edge_cases_result,
            RelationshipType.HAS_FIELD,
            source_name="Calculator",
        )
        assert rel is not None

    def test_multiple_fields(self, edge_cases_result):
        rels = [
            r for r in edge_cases_result.relationships
            if r.relationship_type == RelationshipType.HAS_FIELD
            and "Calculator" in r.source_id
        ]
        assert len(rels) >= 3  # precision, history, _cache


# ===========================================================================
# CALLS Relationships
# ===========================================================================


class TestCallsRelationships:
    """Method calls inside function bodies produce CALLS relationships."""

    def test_method_calls_detected(self, edge_cases_result):
        # Calculator.add calls _compute and _store_result
        rel = _find_relationship(
            edge_cases_result,
            RelationshipType.CALLS,
            source_name="add",
            target_name="_compute",
        )
        assert rel is not None

    def test_another_method_call(self, edge_cases_result):
        rel = _find_relationship(
            edge_cases_result,
            RelationshipType.CALLS,
            source_name="add",
            target_name="_store_result",
        )
        assert rel is not None


# ===========================================================================
# Entity ID Format
# ===========================================================================


class TestEntityIdFormat:
    """Entity IDs follow the format {repository}::{module}.{Class}.{method}."""

    def test_class_entity_id(self, sample_result):
        entity = _find_entity(sample_result, "UserService", EntityType.CLASS)
        assert entity.id.startswith(f"{REPOSITORY}::")
        assert "UserService" in entity.id

    def test_method_entity_id(self, sample_result):
        entity = _find_entity(sample_result, "get_user", EntityType.METHOD)
        assert entity.id.startswith(f"{REPOSITORY}::")
        assert "UserService" in entity.id
        assert "get_user" in entity.id

    def test_top_level_function_entity_id(self, sample_result):
        entity = _find_entity(sample_result, "process_data", EntityType.METHOD)
        assert entity.id.startswith(f"{REPOSITORY}::")
        assert "process_data" in entity.id

    def test_constructor_entity_id(self, sample_result):
        entity = _find_entity(sample_result, "__init__", EntityType.CONSTRUCTOR)
        assert entity.id.startswith(f"{REPOSITORY}::")
        assert "UserService" in entity.id
        assert "__init__" in entity.id


# ===========================================================================
# Empty File
# ===========================================================================


class TestEmptyFile:
    """Empty file produces successful ParseResult with 0 entities."""

    def test_empty_file_success(self, empty_result):
        assert empty_result.success is True

    def test_empty_file_no_errors(self, empty_result):
        assert len(empty_result.errors) == 0

    def test_empty_file_zero_entities(self, empty_result):
        # Only the file entity itself may be present
        non_file_entities = [e for e in empty_result.entities if e.entity_type != EntityType.FILE]
        assert len(non_file_entities) == 0


# ===========================================================================
# Syntax Error File
# ===========================================================================


class TestSyntaxErrorFile:
    """File with syntax errors produces ParseResult with errors, success is False."""

    def test_syntax_error_not_success(self, syntax_error_result):
        assert syntax_error_result.success is False

    def test_syntax_error_has_errors(self, syntax_error_result):
        assert len(syntax_error_result.errors) > 0


# ===========================================================================
# Line Numbers
# ===========================================================================


class TestLineNumbers:
    """Line numbers are 1-indexed and line_end > line_number for multi-line entities."""

    def test_class_line_number_one_indexed(self, sample_result):
        entity = _find_entity(sample_result, "UserService", EntityType.CLASS)
        assert entity.line_number >= 1

    def test_method_line_number_one_indexed(self, sample_result):
        entity = _find_entity(sample_result, "get_user", EntityType.METHOD)
        assert entity.line_number >= 1

    def test_class_line_end_greater(self, sample_result):
        entity = _find_entity(sample_result, "UserService", EntityType.CLASS)
        assert entity.line_end is not None
        assert entity.line_end > entity.line_number

    def test_function_line_end_greater(self, sample_result):
        entity = _find_entity(sample_result, "process_data", EntityType.METHOD)
        assert entity.line_end is not None
        assert entity.line_end > entity.line_number
