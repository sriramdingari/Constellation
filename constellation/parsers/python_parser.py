"""Python parser using Tree-sitter."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node

from constellation.parsers.base import BaseParser, ParseResult
from constellation.models import CodeEntity, CodeRelationship, EntityType, RelationshipType

logger = logging.getLogger(__name__)

# =============================================================================
# Stereotype Detection Constants
# =============================================================================

PYDANTIC_BASES = frozenset({
    "BaseModel", "pydantic.BaseModel",
})

DJANGO_MODEL_BASES = frozenset({
    "Model", "models.Model", "django.db.models.Model",
})

ENUM_BASES = frozenset({
    "Enum", "enum.Enum", "IntEnum", "enum.IntEnum",
    "StrEnum", "enum.StrEnum", "Flag", "enum.Flag",
    "IntFlag", "enum.IntFlag",
})

CELERY_DECORATORS = frozenset({
    "task", "app.task", "shared_task", "celery.shared_task",
})

ENDPOINT_DECORATORS = frozenset({
    "api_view", "require_http_methods", "require_GET", "require_POST",
    "require_safe",
})

FASTAPI_DECORATORS = frozenset({
    "get", "post", "put", "delete", "patch", "options",
})

SPECIAL_METHOD_DECORATORS = {
    "abstractmethod": "abstract",
    "staticmethod": "static",
    "classmethod": "classmethod",
    "property": "property",
}

TEST_CLASS_BASES = frozenset({
    "TestCase", "unittest.TestCase", "django.test.TestCase",
})


class PythonParser(BaseParser):
    """Parser for Python source files using Tree-sitter."""

    def __init__(self) -> None:
        self._language = Language(tspython.language())
        self._parser = Parser(self._language)

    @property
    def language(self) -> str:
        return "python"

    @property
    def file_extensions(self) -> list[str]:
        return [".py"]

    def parse_file(self, file_path: Path, repository: str) -> ParseResult:
        """Parse a Python source file and return structured entities and relationships."""
        result = ParseResult(
            file_path=str(file_path),
            language=self.language,
        )

        try:
            code = file_path.read_bytes()
        except Exception as e:
            result.add_error(f"Failed to read file: {e}")
            return result

        # Empty file: return success with no entities
        if len(code.strip()) == 0:
            return result

        tree = self._parser.parse(code)

        # Check for syntax errors in the tree
        if tree.root_node.has_error:
            result.add_error(f"Syntax errors detected in {file_path}")
            return result

        # Derive module path from file_path
        module_name = self._derive_module_name(file_path)

        # Create file entity
        file_entity_id = f"{repository}::{file_path}"
        file_entity = CodeEntity(
            id=file_entity_id,
            name=file_path.name,
            entity_type=EntityType.FILE,
            repository=repository,
            file_path=str(file_path),
            line_number=1,
            language=self.language,
        )
        result.add_entity(file_entity)

        # Create parsing context
        ctx = _ParsingContext(
            file_path=str(file_path),
            code=code,
            module_name=module_name,
            repository=repository,
            file_entity_id=file_entity_id,
        )

        # Walk top-level definitions
        self._extract_definitions(tree.root_node, ctx, result)

        return result

    # ------------------------------------------------------------------
    # Module name derivation
    # ------------------------------------------------------------------

    def _derive_module_name(self, file_path: Path) -> str:
        """Derive a Python dotted module name from a file path."""
        parts: list[str] = []
        current = file_path

        if current.stem != "__init__":
            parts.append(current.stem)

        current = current.parent
        while current.name:
            init_file = current / "__init__.py"
            if init_file.exists():
                parts.append(current.name)
                current = current.parent
            else:
                break

        parts.reverse()
        return ".".join(parts) if parts else file_path.stem

    # ------------------------------------------------------------------
    # AST walking helpers
    # ------------------------------------------------------------------

    def _find_nodes(self, node: Node, type_name: str) -> Iterator[Node]:
        """Recursively find all nodes of a given type."""
        if node.type == type_name:
            yield node
        for child in node.children:
            yield from self._find_nodes(child, type_name)

    def _get_text(self, node: Node, code: bytes) -> str:
        """Get the text content of a tree-sitter node."""
        return code[node.start_byte:node.end_byte].decode("utf-8")

    # ------------------------------------------------------------------
    # Top-level extraction
    # ------------------------------------------------------------------

    def _extract_definitions(
        self,
        root: Node,
        ctx: _ParsingContext,
        result: ParseResult,
    ) -> None:
        """Extract class and function definitions from the root level."""
        for child in root.children:
            if child.type == "class_definition":
                self._process_class(child, ctx, result, decorators=[])
            elif child.type == "function_definition":
                self._process_top_level_function(child, ctx, result, decorators=[])
            elif child.type == "decorated_definition":
                self._process_decorated_definition(child, ctx, result, class_ctx=None)

    def _process_decorated_definition(
        self,
        node: Node,
        ctx: _ParsingContext,
        result: ParseResult,
        class_ctx: _ClassContext | None,
    ) -> None:
        """Process a decorated definition (class or function)."""
        decorators: list[str] = []
        for child in node.children:
            if child.type == "decorator":
                dec_text = self._get_text(child, ctx.code).lstrip("@").strip()
                decorators.append(dec_text)
            elif child.type == "class_definition":
                self._process_class(child, ctx, result, decorators=decorators)
            elif child.type == "function_definition":
                if class_ctx is not None:
                    self._process_method(child, ctx, class_ctx, result, decorators=decorators)
                else:
                    self._process_top_level_function(child, ctx, result, decorators=decorators)

    # ------------------------------------------------------------------
    # Class processing
    # ------------------------------------------------------------------

    def _process_class(
        self,
        node: Node,
        ctx: _ParsingContext,
        result: ParseResult,
        decorators: list[str],
    ) -> None:
        """Process a class definition node."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        class_name = self._get_text(name_node, ctx.code)
        full_name = f"{ctx.module_name}.{class_name}" if ctx.module_name else class_name
        entity_id = f"{ctx.repository}::{full_name}"

        # Extract base classes
        bases = self._extract_bases(node, ctx.code)

        # Extract docstring
        body_node = node.child_by_field_name("body")
        docstring = self._extract_docstring(body_node, ctx.code) if body_node else None

        # Build signature
        base_str = f"({', '.join(bases)})" if bases else ""
        signature = f"class {class_name}{base_str}"

        # Determine stereotypes
        stereotypes = self._detect_class_stereotypes(class_name, bases, decorators)

        code = self._get_text(node, ctx.code)

        class_entity = CodeEntity(
            id=entity_id,
            name=class_name,
            entity_type=EntityType.CLASS,
            repository=ctx.repository,
            file_path=ctx.file_path,
            line_number=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            language=self.language,
            signature=signature,
            docstring=docstring,
            code=code,
            stereotypes=stereotypes,
        )
        result.add_entity(class_entity)

        # File CONTAINS Class
        result.add_relationship(CodeRelationship(
            source_id=ctx.file_entity_id,
            target_id=entity_id,
            relationship_type=RelationshipType.CONTAINS,
        ))

        # Process class body
        if body_node:
            class_ctx = _ClassContext(
                class_name=class_name,
                class_full_name=full_name,
                class_entity_id=entity_id,
                bases=bases,
                decorators=decorators,
            )
            self._process_class_body(body_node, ctx, class_ctx, result)

    def _process_class_body(
        self,
        body: Node,
        ctx: _ParsingContext,
        class_ctx: _ClassContext,
        result: ParseResult,
    ) -> None:
        """Process the body of a class definition."""
        for child in body.children:
            if child.type == "function_definition":
                self._process_method(child, ctx, class_ctx, result, decorators=[])
            elif child.type == "decorated_definition":
                self._process_decorated_definition(child, ctx, result, class_ctx=class_ctx)
            elif child.type == "class_definition":
                # Nested/inner class
                self._process_class(child, ctx, result, decorators=[])

    # ------------------------------------------------------------------
    # Method processing
    # ------------------------------------------------------------------

    def _process_method(
        self,
        node: Node,
        ctx: _ParsingContext,
        class_ctx: _ClassContext,
        result: ParseResult,
        decorators: list[str],
    ) -> None:
        """Process a method definition inside a class."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        method_name = self._get_text(name_node, ctx.code)
        full_name = f"{class_ctx.class_full_name}.{method_name}"
        entity_id = f"{ctx.repository}::{full_name}"
        is_constructor = method_name == "__init__"

        # Detect async
        is_async = self._is_async(node)

        # Extract return type
        return_type = self._extract_return_type(node, ctx.code)

        # Build signature
        signature = self._build_signature(node, ctx.code, method_name, return_type)

        # Extract docstring
        body_node = node.child_by_field_name("body")
        docstring = self._extract_docstring(body_node, ctx.code) if body_node else None

        code = self._get_text(node, ctx.code)

        # Build modifiers
        modifiers = self._extract_special_modifiers(decorators)
        if is_async:
            modifiers.append("async")

        # Determine entity type
        entity_type = EntityType.CONSTRUCTOR if is_constructor else EntityType.METHOD

        # Determine stereotypes
        stereotypes: list[str] = []
        if method_name.startswith("test_"):
            stereotypes.append("test")

        method_entity = CodeEntity(
            id=entity_id,
            name=method_name,
            entity_type=entity_type,
            repository=ctx.repository,
            file_path=ctx.file_path,
            line_number=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            language=self.language,
            code=code,
            signature=signature,
            return_type=return_type,
            docstring=docstring,
            modifiers=modifiers,
            stereotypes=stereotypes,
        )
        result.add_entity(method_entity)

        # Relationship: Class -> Method / Constructor
        if is_constructor:
            result.add_relationship(CodeRelationship(
                source_id=class_ctx.class_entity_id,
                target_id=entity_id,
                relationship_type=RelationshipType.HAS_CONSTRUCTOR,
            ))
            # Extract fields from __init__ body
            if body_node:
                self._extract_fields_from_init(body_node, ctx, class_ctx, result)
        else:
            result.add_relationship(CodeRelationship(
                source_id=class_ctx.class_entity_id,
                target_id=entity_id,
                relationship_type=RelationshipType.HAS_METHOD,
            ))

        # Extract CALLS relationships
        if body_node:
            self._extract_calls(body_node, ctx, entity_id, class_ctx, result)

    # ------------------------------------------------------------------
    # Top-level function processing
    # ------------------------------------------------------------------

    def _process_top_level_function(
        self,
        node: Node,
        ctx: _ParsingContext,
        result: ParseResult,
        decorators: list[str],
    ) -> None:
        """Process a top-level (module-level) function definition."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        func_name = self._get_text(name_node, ctx.code)
        full_name = f"{ctx.module_name}.{func_name}" if ctx.module_name else func_name
        entity_id = f"{ctx.repository}::{full_name}"

        # Detect async
        is_async = self._is_async(node)

        # Extract return type
        return_type = self._extract_return_type(node, ctx.code)

        # Build signature
        signature = self._build_signature(node, ctx.code, func_name, return_type)

        # Extract docstring
        body_node = node.child_by_field_name("body")
        docstring = self._extract_docstring(body_node, ctx.code) if body_node else None

        code = self._get_text(node, ctx.code)

        # Build modifiers
        modifiers = self._extract_special_modifiers(decorators)
        if is_async:
            modifiers.append("async")

        # Determine stereotypes
        stereotypes = self._detect_function_stereotypes(func_name, decorators)

        method_entity = CodeEntity(
            id=entity_id,
            name=func_name,
            entity_type=EntityType.METHOD,
            repository=ctx.repository,
            file_path=ctx.file_path,
            line_number=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            language=self.language,
            code=code,
            signature=signature,
            return_type=return_type,
            docstring=docstring,
            modifiers=modifiers,
            stereotypes=stereotypes,
        )
        result.add_entity(method_entity)

        # File CONTAINS Function
        result.add_relationship(CodeRelationship(
            source_id=ctx.file_entity_id,
            target_id=entity_id,
            relationship_type=RelationshipType.CONTAINS,
        ))

        # Extract CALLS relationships
        if body_node:
            self._extract_calls(body_node, ctx, entity_id, None, result)

    # ------------------------------------------------------------------
    # Field extraction from __init__
    # ------------------------------------------------------------------

    def _extract_fields_from_init(
        self,
        body_node: Node,
        ctx: _ParsingContext,
        class_ctx: _ClassContext,
        result: ParseResult,
    ) -> None:
        """Extract self.x = ... assignments from an __init__ body as FIELD entities."""
        for assign_node in self._find_nodes(body_node, "assignment"):
            left = assign_node.child_by_field_name("left")
            if left is None or left.type != "attribute":
                continue

            # Check it's self.something
            obj_node = left.child_by_field_name("object")
            attr_node = left.child_by_field_name("attribute")
            if obj_node is None or attr_node is None:
                continue
            if self._get_text(obj_node, ctx.code) != "self":
                continue

            field_name = self._get_text(attr_node, ctx.code)
            field_full_name = f"{class_ctx.class_full_name}.{field_name}"
            field_entity_id = f"{ctx.repository}::{field_full_name}"

            field_entity = CodeEntity(
                id=field_entity_id,
                name=field_name,
                entity_type=EntityType.FIELD,
                repository=ctx.repository,
                file_path=ctx.file_path,
                line_number=assign_node.start_point[0] + 1,
                line_end=assign_node.end_point[0] + 1,
                language=self.language,
            )
            result.add_entity(field_entity)

            result.add_relationship(CodeRelationship(
                source_id=class_ctx.class_entity_id,
                target_id=field_entity_id,
                relationship_type=RelationshipType.HAS_FIELD,
            ))

    # ------------------------------------------------------------------
    # CALLS extraction
    # ------------------------------------------------------------------

    def _extract_calls(
        self,
        body_node: Node,
        ctx: _ParsingContext,
        caller_entity_id: str,
        class_ctx: _ClassContext | None,
        result: ParseResult,
    ) -> None:
        """Extract function/method calls from a function body."""
        seen: set[str] = set()

        for call_node in self._find_nodes(body_node, "call"):
            func_node = call_node.child_by_field_name("function")
            if func_node is None:
                continue

            called_name: str | None = None

            if func_node.type == "identifier":
                called_name = self._get_text(func_node, ctx.code)
            elif func_node.type == "attribute":
                obj_node = func_node.child_by_field_name("object")
                attr_node = func_node.child_by_field_name("attribute")
                if obj_node and attr_node:
                    obj_text = self._get_text(obj_node, ctx.code)
                    method_name = self._get_text(attr_node, ctx.code)
                    if obj_text == "self" and class_ctx:
                        called_name = f"{class_ctx.class_full_name}.{method_name}"
                    else:
                        called_name = f"{obj_text}.{method_name}"

            if called_name and called_name not in seen:
                seen.add(called_name)
                target_id = f"{ctx.repository}::{called_name}"
                result.add_relationship(CodeRelationship(
                    source_id=caller_entity_id,
                    target_id=target_id,
                    relationship_type=RelationshipType.CALLS,
                ))

    # ------------------------------------------------------------------
    # Base class extraction
    # ------------------------------------------------------------------

    def _extract_bases(self, class_node: Node, code: bytes) -> list[str]:
        """Extract base class names from a class definition."""
        bases: list[str] = []
        superclass_node = class_node.child_by_field_name("superclasses")
        if superclass_node:
            for child in superclass_node.children:
                if child.type in ("identifier", "attribute"):
                    bases.append(self._get_text(child, code))
        return bases

    # ------------------------------------------------------------------
    # Docstring extraction
    # ------------------------------------------------------------------

    def _extract_docstring(self, body_node: Node, code: bytes) -> str | None:
        """Extract the docstring from the first expression statement in a body."""
        if not body_node or not body_node.children:
            return None

        first_stmt = None
        for child in body_node.children:
            if child.type == "expression_statement":
                first_stmt = child
                break
            elif child.type not in ("comment",):
                break

        if not first_stmt:
            return None

        for child in first_stmt.children:
            if child.type == "string":
                raw = self._get_text(child, code)
                return self._clean_docstring(raw)

        return None

    def _clean_docstring(self, docstring: str) -> str:
        """Strip quote delimiters and normalize whitespace in a docstring."""
        if docstring.startswith('"""') and docstring.endswith('"""'):
            docstring = docstring[3:-3]
        elif docstring.startswith("'''") and docstring.endswith("'''"):
            docstring = docstring[3:-3]
        elif docstring.startswith('"') and docstring.endswith('"'):
            docstring = docstring[1:-1]
        elif docstring.startswith("'") and docstring.endswith("'"):
            docstring = docstring[1:-1]

        lines = docstring.split("\n")
        cleaned = [line.strip() for line in lines]

        while cleaned and not cleaned[0]:
            cleaned.pop(0)
        while cleaned and not cleaned[-1]:
            cleaned.pop()

        return "\n".join(cleaned)

    # ------------------------------------------------------------------
    # Return type, signature, async helpers
    # ------------------------------------------------------------------

    def _extract_return_type(self, node: Node, code: bytes) -> str | None:
        """Extract the return type annotation from a function definition."""
        return_type_node = node.child_by_field_name("return_type")
        if return_type_node:
            return self._get_text(return_type_node, code)
        return None

    def _build_signature(
        self,
        node: Node,
        code: bytes,
        func_name: str,
        return_type: str | None,
    ) -> str:
        """Build a function signature string."""
        params_node = node.child_by_field_name("parameters")
        if params_node:
            params_text = self._get_text(params_node, code)
        else:
            params_text = "()"

        sig = f"def {func_name}{params_text}"
        if return_type:
            sig += f" -> {return_type}"
        return sig

    def _is_async(self, node: Node) -> bool:
        """Check whether a function definition is async."""
        for child in node.children:
            if child.type == "async":
                return True
        return False

    # ------------------------------------------------------------------
    # Stereotype detection
    # ------------------------------------------------------------------

    def _detect_class_stereotypes(
        self,
        class_name: str,
        bases: list[str],
        decorators: list[str],
    ) -> list[str]:
        """Detect stereotypes for a class based on name, bases, and decorators."""
        stereotypes: list[str] = []

        # Test class: name starts with "Test" or inherits TestCase
        if class_name.startswith("Test"):
            stereotypes.append("test")
        else:
            base_names = {b.split(".")[-1] for b in bases}
            if base_names & TEST_CLASS_BASES:
                stereotypes.append("test")

        # Django model
        if any(b in DJANGO_MODEL_BASES for b in bases):
            stereotypes.append("django_model")

        # Pydantic model
        if any(b in PYDANTIC_BASES for b in bases):
            stereotypes.append("pydantic_model")

        # Enum
        if any(b in ENUM_BASES for b in bases):
            stereotypes.append("enum")

        return stereotypes

    def _detect_function_stereotypes(
        self,
        func_name: str,
        decorators: list[str],
    ) -> list[str]:
        """Detect stereotypes for a top-level function."""
        stereotypes: list[str] = []

        # Test function
        if func_name.startswith("test_"):
            stereotypes.append("test")

        # Extract bare decorator names (strip arguments like "(..." and leading "@")
        dec_names = [d.split("(")[0].strip().split(".")[-1] for d in decorators]
        dec_full_names = [d.split("(")[0].strip() for d in decorators]

        # Celery task
        if any(d in CELERY_DECORATORS for d in dec_full_names) or any(d in CELERY_DECORATORS for d in dec_names):
            stereotypes.append("celery_task")

        # Endpoint (api_view, etc.)
        if any(d in ENDPOINT_DECORATORS for d in dec_names) or any(d in ENDPOINT_DECORATORS for d in dec_full_names):
            stereotypes.append("endpoint")

        return stereotypes

    # ------------------------------------------------------------------
    # Modifier extraction
    # ------------------------------------------------------------------

    def _extract_special_modifiers(self, decorators: list[str]) -> list[str]:
        """Extract special modifiers like static, classmethod, property, abstract."""
        modifiers: list[str] = []
        for dec in decorators:
            dec_name = dec.split("(")[0].strip().split(".")[-1]
            if dec_name in SPECIAL_METHOD_DECORATORS:
                modifiers.append(SPECIAL_METHOD_DECORATORS[dec_name])
        return modifiers


# =============================================================================
# Internal context dataclasses
# =============================================================================


class _ParsingContext:
    """Holds mutable state during parsing of a single file."""

    __slots__ = ("file_path", "code", "module_name", "repository", "file_entity_id")

    def __init__(
        self,
        file_path: str,
        code: bytes,
        module_name: str,
        repository: str,
        file_entity_id: str,
    ) -> None:
        self.file_path = file_path
        self.code = code
        self.module_name = module_name
        self.repository = repository
        self.file_entity_id = file_entity_id


class _ClassContext:
    """Context for the class currently being processed."""

    __slots__ = ("class_name", "class_full_name", "class_entity_id", "bases", "decorators")

    def __init__(
        self,
        class_name: str,
        class_full_name: str,
        class_entity_id: str,
        bases: list[str],
        decorators: list[str],
    ) -> None:
        self.class_name = class_name
        self.class_full_name = class_full_name
        self.class_entity_id = class_entity_id
        self.bases = bases
        self.decorators = decorators
