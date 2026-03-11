"""C#/.NET parser using Tree-sitter."""

from __future__ import annotations

import logging
from pathlib import Path

import tree_sitter_c_sharp as tscsharp
from tree_sitter import Language, Parser, Node

from constellation.models import CodeEntity, CodeRelationship, EntityType, RelationshipType
from constellation.parsers.base import BaseParser, ParseResult

logger = logging.getLogger(__name__)

# =============================================================================
# Test Detection Constants
# =============================================================================

TEST_METHOD_ATTRIBUTES = frozenset({
    # NUnit
    "Test", "TestCase", "TestCaseSource", "Theory",
    # xUnit
    "Fact", "InlineData", "MemberData", "ClassData",
    # MSTest
    "TestMethod", "DataTestMethod",
})

CS_LANGUAGE = Language(tscsharp.language())


# =============================================================================
# DotNet Parser
# =============================================================================


class DotNetParser(BaseParser):
    """Parser for C# source files using Tree-sitter."""

    def __init__(self) -> None:
        self._parser = Parser(CS_LANGUAGE)

    @property
    def language(self) -> str:
        return "csharp"

    @property
    def file_extensions(self) -> list[str]:
        return [".cs"]

    # =========================================================================
    # Main Entry Point
    # =========================================================================

    def parse_file(self, file_path: Path, repository: str) -> ParseResult:
        """Parse a C# source file and return structured results."""
        result = ParseResult(file_path=str(file_path), language=self.language)

        try:
            code = file_path.read_bytes()
            tree = self._parser.parse(code)
        except Exception as exc:
            result.add_error(f"Failed to parse file: {exc}")
            return result

        # Check for syntax errors
        if tree.root_node.has_error:
            result.add_error(f"Syntax errors detected in {file_path}")

        # Create File entity
        file_entity = CodeEntity(
            id=f"{repository}::{file_path}",
            name=file_path.name,
            entity_type=EntityType.FILE,
            repository=repository,
            file_path=str(file_path),
            line_number=1,
            language=self.language,
        )
        result.add_entity(file_entity)

        # Extract namespace and types
        self._process_root(tree.root_node, code, repository, str(file_path), file_entity, result)

        return result

    # =========================================================================
    # Root Processing
    # =========================================================================

    def _process_root(
        self,
        root: Node,
        code: bytes,
        repository: str,
        file_path: str,
        file_entity: CodeEntity,
        result: ParseResult,
    ) -> None:
        """Process the root of the AST to find namespaces and types."""
        # Process block-scoped namespace declarations
        for child in root.children:
            if child.type == "namespace_declaration":
                self._process_namespace(child, code, repository, file_path, file_entity, result, parent_namespace="")
            elif child.type == "file_scoped_namespace_declaration":
                self._process_file_scoped_namespace(child, code, repository, file_path, file_entity, result)

        # Process top-level types (no namespace)
        has_ns = any(
            c.type in ("namespace_declaration", "file_scoped_namespace_declaration")
            for c in root.children
        )
        if not has_ns:
            for child in root.children:
                self._process_type_node(child, code, repository, file_path, file_entity, result, namespace="")

    # =========================================================================
    # Namespace Processing
    # =========================================================================

    def _process_namespace(
        self,
        node: Node,
        code: bytes,
        repository: str,
        file_path: str,
        file_entity: CodeEntity,
        result: ParseResult,
        parent_namespace: str,
    ) -> None:
        """Process a block-scoped namespace declaration."""
        ns_name = self._get_namespace_name(node, code)
        if not ns_name:
            return

        full_ns = f"{parent_namespace}.{ns_name}" if parent_namespace else ns_name

        # Create PACKAGE entity for the namespace
        ns_entity = CodeEntity(
            id=f"{repository}::{full_ns}",
            name=full_ns.split(".")[-1],
            entity_type=EntityType.PACKAGE,
            repository=repository,
            file_path=file_path,
            line_number=node.start_point[0] + 1,
            language=self.language,
        )
        result.add_entity(ns_entity)

        # Process declarations inside the namespace body
        decl_list = self._find_child_by_type(node, "declaration_list")
        if decl_list:
            for child in decl_list.children:
                if child.type == "namespace_declaration":
                    self._process_namespace(child, code, repository, file_path, file_entity, result, parent_namespace=full_ns)
                else:
                    self._process_type_node(child, code, repository, file_path, file_entity, result, namespace=full_ns)

    def _process_file_scoped_namespace(
        self,
        node: Node,
        code: bytes,
        repository: str,
        file_path: str,
        file_entity: CodeEntity,
        result: ParseResult,
    ) -> None:
        """Process a file-scoped namespace declaration (C# 10+)."""
        ns_name = self._get_namespace_name(node, code)
        if not ns_name:
            return

        # Create PACKAGE entity
        ns_entity = CodeEntity(
            id=f"{repository}::{ns_name}",
            name=ns_name.split(".")[-1],
            entity_type=EntityType.PACKAGE,
            repository=repository,
            file_path=file_path,
            line_number=node.start_point[0] + 1,
            language=self.language,
        )
        result.add_entity(ns_entity)

        # File-scoped: all types declared after the namespace directive are in this namespace
        # They appear as children of the file_scoped_namespace_declaration node
        for child in node.children:
            self._process_type_node(child, code, repository, file_path, file_entity, result, namespace=ns_name)

    def _get_namespace_name(self, node: Node, code: bytes) -> str | None:
        """Extract namespace name from a namespace declaration node."""
        for child in node.children:
            if child.type in ("qualified_name", "identifier"):
                return self._get_text(child, code)
        return None

    # =========================================================================
    # Type Routing
    # =========================================================================

    def _process_type_node(
        self,
        node: Node,
        code: bytes,
        repository: str,
        file_path: str,
        file_entity: CodeEntity,
        result: ParseResult,
        namespace: str,
        outer_class_entity: CodeEntity | None = None,
    ) -> None:
        """Route a type declaration node to the appropriate handler."""
        if node.type == "class_declaration":
            self._process_class(node, code, repository, file_path, file_entity, result, namespace, outer_class_entity)
        elif node.type == "interface_declaration":
            self._process_interface(node, code, repository, file_path, file_entity, result, namespace, outer_class_entity)
        elif node.type == "enum_declaration":
            self._process_enum(node, code, repository, file_path, file_entity, result, namespace)

    # =========================================================================
    # Class Processing
    # =========================================================================

    def _process_class(
        self,
        node: Node,
        code: bytes,
        repository: str,
        file_path: str,
        file_entity: CodeEntity,
        result: ParseResult,
        namespace: str,
        outer_class_entity: CodeEntity | None = None,
    ) -> None:
        """Process a class declaration."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        class_name = self._get_text(name_node, code)
        full_name = f"{namespace}.{class_name}" if namespace else class_name

        modifiers = self._extract_modifiers(node, code)
        docstring = self._extract_docstring(node, code)
        class_code = self._get_text(node, code)

        class_entity = CodeEntity(
            id=f"{repository}::{full_name}",
            name=class_name,
            entity_type=EntityType.CLASS,
            repository=repository,
            file_path=file_path,
            line_number=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            language=self.language,
            modifiers=modifiers,
            docstring=docstring,
            code=class_code,
        )
        result.add_entity(class_entity)

        # File CONTAINS class
        result.add_relationship(CodeRelationship(
            source_id=file_entity.id,
            target_id=class_entity.id,
            relationship_type=RelationshipType.CONTAINS,
        ))

        # IN_PACKAGE relationship
        if namespace:
            result.add_relationship(CodeRelationship(
                source_id=class_entity.id,
                target_id=f"{repository}::{namespace}",
                relationship_type=RelationshipType.IN_PACKAGE,
            ))

        # DECLARES relationship for nested classes
        if outer_class_entity is not None:
            result.add_relationship(CodeRelationship(
                source_id=outer_class_entity.id,
                target_id=class_entity.id,
                relationship_type=RelationshipType.DECLARES,
            ))

        # Process base types (EXTENDS / IMPLEMENTS)
        self._extract_base_types(node, code, class_entity, result, is_interface=False)

        # Process class body
        body_node = self._find_child_by_type(node, "declaration_list")
        if body_node:
            self._process_class_body(body_node, code, repository, file_path, file_entity, class_entity, result, namespace)

    # =========================================================================
    # Interface Processing
    # =========================================================================

    def _process_interface(
        self,
        node: Node,
        code: bytes,
        repository: str,
        file_path: str,
        file_entity: CodeEntity,
        result: ParseResult,
        namespace: str,
        outer_class_entity: CodeEntity | None = None,
    ) -> None:
        """Process an interface declaration."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        iface_name = self._get_text(name_node, code)
        full_name = f"{namespace}.{iface_name}" if namespace else iface_name

        modifiers = self._extract_modifiers(node, code)
        docstring = self._extract_docstring(node, code)

        iface_entity = CodeEntity(
            id=f"{repository}::{full_name}",
            name=iface_name,
            entity_type=EntityType.INTERFACE,
            repository=repository,
            file_path=file_path,
            line_number=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            language=self.language,
            modifiers=modifiers,
            docstring=docstring,
        )
        result.add_entity(iface_entity)

        # File CONTAINS interface
        result.add_relationship(CodeRelationship(
            source_id=file_entity.id,
            target_id=iface_entity.id,
            relationship_type=RelationshipType.CONTAINS,
        ))

        # IN_PACKAGE
        if namespace:
            result.add_relationship(CodeRelationship(
                source_id=iface_entity.id,
                target_id=f"{repository}::{namespace}",
                relationship_type=RelationshipType.IN_PACKAGE,
            ))

        # DECLARES for nested interface
        if outer_class_entity is not None:
            result.add_relationship(CodeRelationship(
                source_id=outer_class_entity.id,
                target_id=iface_entity.id,
                relationship_type=RelationshipType.DECLARES,
            ))

        # Process interface body (methods)
        body_node = self._find_child_by_type(node, "declaration_list")
        if body_node:
            for child in body_node.children:
                if child.type == "method_declaration":
                    self._process_method(child, code, repository, file_path, iface_entity, result, namespace)

    # =========================================================================
    # Enum Processing
    # =========================================================================

    def _process_enum(
        self,
        node: Node,
        code: bytes,
        repository: str,
        file_path: str,
        file_entity: CodeEntity,
        result: ParseResult,
        namespace: str,
    ) -> None:
        """Process an enum declaration as a CLASS entity with 'enum' stereotype."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        enum_name = self._get_text(name_node, code)
        full_name = f"{namespace}.{enum_name}" if namespace else enum_name

        modifiers = self._extract_modifiers(node, code)
        docstring = self._extract_docstring(node, code)

        enum_entity = CodeEntity(
            id=f"{repository}::{full_name}",
            name=enum_name,
            entity_type=EntityType.CLASS,
            repository=repository,
            file_path=file_path,
            line_number=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            language=self.language,
            modifiers=modifiers,
            docstring=docstring,
            stereotypes=["enum"],
        )
        result.add_entity(enum_entity)

        # File CONTAINS enum
        result.add_relationship(CodeRelationship(
            source_id=file_entity.id,
            target_id=enum_entity.id,
            relationship_type=RelationshipType.CONTAINS,
        ))

        # IN_PACKAGE
        if namespace:
            result.add_relationship(CodeRelationship(
                source_id=enum_entity.id,
                target_id=f"{repository}::{namespace}",
                relationship_type=RelationshipType.IN_PACKAGE,
            ))

    # =========================================================================
    # Class Body Processing
    # =========================================================================

    def _process_class_body(
        self,
        body: Node,
        code: bytes,
        repository: str,
        file_path: str,
        file_entity: CodeEntity,
        class_entity: CodeEntity,
        result: ParseResult,
        namespace: str,
    ) -> None:
        """Process members inside a class body."""
        for child in body.children:
            if child.type == "method_declaration":
                self._process_method(child, code, repository, file_path, class_entity, result, namespace)
            elif child.type == "constructor_declaration":
                self._process_constructor(child, code, repository, file_path, class_entity, result, namespace)
            elif child.type == "field_declaration":
                self._process_field(child, code, repository, file_path, class_entity, result, namespace)
            elif child.type == "property_declaration":
                self._process_property(child, code, repository, file_path, class_entity, result, namespace)
            elif child.type == "class_declaration":
                # Nested class
                self._process_class(child, code, repository, file_path, file_entity, result, namespace, outer_class_entity=class_entity)
            elif child.type == "interface_declaration":
                # Nested interface
                self._process_interface(child, code, repository, file_path, file_entity, result, namespace, outer_class_entity=class_entity)
            elif child.type == "enum_declaration":
                # Nested enum
                self._process_enum(child, code, repository, file_path, file_entity, result, namespace)

    # =========================================================================
    # Method Processing
    # =========================================================================

    def _process_method(
        self,
        node: Node,
        code: bytes,
        repository: str,
        file_path: str,
        class_entity: CodeEntity,
        result: ParseResult,
        namespace: str,
    ) -> None:
        """Process a method declaration."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        method_name = self._get_text(name_node, code)
        class_full_name = class_entity.id.split("::", 1)[1] if "::" in class_entity.id else class_entity.name
        full_name = f"{class_full_name}.{method_name}"

        # Return type
        return_type = self._extract_return_type(node, code)

        # Signature
        params = self._extract_parameters(node, code)
        param_str = ", ".join(f"{p['type']} {p['name']}" for p in params)
        signature = f"{return_type or 'void'} {method_name}({param_str})"

        modifiers = self._extract_modifiers(node, code)
        docstring = self._extract_docstring(node, code)
        method_code = self._get_text(node, code)

        # Detect test stereotype via attributes
        attributes = self._extract_attributes(node, code)
        stereotypes: list[str] = []
        attr_names = {a["name"] for a in attributes}
        if attr_names & TEST_METHOD_ATTRIBUTES:
            stereotypes.append("test")

        method_entity = CodeEntity(
            id=f"{repository}::{full_name}",
            name=method_name,
            entity_type=EntityType.METHOD,
            repository=repository,
            file_path=file_path,
            line_number=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            language=self.language,
            signature=signature,
            return_type=return_type,
            modifiers=modifiers,
            docstring=docstring,
            code=method_code,
            stereotypes=stereotypes,
        )
        result.add_entity(method_entity)

        # HAS_METHOD
        result.add_relationship(CodeRelationship(
            source_id=class_entity.id,
            target_id=method_entity.id,
            relationship_type=RelationshipType.HAS_METHOD,
        ))

    # =========================================================================
    # Constructor Processing
    # =========================================================================

    def _process_constructor(
        self,
        node: Node,
        code: bytes,
        repository: str,
        file_path: str,
        class_entity: CodeEntity,
        result: ParseResult,
        namespace: str,
    ) -> None:
        """Process a constructor declaration."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        ctor_name = self._get_text(name_node, code)
        class_full_name = class_entity.id.split("::", 1)[1] if "::" in class_entity.id else class_entity.name
        full_name = f"{class_full_name}.{ctor_name}"

        params = self._extract_parameters(node, code)
        param_str = ", ".join(f"{p['type']} {p['name']}" for p in params)
        signature = f"{ctor_name}({param_str})"

        modifiers = self._extract_modifiers(node, code)
        docstring = self._extract_docstring(node, code)
        ctor_code = self._get_text(node, code)

        ctor_entity = CodeEntity(
            id=f"{repository}::{full_name}",
            name=ctor_name,
            entity_type=EntityType.CONSTRUCTOR,
            repository=repository,
            file_path=file_path,
            line_number=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            language=self.language,
            signature=signature,
            modifiers=modifiers,
            docstring=docstring,
            code=ctor_code,
        )
        result.add_entity(ctor_entity)

        # HAS_CONSTRUCTOR
        result.add_relationship(CodeRelationship(
            source_id=class_entity.id,
            target_id=ctor_entity.id,
            relationship_type=RelationshipType.HAS_CONSTRUCTOR,
        ))

    # =========================================================================
    # Field Processing
    # =========================================================================

    def _process_field(
        self,
        node: Node,
        code: bytes,
        repository: str,
        file_path: str,
        class_entity: CodeEntity,
        result: ParseResult,
        namespace: str,
    ) -> None:
        """Process a field declaration (may declare multiple variables)."""
        modifiers = self._extract_modifiers(node, code)

        # Find variable_declaration child
        var_decl = self._find_child_by_type(node, "variable_declaration")
        if not var_decl:
            return

        # Get field type (may be on variable_declaration or on field_declaration)
        field_type = None
        type_node = var_decl.child_by_field_name("type")
        if not type_node:
            type_node = node.child_by_field_name("type")
        if type_node:
            field_type = self._extract_type_name(type_node, code)

        # Process each variable declarator
        for child in var_decl.children:
            if child.type == "variable_declarator":
                name_node = child.child_by_field_name("name")
                if not name_node:
                    # Fallback: find identifier child
                    name_node = self._find_child_by_type(child, "identifier")
                if not name_node:
                    continue

                field_name = self._get_text(name_node, code)
                class_full_name = class_entity.id.split("::", 1)[1] if "::" in class_entity.id else class_entity.name
                full_name = f"{class_full_name}.{field_name}"

                field_entity = CodeEntity(
                    id=f"{repository}::{full_name}",
                    name=field_name,
                    entity_type=EntityType.FIELD,
                    repository=repository,
                    file_path=file_path,
                    line_number=node.start_point[0] + 1,
                    language=self.language,
                    return_type=field_type,
                    modifiers=modifiers,
                )
                result.add_entity(field_entity)

                # HAS_FIELD
                result.add_relationship(CodeRelationship(
                    source_id=class_entity.id,
                    target_id=field_entity.id,
                    relationship_type=RelationshipType.HAS_FIELD,
                ))

    # =========================================================================
    # Property Processing (as Field)
    # =========================================================================

    def _process_property(
        self,
        node: Node,
        code: bytes,
        repository: str,
        file_path: str,
        class_entity: CodeEntity,
        result: ParseResult,
        namespace: str,
    ) -> None:
        """Process a property declaration as a FIELD entity."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        prop_name = self._get_text(name_node, code)
        class_full_name = class_entity.id.split("::", 1)[1] if "::" in class_entity.id else class_entity.name
        full_name = f"{class_full_name}.{prop_name}"

        modifiers = self._extract_modifiers(node, code)
        modifiers.append("property")

        prop_type = None
        type_node = node.child_by_field_name("type")
        if type_node:
            prop_type = self._extract_type_name(type_node, code)

        prop_entity = CodeEntity(
            id=f"{repository}::{full_name}",
            name=prop_name,
            entity_type=EntityType.FIELD,
            repository=repository,
            file_path=file_path,
            line_number=node.start_point[0] + 1,
            language=self.language,
            return_type=prop_type,
            modifiers=modifiers,
        )
        result.add_entity(prop_entity)

        # HAS_FIELD
        result.add_relationship(CodeRelationship(
            source_id=class_entity.id,
            target_id=prop_entity.id,
            relationship_type=RelationshipType.HAS_FIELD,
        ))

    # =========================================================================
    # Base Type Extraction (EXTENDS / IMPLEMENTS)
    # =========================================================================

    def _extract_base_types(
        self,
        node: Node,
        code: bytes,
        class_entity: CodeEntity,
        result: ParseResult,
        is_interface: bool,
    ) -> None:
        """Extract EXTENDS and IMPLEMENTS relationships from base_list."""
        for child in node.children:
            if child.type != "base_list":
                continue

            is_first = True
            for base_child in child.children:
                base_name = self._extract_base_type_name(base_child, code)
                if not base_name:
                    continue

                # Convention: first non-I-prefixed type is base class (for classes)
                if (
                    is_first
                    and not is_interface
                    and not base_name.startswith("I")
                ):
                    result.add_relationship(CodeRelationship(
                        source_id=class_entity.id,
                        target_id=f"external::{base_name}",
                        relationship_type=RelationshipType.EXTENDS,
                    ))
                else:
                    result.add_relationship(CodeRelationship(
                        source_id=class_entity.id,
                        target_id=f"external::{base_name}",
                        relationship_type=RelationshipType.IMPLEMENTS,
                    ))
                is_first = False

    def _extract_base_type_name(self, node: Node, code: bytes) -> str | None:
        """Extract a base type name from a base_list child node."""
        if node.type in ("identifier", "qualified_name", "generic_name"):
            return self._extract_type_name(node, code)
        if node.type == "simple_base_type":
            for child in node.children:
                if child.type in ("identifier", "qualified_name", "generic_name"):
                    return self._extract_type_name(child, code)
        return None

    # =========================================================================
    # Helper: Extract Modifiers
    # =========================================================================

    def _extract_modifiers(self, node: Node, code: bytes) -> list[str]:
        """Extract modifiers (public, static, async, etc.) from a declaration."""
        modifiers: list[str] = []
        for child in node.children:
            if child.type == "modifier":
                modifiers.append(self._get_text(child, code))
        return modifiers

    # =========================================================================
    # Helper: Extract Attributes
    # =========================================================================

    def _extract_attributes(self, node: Node, code: bytes) -> list[dict[str, str | None]]:
        """Extract attributes as [{"name": "TestMethod", "value": None}]."""
        attributes: list[dict[str, str | None]] = []
        for child in node.children:
            if child.type == "attribute_list":
                for attr_node in child.children:
                    if attr_node.type == "attribute":
                        parsed = self._parse_attribute(attr_node, code)
                        if parsed:
                            attributes.append(parsed)
        return attributes

    def _parse_attribute(self, node: Node, code: bytes) -> dict[str, str | None] | None:
        """Parse a single attribute node."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            for child in node.children:
                if child.type == "identifier":
                    name_node = child
                    break
        if not name_node:
            return None

        name = self._get_text(name_node, code)
        # Strip "Attribute" suffix
        if name.endswith("Attribute"):
            name = name[:-9]

        return {"name": name, "value": None}

    # =========================================================================
    # Helper: Extract Docstring (XML doc comments)
    # =========================================================================

    def _extract_docstring(self, node: Node, code: bytes) -> str | None:
        """Extract /// XML doc comments preceding a declaration."""
        comments: list[str] = []
        prev = node.prev_sibling

        while prev:
            if prev.type == "comment":
                text = self._get_text(prev, code).strip()
                if text.startswith("///"):
                    line = text[3:].strip()
                    comments.insert(0, line)
                else:
                    break
            else:
                break
            prev = prev.prev_sibling

        if not comments:
            return None

        # Clean up XML tags
        cleaned: list[str] = []
        for line in comments:
            line = line.replace("<summary>", "").replace("</summary>", "")
            line = line.replace("<param", "").replace("</param>", "")
            line = line.replace("<returns>", "Returns: ").replace("</returns>", "")
            line = line.replace("<remarks>", "").replace("</remarks>", "")
            line = line.strip()
            if line:
                cleaned.append(line)

        return "\n".join(cleaned) if cleaned else None

    # =========================================================================
    # Helper: Extract Return Type
    # =========================================================================

    def _extract_return_type(self, node: Node, code: bytes) -> str | None:
        """Extract return type from a method declaration."""
        # Try 'returns' field first (common in C# tree-sitter grammar)
        type_node = node.child_by_field_name("returns")
        if not type_node:
            type_node = node.child_by_field_name("type")
        if type_node:
            return self._extract_type_name(type_node, code)
        return None

    # =========================================================================
    # Helper: Extract Type Name
    # =========================================================================

    def _extract_type_name(self, node: Node, code: bytes) -> str | None:
        """Extract a type name, handling generics, nullable, arrays, predefined."""
        if node is None:
            return None

        if node.type in ("identifier", "qualified_name", "predefined_type"):
            return self._get_text(node, code)
        elif node.type == "generic_name":
            base_name = None
            type_args: list[str] = []
            for child in node.children:
                if child.type == "identifier":
                    base_name = self._get_text(child, code)
                elif child.type == "type_argument_list":
                    for arg in child.children:
                        if arg.type not in (",", "<", ">"):
                            arg_type = self._extract_type_name(arg, code)
                            if arg_type:
                                type_args.append(arg_type)
            if base_name:
                if type_args:
                    return f"{base_name}<{', '.join(type_args)}>"
                return base_name
        elif node.type == "nullable_type":
            for child in node.children:
                if child.type != "?":
                    inner = self._extract_type_name(child, code)
                    if inner:
                        return f"{inner}?"
        elif node.type == "array_type":
            for child in node.children:
                if child.type != "array_rank_specifier":
                    inner = self._extract_type_name(child, code)
                    if inner:
                        return f"{inner}[]"
        elif node.type == "tuple_type":
            return self._get_text(node, code)

        # Fallback
        return self._get_text(node, code)

    # =========================================================================
    # Helper: Extract Parameters
    # =========================================================================

    def _extract_parameters(self, node: Node, code: bytes) -> list[dict[str, str]]:
        """Extract method/constructor parameters."""
        params: list[dict[str, str]] = []
        params_node = node.child_by_field_name("parameters")
        if not params_node:
            return params

        for child in params_node.children:
            if child.type == "parameter":
                type_node = child.child_by_field_name("type")
                name_node = child.child_by_field_name("name")
                if type_node and name_node:
                    param_type = self._extract_type_name(type_node, code) or ""
                    param_name = self._get_text(name_node, code)
                    params.append({"type": param_type, "name": param_name})

        return params

    # =========================================================================
    # Helper: Node Utilities
    # =========================================================================

    def _get_text(self, node: Node, code: bytes) -> str:
        """Get text content of a node."""
        return code[node.start_byte:node.end_byte].decode("utf-8")

    def _find_child_by_type(self, node: Node, type_name: str) -> Node | None:
        """Find first direct child of a given type."""
        for child in node.children:
            if child.type == type_name:
                return child
        return None
