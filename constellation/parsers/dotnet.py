"""C#/.NET parser using Tree-sitter."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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

COMPILE_TIME_OPERATORS = frozenset({"nameof", "typeof", "sizeof", "default"})

CS_LANGUAGE = Language(tscsharp.language())


# =============================================================================
# Parsing Context
# =============================================================================


@dataclass
class _ParsingContext:
    """Mutable state threaded through the recursive walk."""

    file_path: str
    repository: str
    code: bytes
    namespace: str = ""
    current_class: str = ""
    current_class_full_id: str = ""

    # Using-directive tracking (populated by _collect_usings)
    usings: list[str] = field(default_factory=list)
    using_statics: list[str] = field(default_factory=list)
    using_aliases: dict[str, str] = field(default_factory=dict)

    # Entity ID maps (populated by later tasks)
    module_class_ids: dict[str, str] = field(default_factory=dict)
    class_method_ids: dict[str, dict[str, str]] = field(default_factory=dict)
    class_static_method_ids: dict[str, dict[str, str]] = field(default_factory=dict)


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

        # Log syntax errors but continue — tree-sitter is error-tolerant
        # and can still produce a usable partial AST.
        if tree.root_node.has_error:
            logger.warning("Syntax errors detected in %s (continuing with partial AST)", file_path)

        ctx = _ParsingContext(
            file_path=str(file_path),
            repository=repository,
            code=code,
        )

        # Collect using directives before processing entities
        self._collect_usings(tree.root_node, ctx)

        # Pre-collect class/method IDs for call resolution
        self._pre_collect_classes(tree.root_node, ctx)

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
        self._process_root(tree.root_node, ctx, file_entity, result)

        return result

    # =========================================================================
    # Using-Directive Collection
    # =========================================================================

    def _collect_usings(self, root: Node, ctx: _ParsingContext) -> None:
        """Scan the AST for using_directive nodes and populate *ctx*.

        Handles all C# using forms:
        - ``using System;``                -> ctx.usings
        - ``using static System.Math;``    -> ctx.using_statics
        - ``using MyAlias = Some.Type;``   -> ctx.using_aliases
        - ``global using System.Linq;``    -> ctx.usings  (same bucket)
        - ``global using static System.Console;`` -> ctx.using_statics
        """
        self._collect_usings_from(root, ctx)

    def _collect_usings_from(self, node: Node, ctx: _ParsingContext) -> None:
        """Recursively find using_directive nodes (they can appear inside
        namespace declaration_list blocks too)."""
        for child in node.children:
            if child.type == "using_directive":
                self._process_using_directive(child, ctx)
            elif child.type in ("namespace_declaration", "declaration_list"):
                self._collect_usings_from(child, ctx)

    def _process_using_directive(self, node: Node, ctx: _ParsingContext) -> None:
        """Classify a single using_directive and record it in *ctx*."""
        # Detect alias: tree-sitter C# exposes the alias name via the "name" field
        alias_node = node.child_by_field_name("name")
        if alias_node is not None:
            alias_name = self._get_text(alias_node, ctx.code)
            # The target is the qualified/generic name after the '=' token
            target = self._using_target_text(node, ctx.code, skip_alias=True)
            if target:
                ctx.using_aliases[alias_name] = target
            return

        # Detect 'static' modifier
        has_static = any(child.type == "static" for child in node.children)

        target = self._using_target_text(node, ctx.code, skip_alias=False)
        if not target:
            return

        if has_static:
            ctx.using_statics.append(target)
        else:
            ctx.usings.append(target)

    @staticmethod
    def _using_target_text(node: Node, code: bytes, *, skip_alias: bool) -> str | None:
        """Extract the namespace/type target from a using_directive node.

        When *skip_alias* is True, skip past the alias identifier and ``=``
        token to find the real target.
        """
        # The target is the first qualified_name, identifier, or generic_name
        # child that is NOT the alias identifier (which has the "name" field).
        for child in node.children:
            if child.type in ("qualified_name", "identifier", "generic_name"):
                if skip_alias and node.child_by_field_name("name") == child:
                    continue
                return code[child.start_byte:child.end_byte].decode("utf-8")
        return None

    # =========================================================================
    # Class/Method Pre-Collection (for call resolution)
    # =========================================================================

    def _pre_collect_classes(self, root: Node, ctx: _ParsingContext) -> None:
        """Pre-collect all class and method entity IDs before the main walk.

        Populates ``ctx.module_class_ids``, ``ctx.class_method_ids``, and
        ``ctx.class_static_method_ids`` so that the call-resolution pass
        (Tasks 3-4) can resolve targets without a second full traversal.
        """
        self._pre_collect_from(root, ctx, namespace="", outer_class_qname="")

    def _pre_collect_from(
        self,
        node: Node,
        ctx: _ParsingContext,
        namespace: str,
        outer_class_qname: str,
    ) -> None:
        """Recursively scan *node* for class declarations and register their
        entity IDs together with the IDs of their method members."""
        for child in node.children:
            if child.type in ("namespace_declaration", "file_scoped_namespace_declaration"):
                ns_name = self._get_namespace_name(child, ctx.code)
                if not ns_name:
                    continue
                # Block-scoped namespaces can be nested
                full_ns = f"{namespace}.{ns_name}" if namespace else ns_name
                # Recurse into the namespace body (declaration_list or direct children)
                decl_list = self._find_child_by_type(child, "declaration_list")
                if decl_list:
                    self._pre_collect_from(decl_list, ctx, namespace=full_ns, outer_class_qname="")
                else:
                    # file_scoped_namespace_declaration: types are direct children
                    self._pre_collect_from(child, ctx, namespace=full_ns, outer_class_qname="")

            elif child.type == "class_declaration":
                self._pre_collect_class(child, ctx, namespace, outer_class_qname)

    def _pre_collect_class(
        self,
        node: Node,
        ctx: _ParsingContext,
        namespace: str,
        outer_class_qname: str,
    ) -> None:
        """Register a single class and its methods in the pre-collection maps."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        class_name = self._get_text(name_node, ctx.code)

        # Build the qualified name the same way _process_class / _qualified_type_name does:
        # nested: "{outer_qname}.{class_name}", top-level: "{namespace}.{class_name}"
        if outer_class_qname:
            full_qname = f"{outer_class_qname}.{class_name}"
        elif namespace:
            full_qname = f"{namespace}.{class_name}"
        else:
            full_qname = class_name

        class_id = f"{ctx.repository}::{full_qname}"

        # Store class name -> class entity ID
        ctx.module_class_ids[class_name] = class_id

        # Walk the class body for method_declaration children
        body = self._find_child_by_type(node, "declaration_list")
        if body:
            method_ids: dict[str, str] = {}
            static_method_ids: dict[str, str] = {}

            for child in body.children:
                if child.type == "method_declaration":
                    mname_node = child.child_by_field_name("name")
                    if not mname_node:
                        continue
                    method_name = self._get_text(mname_node, ctx.code)
                    method_id = f"{ctx.repository}::{full_qname}.{method_name}"

                    method_ids[method_name] = method_id

                    # Check for static modifier
                    modifiers = self._extract_modifiers(child, ctx.code)
                    if "static" in modifiers:
                        static_method_ids[method_name] = method_id

            if method_ids:
                ctx.class_method_ids[class_id] = method_ids
            if static_method_ids:
                ctx.class_static_method_ids[class_id] = static_method_ids

            # Recurse into nested classes
            for child in body.children:
                if child.type == "class_declaration":
                    self._pre_collect_class(child, ctx, namespace, outer_class_qname=full_qname)

    # =========================================================================
    # Root Processing
    # =========================================================================

    def _process_root(
        self,
        root: Node,
        ctx: _ParsingContext,
        file_entity: CodeEntity,
        result: ParseResult,
    ) -> None:
        """Process the root of the AST to find namespaces and types."""
        # Process block-scoped namespace declarations
        for child in root.children:
            if child.type == "namespace_declaration":
                self._process_namespace(child, ctx, file_entity, result, parent_namespace="")
            elif child.type == "file_scoped_namespace_declaration":
                self._process_file_scoped_namespace(child, ctx, file_entity, result)

        # Process top-level types (no namespace)
        has_ns = any(
            c.type in ("namespace_declaration", "file_scoped_namespace_declaration")
            for c in root.children
        )
        if not has_ns:
            for child in root.children:
                self._process_type_node(child, ctx, file_entity, result, namespace="")

    # =========================================================================
    # Namespace Processing
    # =========================================================================

    def _process_namespace(
        self,
        node: Node,
        ctx: _ParsingContext,
        file_entity: CodeEntity,
        result: ParseResult,
        parent_namespace: str,
    ) -> None:
        """Process a block-scoped namespace declaration."""
        ns_name = self._get_namespace_name(node, ctx.code)
        if not ns_name:
            return

        full_ns = f"{parent_namespace}.{ns_name}" if parent_namespace else ns_name

        # Create PACKAGE entity for the namespace
        ns_entity = CodeEntity(
            id=f"{ctx.repository}::{full_ns}",
            name=full_ns.split(".")[-1],
            entity_type=EntityType.PACKAGE,
            repository=ctx.repository,
            file_path=ctx.file_path,
            line_number=node.start_point[0] + 1,
            language=self.language,
        )
        result.add_entity(ns_entity)

        # Process declarations inside the namespace body
        decl_list = self._find_child_by_type(node, "declaration_list")
        if decl_list:
            for child in decl_list.children:
                if child.type == "namespace_declaration":
                    self._process_namespace(child, ctx, file_entity, result, parent_namespace=full_ns)
                else:
                    self._process_type_node(child, ctx, file_entity, result, namespace=full_ns)

    def _process_file_scoped_namespace(
        self,
        node: Node,
        ctx: _ParsingContext,
        file_entity: CodeEntity,
        result: ParseResult,
    ) -> None:
        """Process a file-scoped namespace declaration (C# 10+)."""
        ns_name = self._get_namespace_name(node, ctx.code)
        if not ns_name:
            return

        # Create PACKAGE entity
        ns_entity = CodeEntity(
            id=f"{ctx.repository}::{ns_name}",
            name=ns_name.split(".")[-1],
            entity_type=EntityType.PACKAGE,
            repository=ctx.repository,
            file_path=ctx.file_path,
            line_number=node.start_point[0] + 1,
            language=self.language,
        )
        result.add_entity(ns_entity)

        # File-scoped: all types declared after the namespace directive are in this namespace
        # They appear as children of the file_scoped_namespace_declaration node
        for child in node.children:
            self._process_type_node(child, ctx, file_entity, result, namespace=ns_name)

    def _get_namespace_name(self, node: Node, code: bytes) -> str | None:
        """Extract namespace name from a namespace declaration node."""
        for child in node.children:
            if child.type in ("qualified_name", "identifier"):
                return self._get_text(child, code)
        return None

    @staticmethod
    def _qualified_type_name(
        namespace: str,
        type_name: str,
        *,
        outer_class_entity: CodeEntity | None = None,
    ) -> str:
        """Build a stable type name for top-level and nested declarations."""
        if outer_class_entity is not None and "::" in outer_class_entity.id:
            outer_full_name = outer_class_entity.id.split("::", 1)[1]
            return f"{outer_full_name}.{type_name}"
        return f"{namespace}.{type_name}" if namespace else type_name

    # =========================================================================
    # Type Routing
    # =========================================================================

    def _process_type_node(
        self,
        node: Node,
        ctx: _ParsingContext,
        file_entity: CodeEntity,
        result: ParseResult,
        namespace: str,
        outer_class_entity: CodeEntity | None = None,
    ) -> None:
        """Route a type declaration node to the appropriate handler."""
        if node.type == "class_declaration":
            self._process_class(node, ctx, file_entity, result, namespace, outer_class_entity)
        elif node.type == "interface_declaration":
            self._process_interface(node, ctx, file_entity, result, namespace, outer_class_entity)
        elif node.type == "enum_declaration":
            self._process_enum(node, ctx, file_entity, result, namespace)

    # =========================================================================
    # Class Processing
    # =========================================================================

    def _process_class(
        self,
        node: Node,
        ctx: _ParsingContext,
        file_entity: CodeEntity,
        result: ParseResult,
        namespace: str,
        outer_class_entity: CodeEntity | None = None,
    ) -> None:
        """Process a class declaration."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        class_name = self._get_text(name_node, ctx.code)
        full_name = self._qualified_type_name(
            namespace,
            class_name,
            outer_class_entity=outer_class_entity,
        )

        modifiers = self._extract_modifiers(node, ctx.code)
        docstring = self._extract_docstring(node, ctx.code)
        class_code = self._get_text(node, ctx.code)

        class_entity = CodeEntity(
            id=f"{ctx.repository}::{full_name}",
            name=class_name,
            entity_type=EntityType.CLASS,
            repository=ctx.repository,
            file_path=ctx.file_path,
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
                target_id=f"{ctx.repository}::{namespace}",
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
        self._extract_base_types(node, ctx.code, class_entity, result, is_interface=False)

        # Process class body — set current_class context for call resolution
        body_node = self._find_child_by_type(node, "declaration_list")
        if body_node:
            saved_class = ctx.current_class
            saved_class_full_id = ctx.current_class_full_id
            ctx.current_class = class_name
            ctx.current_class_full_id = class_entity.id
            self._process_class_body(body_node, ctx, file_entity, class_entity, result, namespace)
            ctx.current_class = saved_class
            ctx.current_class_full_id = saved_class_full_id

    # =========================================================================
    # Interface Processing
    # =========================================================================

    def _process_interface(
        self,
        node: Node,
        ctx: _ParsingContext,
        file_entity: CodeEntity,
        result: ParseResult,
        namespace: str,
        outer_class_entity: CodeEntity | None = None,
    ) -> None:
        """Process an interface declaration."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        iface_name = self._get_text(name_node, ctx.code)
        full_name = self._qualified_type_name(
            namespace,
            iface_name,
            outer_class_entity=outer_class_entity,
        )

        modifiers = self._extract_modifiers(node, ctx.code)
        docstring = self._extract_docstring(node, ctx.code)

        iface_entity = CodeEntity(
            id=f"{ctx.repository}::{full_name}",
            name=iface_name,
            entity_type=EntityType.INTERFACE,
            repository=ctx.repository,
            file_path=ctx.file_path,
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
                target_id=f"{ctx.repository}::{namespace}",
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
                    self._process_method(child, ctx, iface_entity, result, namespace)

    # =========================================================================
    # Enum Processing
    # =========================================================================

    def _process_enum(
        self,
        node: Node,
        ctx: _ParsingContext,
        file_entity: CodeEntity,
        result: ParseResult,
        namespace: str,
        outer_class_entity: CodeEntity | None = None,
    ) -> None:
        """Process an enum declaration as a CLASS entity with 'enum' stereotype."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        enum_name = self._get_text(name_node, ctx.code)
        full_name = self._qualified_type_name(
            namespace,
            enum_name,
            outer_class_entity=outer_class_entity,
        )

        modifiers = self._extract_modifiers(node, ctx.code)
        docstring = self._extract_docstring(node, ctx.code)

        enum_entity = CodeEntity(
            id=f"{ctx.repository}::{full_name}",
            name=enum_name,
            entity_type=EntityType.CLASS,
            repository=ctx.repository,
            file_path=ctx.file_path,
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
                target_id=f"{ctx.repository}::{namespace}",
                relationship_type=RelationshipType.IN_PACKAGE,
            ))

        if outer_class_entity is not None:
            result.add_relationship(CodeRelationship(
                source_id=outer_class_entity.id,
                target_id=enum_entity.id,
                relationship_type=RelationshipType.DECLARES,
            ))

    # =========================================================================
    # Class Body Processing
    # =========================================================================

    def _process_class_body(
        self,
        body: Node,
        ctx: _ParsingContext,
        file_entity: CodeEntity,
        class_entity: CodeEntity,
        result: ParseResult,
        namespace: str,
    ) -> None:
        """Process members inside a class body."""
        for child in body.children:
            if child.type == "method_declaration":
                self._process_method(child, ctx, class_entity, result, namespace)
            elif child.type == "constructor_declaration":
                self._process_constructor(child, ctx, class_entity, result, namespace)
            elif child.type == "field_declaration":
                self._process_field(child, ctx, class_entity, result, namespace)
            elif child.type == "property_declaration":
                self._process_property(child, ctx, class_entity, result, namespace)
            elif child.type == "class_declaration":
                # Nested class
                self._process_class(child, ctx, file_entity, result, namespace, outer_class_entity=class_entity)
            elif child.type == "interface_declaration":
                # Nested interface
                self._process_interface(child, ctx, file_entity, result, namespace, outer_class_entity=class_entity)
            elif child.type == "enum_declaration":
                # Nested enum
                self._process_enum(
                    child,
                    ctx,
                    file_entity,
                    result,
                    namespace,
                    outer_class_entity=class_entity,
                )

    # =========================================================================
    # Method Processing
    # =========================================================================

    def _process_method(
        self,
        node: Node,
        ctx: _ParsingContext,
        class_entity: CodeEntity,
        result: ParseResult,
        namespace: str,
    ) -> None:
        """Process a method declaration."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        method_name = self._get_text(name_node, ctx.code)
        class_full_name = class_entity.id.split("::", 1)[1] if "::" in class_entity.id else class_entity.name
        full_name = f"{class_full_name}.{method_name}"

        # Return type
        return_type = self._extract_return_type(node, ctx.code)

        # Signature
        params = self._extract_parameters(node, ctx.code)
        param_str = ", ".join(f"{p['type']} {p['name']}" for p in params)
        signature = f"{return_type or 'void'} {method_name}({param_str})"

        modifiers = self._extract_modifiers(node, ctx.code)
        docstring = self._extract_docstring(node, ctx.code)
        method_code = self._get_text(node, ctx.code)

        # Detect test stereotype via attributes
        attributes = self._extract_attributes(node, ctx.code)
        stereotypes: list[str] = []
        attr_names = {a["name"] for a in attributes}
        if attr_names & TEST_METHOD_ATTRIBUTES:
            stereotypes.append("test")

        method_entity = CodeEntity(
            id=f"{ctx.repository}::{full_name}",
            name=method_name,
            entity_type=EntityType.METHOD,
            repository=ctx.repository,
            file_path=ctx.file_path,
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

        # Extract calls from method body
        body_node = node.child_by_field_name("body")
        if body_node:
            self._extract_calls(body_node, ctx, result, method_entity.id)

    # =========================================================================
    # Constructor Processing
    # =========================================================================

    def _process_constructor(
        self,
        node: Node,
        ctx: _ParsingContext,
        class_entity: CodeEntity,
        result: ParseResult,
        namespace: str,
    ) -> None:
        """Process a constructor declaration."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        ctor_name = self._get_text(name_node, ctx.code)
        class_full_name = class_entity.id.split("::", 1)[1] if "::" in class_entity.id else class_entity.name
        full_name = f"{class_full_name}.{ctor_name}"

        params = self._extract_parameters(node, ctx.code)
        param_str = ", ".join(f"{p['type']} {p['name']}" for p in params)
        signature = f"{ctor_name}({param_str})"

        modifiers = self._extract_modifiers(node, ctx.code)
        docstring = self._extract_docstring(node, ctx.code)
        ctor_code = self._get_text(node, ctx.code)

        ctor_entity = CodeEntity(
            id=f"{ctx.repository}::{full_name}",
            name=ctor_name,
            entity_type=EntityType.CONSTRUCTOR,
            repository=ctx.repository,
            file_path=ctx.file_path,
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

        # Extract calls from constructor body
        body_node = node.child_by_field_name("body")
        if body_node:
            self._extract_calls(body_node, ctx, result, ctor_entity.id)

    # =========================================================================
    # Field Processing
    # =========================================================================

    def _process_field(
        self,
        node: Node,
        ctx: _ParsingContext,
        class_entity: CodeEntity,
        result: ParseResult,
        namespace: str,
    ) -> None:
        """Process a field declaration (may declare multiple variables)."""
        modifiers = self._extract_modifiers(node, ctx.code)

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
            field_type = self._extract_type_name(type_node, ctx.code)

        # Process each variable declarator
        for child in var_decl.children:
            if child.type == "variable_declarator":
                name_node = child.child_by_field_name("name")
                if not name_node:
                    # Fallback: find identifier child
                    name_node = self._find_child_by_type(child, "identifier")
                if not name_node:
                    continue

                field_name = self._get_text(name_node, ctx.code)
                class_full_name = class_entity.id.split("::", 1)[1] if "::" in class_entity.id else class_entity.name
                full_name = f"{class_full_name}.{field_name}"

                field_entity = CodeEntity(
                    id=f"{ctx.repository}::{full_name}",
                    name=field_name,
                    entity_type=EntityType.FIELD,
                    repository=ctx.repository,
                    file_path=ctx.file_path,
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
        ctx: _ParsingContext,
        class_entity: CodeEntity,
        result: ParseResult,
        namespace: str,
    ) -> None:
        """Process a property declaration as a FIELD entity."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        prop_name = self._get_text(name_node, ctx.code)
        class_full_name = class_entity.id.split("::", 1)[1] if "::" in class_entity.id else class_entity.name
        full_name = f"{class_full_name}.{prop_name}"

        modifiers = self._extract_modifiers(node, ctx.code)
        modifiers.append("property")

        prop_type = None
        type_node = node.child_by_field_name("type")
        if type_node:
            prop_type = self._extract_type_name(type_node, ctx.code)

        prop_entity = CodeEntity(
            id=f"{ctx.repository}::{full_name}",
            name=prop_name,
            entity_type=EntityType.FIELD,
            repository=ctx.repository,
            file_path=ctx.file_path,
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
    # Call Extraction
    # =========================================================================

    def _extract_calls(
        self,
        body: Node,
        ctx: _ParsingContext,
        result: ParseResult,
        source_id: str,
    ) -> None:
        """Recursively walk *body* to find invocation_expression nodes
        and record each as a call from *source_id*.

        Lambda bodies and anonymous delegate bodies are walked and calls
        within them are attributed to the enclosing method/constructor.
        """
        seen_reference_targets: set[str] = set()
        self._walk_for_calls(body, ctx, result, source_id, seen_reference_targets)

    def _walk_for_calls(
        self,
        node: Node,
        ctx: _ParsingContext,
        result: ParseResult,
        source_id: str,
        seen_reference_targets: set[str],
    ) -> None:
        """Recursive walk helper for call extraction."""
        if node.type == "invocation_expression":
            self._record_call(node, ctx, result, source_id, seen_reference_targets)

        # Recurse into all children (including lambdas and anonymous delegates)
        for child in node.children:
            self._walk_for_calls(child, ctx, result, source_id, seen_reference_targets)

    def _record_call(
        self,
        call_node: Node,
        ctx: _ParsingContext,
        result: ParseResult,
        source_id: str,
        seen_reference_targets: set[str],
    ) -> None:
        """Resolve and record a single invocation_expression."""
        # The function part is the first child of invocation_expression
        # (the second child is argument_list)
        func_node = call_node.children[0] if call_node.children else None
        if func_node is None:
            return

        called_symbol: str | None = None
        target_id: str | None = None
        receiver_text: str | None = None

        if func_node.type == "identifier":
            called_symbol = self._get_text(func_node, ctx.code)

            # Skip compile-time operators
            if called_symbol in COMPILE_TIME_OPERATORS:
                return

            # Tier 1: Same-class method resolution
            class_methods = ctx.class_method_ids.get(ctx.current_class_full_id, {})
            target_id = class_methods.get(called_symbol)

        elif func_node.type == "generic_name":
            # Generic method call like Method<T>() — extract base name
            name_node = self._find_child_by_type(func_node, "identifier")
            if name_node:
                called_symbol = self._get_text(name_node, ctx.code)

                # Skip compile-time operators
                if called_symbol in COMPILE_TIME_OPERATORS:
                    return

                # Tier 1: Same-class method resolution
                class_methods = ctx.class_method_ids.get(ctx.current_class_full_id, {})
                target_id = class_methods.get(called_symbol)

        elif func_node.type == "member_access_expression":
            # object.Method() — check for this/base
            object_node = func_node.children[0] if func_node.children else None
            member_node = None
            for child in func_node.children:
                if child.type == "identifier" and child != object_node:
                    member_node = child
            # The member name is the rightmost identifier (after the '.')
            if member_node is None:
                # Last child that is an identifier
                for child in reversed(func_node.children):
                    if child.type == "identifier":
                        member_node = child
                        break

            if member_node is None or object_node is None:
                return

            called_symbol = self._get_text(member_node, ctx.code)
            receiver_text = self._get_text(object_node, ctx.code)

            if object_node.type == "this":
                # Tier 2: this.Method() — resolve via same-class
                class_methods = ctx.class_method_ids.get(ctx.current_class_full_id, {})
                target_id = class_methods.get(called_symbol)
            elif object_node.type == "base":
                # Tier 2b: base.Method() — stays unresolved
                target_id = None
            else:
                # Tier 7: Other member access — unresolved
                # Use the full expression text as the called symbol
                called_symbol = self._get_text(func_node, ctx.code)
                target_id = None

        else:
            # Fallback: use the full text of the function node
            called_symbol = self._get_text(func_node, ctx.code).strip()
            if called_symbol in COMPILE_TIME_OPERATORS:
                return

        if not called_symbol:
            return

        # If we resolved to a target, emit a CALLS edge directly
        if target_id is not None:
            result.add_relationship(CodeRelationship(
                source_id=source_id,
                target_id=target_id,
                relationship_type=RelationshipType.CALLS,
            ))
            return

        # Otherwise: unresolved — create a Reference entity
        target_id = self._reference_target_id(ctx, source_id, called_symbol, call_node)
        if target_id not in seen_reference_targets:
            seen_reference_targets.add(target_id)
            result.add_entity(CodeEntity(
                id=target_id,
                name=called_symbol,
                entity_type=EntityType.REFERENCE,
                repository=ctx.repository,
                file_path=ctx.file_path,
                line_number=call_node.start_point[0] + 1,
                line_end=call_node.end_point[0] + 1,
                language=self.language,
                properties=self._reference_properties(called_symbol, source_id, receiver_text),
            ))
        result.add_relationship(CodeRelationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type=RelationshipType.CALLS,
        ))

    def _reference_target_id(
        self,
        ctx: _ParsingContext,
        source_id: str,
        called_symbol: str,
        call_node: Node,
    ) -> str:
        """Build a unique ID for an unresolved call-site reference."""
        line = call_node.start_point[0] + 1
        col = call_node.start_point[1] + 1
        return f"{source_id}::ref:{ctx.file_path}:{line}:{col}:{called_symbol}"

    def _reference_properties(
        self, called_symbol: str, source_id: str, receiver_text: str | None,
    ) -> dict:
        """Build the properties dict for an unresolved Reference entity."""
        props: dict = {
            "symbol": called_symbol,
            "enclosing_declaration_id": source_id,
            "enclosing_declaration_name": self._enclosing_declaration_name(source_id),
        }
        if receiver_text:
            props["receiver"] = receiver_text
        return props

    @staticmethod
    def _enclosing_declaration_name(source_id: str) -> str:
        """Extract the short name from a fully-qualified entity ID."""
        local_id = source_id.split("::", 1)[-1]
        return local_id.rsplit(".", 1)[-1]

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
