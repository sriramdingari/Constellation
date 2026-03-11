"""Java parser using Tree-sitter."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import tree_sitter_java as tsjava
from tree_sitter import Language, Parser, Node

from constellation.models import CodeEntity, CodeRelationship, EntityType, RelationshipType
from constellation.parsers.base import BaseParser, ParseResult

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

TEST_ANNOTATIONS = frozenset({
    "Test",
    "ParameterizedTest",
    "RepeatedTest",
    "TestFactory",
})

ENDPOINT_METHOD_ANNOTATIONS = {
    "RequestMapping": None,
    "GetMapping": "GET",
    "PostMapping": "POST",
    "PutMapping": "PUT",
    "DeleteMapping": "DELETE",
    "PatchMapping": "PATCH",
}

ENDPOINT_CLASS_ANNOTATIONS = frozenset({
    "RestController",
    "Controller",
})

SPRING_STEREOTYPE_MAP = {
    "Service": "service",
    "Repository": "repository",
    "Component": "component",
    "RestController": "rest_controller",
    "Controller": "controller",
    "Configuration": "configuration",
}

JAVA_LANGUAGE = Language(tsjava.language())


class JavaParser(BaseParser):
    """Parser for Java source files using Tree-sitter."""

    def __init__(self) -> None:
        self._parser = Parser(JAVA_LANGUAGE)

    @property
    def language(self) -> str:
        return "java"

    @property
    def file_extensions(self) -> list[str]:
        return [".java"]

    def parse_file(self, file_path: Path, repository: str) -> ParseResult:
        """Parse a Java source file and return structured results."""
        result = ParseResult(file_path=str(file_path), language=self.language)

        try:
            code = file_path.read_bytes()
        except Exception as e:
            result.add_error(f"Failed to read file: {e}")
            return result

        tree = self._parser.parse(code)

        # File entity
        file_id = f"{repository}::{file_path}"
        file_entity = CodeEntity(
            id=file_id,
            name=file_path.name,
            entity_type=EntityType.FILE,
            repository=repository,
            file_path=str(file_path),
            line_number=1,
            language=self.language,
        )
        result.add_entity(file_entity)

        ctx = _ParsingContext(
            file_path=str(file_path),
            file_id=file_id,
            code=code,
            repository=repository,
        )

        # Extract package
        self._extract_package(tree.root_node, ctx, result)

        # Extract top-level type declarations
        self._extract_types(tree.root_node, ctx, result)

        return result

    # ------------------------------------------------------------------
    # Package
    # ------------------------------------------------------------------

    def _extract_package(
        self,
        root: Node,
        ctx: _ParsingContext,
        result: ParseResult,
    ) -> None:
        for pkg_node in self._find_nodes(root, "package_declaration"):
            for child in pkg_node.children:
                if child.type in ("scoped_identifier", "identifier"):
                    pkg_name = self._text(child, ctx.code)
                    ctx.package_name = pkg_name
                    pkg_entity = CodeEntity(
                        id=f"{ctx.repository}::{pkg_name}",
                        name=pkg_name,
                        entity_type=EntityType.PACKAGE,
                        repository=ctx.repository,
                        file_path=ctx.file_path,
                        line_number=pkg_node.start_point[0] + 1,
                        language=self.language,
                    )
                    result.add_entity(pkg_entity)
                    return

    # ------------------------------------------------------------------
    # Top-level types
    # ------------------------------------------------------------------

    def _extract_types(
        self,
        root: Node,
        ctx: _ParsingContext,
        result: ParseResult,
    ) -> None:
        for child in root.children:
            if child.type == "class_declaration":
                self._process_class(child, ctx, result, is_interface=False)
            elif child.type == "interface_declaration":
                self._process_class(child, ctx, result, is_interface=True)
            elif child.type == "enum_declaration":
                self._process_enum(child, ctx, result)

    # ------------------------------------------------------------------
    # Class / Interface
    # ------------------------------------------------------------------

    def _process_class(
        self,
        node: Node,
        ctx: _ParsingContext,
        result: ParseResult,
        *,
        is_interface: bool = False,
        outer_class_id: str | None = None,
    ) -> None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        class_name = self._text(name_node, ctx.code)
        if outer_class_id and ctx.current_class_full_name:
            full_name = f"{ctx.current_class_full_name}.{class_name}"
        else:
            full_name = f"{ctx.package_name}.{class_name}" if ctx.package_name else class_name
        entity_id = f"{ctx.repository}::{full_name}"

        modifiers = self._extract_modifiers(node, ctx.code)
        annotations = self._extract_annotations(node, ctx.code)
        annotation_names = [a["name"] for a in annotations]
        docstring = self._extract_docstring(node, ctx.code)

        stereotypes: list[str] = []
        for anno_name, stereo in SPRING_STEREOTYPE_MAP.items():
            if anno_name in annotation_names:
                stereotypes.append(stereo)

        etype = EntityType.INTERFACE if is_interface else EntityType.CLASS

        class_entity = CodeEntity(
            id=entity_id,
            name=class_name,
            entity_type=etype,
            repository=ctx.repository,
            file_path=ctx.file_path,
            line_number=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            language=self.language,
            modifiers=modifiers,
            stereotypes=stereotypes,
            docstring=docstring,
        )
        result.add_entity(class_entity)

        # CONTAINS: file -> class
        result.add_relationship(CodeRelationship(
            source_id=ctx.file_id,
            target_id=entity_id,
            relationship_type=RelationshipType.CONTAINS,
        ))

        # IN_PACKAGE
        if ctx.package_name:
            result.add_relationship(CodeRelationship(
                source_id=entity_id,
                target_id=f"{ctx.repository}::{ctx.package_name}",
                relationship_type=RelationshipType.IN_PACKAGE,
            ))

        # DECLARES (inner class)
        if outer_class_id:
            result.add_relationship(CodeRelationship(
                source_id=outer_class_id,
                target_id=entity_id,
                relationship_type=RelationshipType.DECLARES,
            ))

        # EXTENDS
        if is_interface:
            # Interface extends: uses "extends_interfaces" field or child named "extends_interfaces"
            self._extract_interface_extends(node, ctx, entity_id, result)
        else:
            superclass = node.child_by_field_name("superclass")
            if superclass:
                for child in superclass.children:
                    if child.type in ("type_identifier", "generic_type"):
                        super_name = self._extract_type_name(child, ctx.code)
                        if super_name:
                            super_full = f"{ctx.package_name}.{super_name}" if ctx.package_name else super_name
                            result.add_relationship(CodeRelationship(
                                source_id=entity_id,
                                target_id=f"{ctx.repository}::{super_full}",
                                relationship_type=RelationshipType.EXTENDS,
                            ))
                        break

        # IMPLEMENTS
        if not is_interface:
            interfaces = node.child_by_field_name("interfaces")
            if interfaces:
                self._extract_type_list_relationships(
                    interfaces, ctx, entity_id, RelationshipType.IMPLEMENTS, result,
                )

        # Process body
        body_node = node.child_by_field_name("body")
        if body_node:
            exact_call_targets, call_targets_by_name = self._collect_class_call_targets(
                body_node,
                ctx,
                full_name,
            )
            class_ctx = ctx.with_class(
                class_name,
                full_name,
                annotations,
                exact_call_targets=exact_call_targets,
                call_targets_by_name=call_targets_by_name,
            )
            self._process_class_body(body_node, class_ctx, entity_id, result, is_interface=is_interface)
            # Nested classes
            self._process_nested_types(body_node, class_ctx, entity_id, result)

    # ------------------------------------------------------------------
    # Interface extends
    # ------------------------------------------------------------------

    def _extract_interface_extends(
        self,
        node: Node,
        ctx: _ParsingContext,
        interface_id: str,
        result: ParseResult,
    ) -> None:
        """Extract extends clause for interfaces (interface extends interface)."""
        extends_node = node.child_by_field_name("extends_interfaces")
        if extends_node:
            self._extract_type_list_relationships(
                extends_node, ctx, interface_id, RelationshipType.EXTENDS, result,
            )
            return
        # Fallback: scan children for extends_interfaces type_list
        for child in node.children:
            if child.type == "extends_interfaces":
                self._extract_type_list_relationships(
                    child, ctx, interface_id, RelationshipType.EXTENDS, result,
                )
                return

    def _extract_type_list_relationships(
        self,
        node: Node,
        ctx: _ParsingContext,
        source_id: str,
        rel_type: RelationshipType,
        result: ParseResult,
    ) -> None:
        """Extract type references from a type_list node and create relationships."""
        for child in node.children:
            if child.type == "type_list":
                for type_child in child.children:
                    if type_child.type in ("type_identifier", "generic_type"):
                        type_name = self._extract_type_name(type_child, ctx.code)
                        if type_name:
                            full = f"{ctx.package_name}.{type_name}" if ctx.package_name else type_name
                            result.add_relationship(CodeRelationship(
                                source_id=source_id,
                                target_id=f"{ctx.repository}::{full}",
                                relationship_type=rel_type,
                            ))
            elif child.type in ("type_identifier", "generic_type"):
                type_name = self._extract_type_name(child, ctx.code)
                if type_name:
                    full = f"{ctx.package_name}.{type_name}" if ctx.package_name else type_name
                    result.add_relationship(CodeRelationship(
                        source_id=source_id,
                        target_id=f"{ctx.repository}::{full}",
                        relationship_type=rel_type,
                    ))

    # ------------------------------------------------------------------
    # Enum
    # ------------------------------------------------------------------

    def _process_enum(
        self,
        node: Node,
        ctx: _ParsingContext,
        result: ParseResult,
        outer_class_id: str | None = None,
    ) -> None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        enum_name = self._text(name_node, ctx.code)
        if outer_class_id and ctx.current_class_full_name:
            full_name = f"{ctx.current_class_full_name}.{enum_name}"
        else:
            full_name = f"{ctx.package_name}.{enum_name}" if ctx.package_name else enum_name
        entity_id = f"{ctx.repository}::{full_name}"

        modifiers = self._extract_modifiers(node, ctx.code)
        docstring = self._extract_docstring(node, ctx.code)

        enum_entity = CodeEntity(
            id=entity_id,
            name=enum_name,
            entity_type=EntityType.CLASS,
            repository=ctx.repository,
            file_path=ctx.file_path,
            line_number=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            language=self.language,
            modifiers=modifiers,
            stereotypes=["enum"],
            docstring=docstring,
        )
        result.add_entity(enum_entity)

        # CONTAINS
        result.add_relationship(CodeRelationship(
            source_id=ctx.file_id,
            target_id=entity_id,
            relationship_type=RelationshipType.CONTAINS,
        ))

        # IN_PACKAGE
        if ctx.package_name:
            result.add_relationship(CodeRelationship(
                source_id=entity_id,
                target_id=f"{ctx.repository}::{ctx.package_name}",
                relationship_type=RelationshipType.IN_PACKAGE,
            ))

        # DECLARES
        if outer_class_id:
            result.add_relationship(CodeRelationship(
                source_id=outer_class_id,
                target_id=entity_id,
                relationship_type=RelationshipType.DECLARES,
            ))

        # Process enum body
        body_node = node.child_by_field_name("body")
        if body_node:
            exact_call_targets, call_targets_by_name = self._collect_class_call_targets(
                body_node,
                ctx,
                full_name,
            )
            enum_ctx = ctx.with_class(
                enum_name,
                full_name,
                [],
                exact_call_targets=exact_call_targets,
                call_targets_by_name=call_targets_by_name,
            )
            for child in body_node.children:
                if child.type == "enum_body_declarations":
                    for decl in child.children:
                        if decl.type == "method_declaration":
                            self._process_method(decl, enum_ctx, entity_id, result)
                        elif decl.type == "constructor_declaration":
                            self._process_constructor(decl, enum_ctx, entity_id, result)
                        elif decl.type == "field_declaration":
                            self._process_field(decl, enum_ctx, entity_id, result)

    # ------------------------------------------------------------------
    # Class body processing
    # ------------------------------------------------------------------

    def _process_class_body(
        self,
        body: Node,
        ctx: _ParsingContext,
        class_id: str,
        result: ParseResult,
        *,
        is_interface: bool = False,
    ) -> None:
        for child in body.children:
            if child.type == "method_declaration":
                self._process_method(child, ctx, class_id, result)
            elif child.type == "constructor_declaration":
                self._process_constructor(child, ctx, class_id, result)
            elif child.type == "field_declaration":
                self._process_field(child, ctx, class_id, result)

    def _process_nested_types(
        self,
        body: Node,
        ctx: _ParsingContext,
        outer_class_id: str,
        result: ParseResult,
    ) -> None:
        for child in body.children:
            if child.type == "class_declaration":
                self._process_class(
                    child, ctx, result,
                    is_interface=False,
                    outer_class_id=outer_class_id,
                )
            elif child.type == "interface_declaration":
                self._process_class(
                    child, ctx, result,
                    is_interface=True,
                    outer_class_id=outer_class_id,
                )
            elif child.type == "enum_declaration":
                self._process_enum(child, ctx, result, outer_class_id=outer_class_id)

    # ------------------------------------------------------------------
    # Method
    # ------------------------------------------------------------------

    def _process_method(
        self,
        node: Node,
        ctx: _ParsingContext,
        class_id: str,
        result: ParseResult,
    ) -> None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        method_name = self._text(name_node, ctx.code)
        parameters = self._extract_parameters(node, ctx.code)
        param_type_str = ",".join(p["type"] for p in parameters)
        full_name = f"{ctx.current_class_full_name}.{method_name}({param_type_str})"
        entity_id = f"{ctx.repository}::{full_name}"

        # Return type
        return_type = None
        type_node = node.child_by_field_name("type")
        if type_node:
            return_type = self._text(type_node, ctx.code)

        # Signature
        param_sig = ", ".join(f"{p['type']} {p['name']}" for p in parameters)
        signature = f"{return_type or 'void'} {method_name}({param_sig})"

        modifiers = self._extract_modifiers(node, ctx.code)
        docstring = self._extract_docstring(node, ctx.code)
        annotations = self._extract_annotations(node, ctx.code)
        annotation_names = [a["name"] for a in annotations]

        stereotypes: list[str] = []

        # Test stereotype
        if self._is_test_method(method_name, annotations):
            stereotypes.append("test")

        # Endpoint stereotype
        if self._is_endpoint_method(annotations, ctx.class_annotations):
            stereotypes.append("endpoint")

        method_entity = CodeEntity(
            id=entity_id,
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
            stereotypes=stereotypes,
            docstring=docstring,
        )
        result.add_entity(method_entity)

        # HAS_METHOD
        result.add_relationship(CodeRelationship(
            source_id=class_id,
            target_id=entity_id,
            relationship_type=RelationshipType.HAS_METHOD,
        ))

        # CALLS
        self._extract_method_calls(node, ctx, entity_id, result)

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def _process_constructor(
        self,
        node: Node,
        ctx: _ParsingContext,
        class_id: str,
        result: ParseResult,
    ) -> None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        ctor_name = self._text(name_node, ctx.code)
        parameters = self._extract_parameters(node, ctx.code)
        param_type_str = ",".join(p["type"] for p in parameters)
        full_name = f"{ctx.current_class_full_name}.{ctor_name}({param_type_str})"
        entity_id = f"{ctx.repository}::{full_name}"

        param_sig = ", ".join(f"{p['type']} {p['name']}" for p in parameters)
        signature = f"{ctor_name}({param_sig})"

        modifiers = self._extract_modifiers(node, ctx.code)
        docstring = self._extract_docstring(node, ctx.code)

        ctor_entity = CodeEntity(
            id=entity_id,
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
        )
        result.add_entity(ctor_entity)

        # HAS_CONSTRUCTOR
        result.add_relationship(CodeRelationship(
            source_id=class_id,
            target_id=entity_id,
            relationship_type=RelationshipType.HAS_CONSTRUCTOR,
        ))

    # ------------------------------------------------------------------
    # Field
    # ------------------------------------------------------------------

    def _process_field(
        self,
        node: Node,
        ctx: _ParsingContext,
        class_id: str,
        result: ParseResult,
    ) -> None:
        type_node = node.child_by_field_name("type")
        field_type = self._text(type_node, ctx.code) if type_node else None
        modifiers = self._extract_modifiers(node, ctx.code)

        for declarator in self._direct_children(node, "variable_declarator"):
            name_node = declarator.child_by_field_name("name")
            if not name_node:
                continue
            field_name = self._text(name_node, ctx.code)
            full_name = f"{ctx.current_class_full_name}.{field_name}"
            entity_id = f"{ctx.repository}::{full_name}"

            field_entity = CodeEntity(
                id=entity_id,
                name=field_name,
                entity_type=EntityType.FIELD,
                repository=ctx.repository,
                file_path=ctx.file_path,
                line_number=node.start_point[0] + 1,
                language=self.language,
                modifiers=modifiers,
                properties={"field_type": field_type} if field_type else {},
            )
            result.add_entity(field_entity)

            # HAS_FIELD
            result.add_relationship(CodeRelationship(
                source_id=class_id,
                target_id=entity_id,
                relationship_type=RelationshipType.HAS_FIELD,
            ))

    # ------------------------------------------------------------------
    # Method calls extraction
    # ------------------------------------------------------------------

    def _extract_method_calls(
        self,
        node: Node,
        ctx: _ParsingContext,
        source_method_id: str,
        result: ParseResult,
    ) -> None:
        processed: set[str] = set()
        for call_node in self._find_nodes(node, "method_invocation"):
            call_name_node = call_node.child_by_field_name("name")
            if not call_name_node:
                continue
            called_method = self._text(call_name_node, ctx.code)

            call_object_node = call_node.child_by_field_name("object")
            is_current_class_call = False
            if call_object_node:
                object_text = self._text(call_object_node, ctx.code)
                is_current_class_call = object_text in {
                    "this",
                    ctx.current_class,
                    ctx.current_class_full_name,
                }
                if is_current_class_call:
                    called_full = f"{ctx.current_class_full_name}.{called_method}"
                else:
                    called_full = f"{object_text}.{called_method}"
            else:
                called_full = f"{ctx.current_class_full_name}.{called_method}"
                is_current_class_call = True

            if called_full not in processed:
                processed.add(called_full)
                target_id = None
                if is_current_class_call:
                    target_id = self._resolve_current_class_call(
                        called_method,
                        call_node,
                        ctx,
                    )
                if target_id is None:
                    target_id = f"{source_method_id}::ref:{called_full}"
                    result.add_entity(CodeEntity(
                        id=target_id,
                        name=called_full,
                        entity_type=EntityType.REFERENCE,
                        repository=ctx.repository,
                        file_path=ctx.file_path,
                        line_number=call_node.start_point[0] + 1,
                        language=self.language,
                        properties={"symbol": called_full},
                    ))
                result.add_relationship(CodeRelationship(
                    source_id=source_method_id,
                    target_id=target_id,
                    relationship_type=RelationshipType.CALLS,
                ))

    def _collect_class_call_targets(
        self,
        body_node: Node,
        ctx: _ParsingContext,
        class_full_name: str,
    ) -> tuple[dict[tuple[str, int], list[str]], dict[str, list[str]]]:
        """Collect current-class method targets so local calls can resolve to Methods."""
        exact_targets: dict[tuple[str, int], list[str]] = {}
        name_targets: dict[str, list[str]] = {}

        method_nodes: list[Node] = []
        for child in body_node.children:
            if child.type == "method_declaration":
                method_nodes.append(child)
            elif child.type == "enum_body_declarations":
                for decl in child.children:
                    if decl.type == "method_declaration":
                        method_nodes.append(decl)

        for method_node in method_nodes:
            name_node = method_node.child_by_field_name("name")
            if not name_node:
                continue
            method_name = self._text(name_node, ctx.code)
            parameters = self._extract_parameters(method_node, ctx.code)
            param_type_str = ",".join(p["type"] for p in parameters)
            full_name = f"{class_full_name}.{method_name}({param_type_str})"
            target_id = f"{ctx.repository}::{full_name}"
            exact_targets.setdefault((method_name, len(parameters)), []).append(target_id)
            name_targets.setdefault(method_name, []).append(target_id)

        return exact_targets, name_targets

    def _resolve_current_class_call(
        self,
        called_method: str,
        call_node: Node,
        ctx: _ParsingContext,
    ) -> str | None:
        """Resolve a call to a current-class method when the target is unambiguous."""
        exact_matches = ctx.exact_call_targets.get(
            (called_method, self._argument_count(call_node)),
            [],
        )
        if len(exact_matches) == 1:
            return exact_matches[0]

        name_matches = ctx.call_targets_by_name.get(called_method, [])
        if len(name_matches) == 1:
            return name_matches[0]

        return None

    def _argument_count(self, call_node: Node) -> int:
        arguments_node = call_node.child_by_field_name("arguments")
        if not arguments_node:
            return 0
        return sum(1 for child in arguments_node.children if child.is_named)

    # ------------------------------------------------------------------
    # Helpers: Modifiers, Annotations, Parameters, Docstrings
    # ------------------------------------------------------------------

    def _extract_modifiers(self, node: Node, code: bytes) -> list[str]:
        modifiers: list[str] = []
        for child in node.children:
            if child.type == "modifiers":
                for m in child.children:
                    if m.type not in ("marker_annotation", "annotation"):
                        modifiers.append(self._text(m, code))
        return modifiers

    def _extract_annotations(self, node: Node, code: bytes) -> list[dict[str, str | None]]:
        annotations: list[dict[str, str | None]] = []
        for child in node.children:
            if child.type == "modifiers":
                for m in child.children:
                    if m.type == "marker_annotation":
                        name_node = m.child_by_field_name("name")
                        if name_node:
                            annotations.append({
                                "name": self._text(name_node, code),
                                "value": None,
                            })
                    elif m.type == "annotation":
                        name_node = m.child_by_field_name("name")
                        args_node = m.child_by_field_name("arguments")
                        name = self._text(name_node, code) if name_node else None
                        value = None
                        if args_node:
                            for arg_child in args_node.children:
                                if arg_child.type == "string_literal":
                                    value = self._text(arg_child, code).strip('"\'')
                                    break
                        if name:
                            annotations.append({"name": name, "value": value})
        return annotations

    def _extract_parameters(self, node: Node, code: bytes) -> list[dict[str, str]]:
        params: list[dict[str, str]] = []
        params_node = node.child_by_field_name("parameters")
        if params_node:
            for child in params_node.children:
                if child.type == "formal_parameter":
                    type_node = child.child_by_field_name("type")
                    name_node = child.child_by_field_name("name")
                    if type_node and name_node:
                        params.append({
                            "type": self._text(type_node, code),
                            "name": self._text(name_node, code),
                        })
        return params

    def _extract_docstring(self, node: Node, code: bytes) -> str | None:
        prev = node.prev_sibling
        while prev:
            if prev.type == "block_comment":
                text = self._text(prev, code)
                if text.startswith("/**"):
                    lines = text.split("\n")
                    cleaned: list[str] = []
                    for line in lines:
                        line = line.strip()
                        if line.startswith("/**"):
                            line = line[3:].strip()
                        elif line.startswith("*/"):
                            continue
                        elif line.startswith("*"):
                            line = line[1:].strip()
                        if line:
                            cleaned.append(line)
                    return "\n".join(cleaned) if cleaned else None
                break
            elif prev.type not in ("line_comment", "block_comment"):
                break
            prev = prev.prev_sibling
        return None

    def _extract_type_name(self, node: Node, code: bytes) -> str | None:
        if node.type == "type_identifier":
            return self._text(node, code)
        elif node.type == "generic_type":
            for child in node.children:
                if child.type == "type_identifier":
                    return self._text(child, code)
        return self._text(node, code)

    def _is_test_method(
        self,
        method_name: str,
        annotations: list[dict[str, str | None]],
    ) -> bool:
        annotation_names = {a["name"] for a in annotations}
        if annotation_names & TEST_ANNOTATIONS:
            return True
        return False

    def _is_endpoint_method(
        self,
        annotations: list[dict[str, str | None]],
        class_annotations: list[dict[str, str | None]],
    ) -> bool:
        annotation_names = {a["name"] for a in annotations}
        has_endpoint_anno = bool(annotation_names & set(ENDPOINT_METHOD_ANNOTATIONS.keys()))
        if has_endpoint_anno:
            return True
        return False

    # ------------------------------------------------------------------
    # Tree traversal utilities
    # ------------------------------------------------------------------

    def _find_nodes(self, node: Node, type_name: str) -> Iterator[Node]:
        if node.type == type_name:
            yield node
        for child in node.children:
            yield from self._find_nodes(child, type_name)

    def _direct_children(self, node: Node, type_name: str) -> Iterator[Node]:
        for child in node.children:
            if child.type == type_name:
                yield child

    def _text(self, node: Node, code: bytes) -> str:
        return code[node.start_byte:node.end_byte].decode("utf-8")


# =============================================================================
# Parsing Context
# =============================================================================


class _ParsingContext:
    """Internal state maintained while parsing a single Java file."""

    def __init__(
        self,
        file_path: str,
        file_id: str,
        code: bytes,
        repository: str,
        package_name: str = "",
        current_class: str = "",
        current_class_full_name: str = "",
        class_annotations: list[dict[str, str | None]] | None = None,
        exact_call_targets: dict[tuple[str, int], list[str]] | None = None,
        call_targets_by_name: dict[str, list[str]] | None = None,
    ) -> None:
        self.file_path = file_path
        self.file_id = file_id
        self.code = code
        self.repository = repository
        self.package_name = package_name
        self.current_class = current_class
        self.current_class_full_name = current_class_full_name
        self.class_annotations = class_annotations or []
        self.exact_call_targets = exact_call_targets or {}
        self.call_targets_by_name = call_targets_by_name or {}

    def with_class(
        self,
        class_name: str,
        class_full_name: str,
        class_annotations: list[dict[str, str | None]] | None = None,
        exact_call_targets: dict[tuple[str, int], list[str]] | None = None,
        call_targets_by_name: dict[str, list[str]] | None = None,
    ) -> _ParsingContext:
        return _ParsingContext(
            file_path=self.file_path,
            file_id=self.file_id,
            code=self.code,
            repository=self.repository,
            package_name=self.package_name,
            current_class=class_name,
            current_class_full_name=class_full_name,
            class_annotations=class_annotations or [],
            exact_call_targets=exact_call_targets,
            call_targets_by_name=call_targets_by_name,
        )
