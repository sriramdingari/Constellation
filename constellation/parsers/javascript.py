"""JavaScript/TypeScript parser using Tree-sitter."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript
from tree_sitter import Language, Node, Parser

from constellation.models import (
    CodeEntity,
    CodeRelationship,
    EntityType,
    RelationshipType,
)
from constellation.parsers.base import BaseParser, ParseResult

logger = logging.getLogger(__name__)

# Language objects -- initialised once at module level.
TS_LANGUAGE = Language(tstypescript.language_typescript())
TSX_LANGUAGE = Language(tstypescript.language_tsx())
JS_LANGUAGE = Language(tsjavascript.language())

REACT_HOOKS = frozenset(
    {
        "useState",
        "useEffect",
        "useContext",
        "useReducer",
        "useCallback",
        "useMemo",
        "useRef",
        "useImperativeHandle",
        "useLayoutEffect",
        "useDebugValue",
        "useDeferredValue",
        "useTransition",
        "useId",
        "useSyncExternalStore",
        "useInsertionEffect",
    }
)

TEST_CALL_NAMES = frozenset({"describe", "it", "test", "beforeEach", "afterEach", "beforeAll", "afterAll"})


@dataclass
class _ParsingContext:
    """Mutable state threaded through the recursive walk."""

    file_path: str
    repository: str
    code: bytes
    module_name: str = ""
    current_class: str = ""
    current_class_full_id: str = ""

    # Export tracking (populated in first pass)
    exported_names: set[str] = field(default_factory=set)
    default_export: str | None = None
    emitted_hooks: set[str] = field(default_factory=set)

    def entity_id(self, *parts: str) -> str:
        """Build an entity ID in the format ``{repository}::{qualified.name}``."""
        qualified = ".".join(p for p in parts if p)
        return f"{self.repository}::{qualified}"

    def is_exported(self, name: str) -> tuple[str | None, bool]:
        if self.default_export == name:
            return "default", True
        if name in self.exported_names:
            return "named", False
        return None, False


class JavaScriptParser(BaseParser):
    """Parser for JavaScript/TypeScript files using Tree-sitter."""

    def __init__(self) -> None:
        self._ts_parser = Parser(TS_LANGUAGE)
        self._tsx_parser = Parser(TSX_LANGUAGE)
        self._js_parser = Parser(JS_LANGUAGE)

    # -- BaseParser interface -----------------------------------------------

    @property
    def language(self) -> str:
        return "javascript"

    @property
    def file_extensions(self) -> list[str]:
        return [".js", ".jsx", ".ts", ".tsx"]

    def parse_file(self, file_path: Path, repository: str) -> ParseResult:
        result = ParseResult(file_path=str(file_path), language=self.language)

        try:
            code = file_path.read_bytes()
            tree = self._parse_tree(file_path, code)
        except Exception as exc:
            result.add_error(f"Failed to parse {file_path}: {exc}")
            return result

        module_name = self._derive_module_name(file_path)

        ctx = _ParsingContext(
            file_path=str(file_path),
            repository=repository,
            code=code,
            module_name=module_name,
        )

        # File entity
        file_id = ctx.entity_id(module_name)
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

        # Two-pass approach: gather exports first, then definitions.
        self._collect_exports(tree.root_node, ctx)
        self._walk_root(tree.root_node, ctx, result, file_id)

        return result

    # -- Parsing helpers ----------------------------------------------------

    def _parse_tree(self, file_path: Path, code: bytes):
        suffix = file_path.suffix.lower()
        if suffix in (".tsx", ".jsx"):
            return self._tsx_parser.parse(code)
        if suffix == ".ts":
            return self._ts_parser.parse(code)
        return self._js_parser.parse(code)

    @staticmethod
    def _derive_module_name(file_path: Path) -> str:
        if file_path.stem == "index":
            return file_path.parent.name
        return file_path.stem

    @staticmethod
    def _get_text(node: Node, code: bytes) -> str:
        return code[node.start_byte : node.end_byte].decode("utf-8")

    @staticmethod
    def _find_nodes(node: Node, type_name: str) -> Iterator[Node]:
        if node.type == type_name:
            yield node
        for child in node.children:
            yield from JavaScriptParser._find_nodes(child, type_name)

    # -- Export collection (first pass) -------------------------------------

    def _collect_exports(self, root: Node, ctx: _ParsingContext) -> None:
        for child in root.children:
            if child.type != "export_statement":
                continue
            is_default = any(c.type == "default" for c in child.children)
            for sub in child.children:
                if is_default:
                    if sub.type == "identifier":
                        ctx.default_export = self._get_text(sub, ctx.code)
                    elif sub.type in ("function_declaration", "class_declaration"):
                        name_node = sub.child_by_field_name("name")
                        if name_node:
                            ctx.default_export = self._get_text(name_node, ctx.code)
                else:
                    if sub.type == "export_clause":
                        for spec in sub.children:
                            if spec.type == "export_specifier":
                                name_node = spec.child_by_field_name("name")
                                if name_node:
                                    ctx.exported_names.add(self._get_text(name_node, ctx.code))
                    elif sub.type in (
                        "function_declaration",
                        "class_declaration",
                        "interface_declaration",
                    ):
                        name_node = sub.child_by_field_name("name")
                        if name_node:
                            ctx.exported_names.add(self._get_text(name_node, ctx.code))
                    elif sub.type in ("lexical_declaration", "variable_declaration"):
                        for decl in self._find_nodes(sub, "variable_declarator"):
                            name_node = decl.child_by_field_name("name")
                            if name_node and name_node.type == "identifier":
                                ctx.exported_names.add(self._get_text(name_node, ctx.code))

    # -- Main walk (second pass) --------------------------------------------

    def _walk_root(self, root: Node, ctx: _ParsingContext, result: ParseResult, file_id: str) -> None:
        for child in root.children:
            if child.type == "export_statement":
                for sub in child.children:
                    self._process_top_level(sub, ctx, result, file_id)
            else:
                self._process_top_level(child, ctx, result, file_id)

        # Create EXPORTS relationships for all tracked exports.
        self._emit_export_relationships(ctx, result, file_id)

    def _process_top_level(self, node: Node, ctx: _ParsingContext, result: ParseResult, file_id: str) -> None:
        if node.type == "class_declaration":
            self._process_class(node, ctx, result, file_id)
        elif node.type == "interface_declaration":
            self._process_interface(node, ctx, result, file_id)
        elif node.type == "function_declaration":
            self._process_function(node, ctx, result, file_id)
        elif node.type in ("lexical_declaration", "variable_declaration"):
            self._process_variable_declaration(node, ctx, result, file_id)
        elif node.type in ("internal_module", "module"):
            self._process_namespace(node, ctx, result, file_id)
        elif node.type == "expression_statement":
            # Namespace declarations can be wrapped in expression_statement
            for sub in node.children:
                if sub.type in ("internal_module", "module"):
                    self._process_namespace(sub, ctx, result, file_id)
                    return
            self._check_test_expression(node, ctx, result, file_id)

    # -- Namespace (TypeScript) -> PACKAGE ----------------------------------

    def _process_namespace(self, node: Node, ctx: _ParsingContext, result: ParseResult, file_id: str) -> None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        ns_name = self._get_text(name_node, ctx.code)
        ns_id = ctx.entity_id(ctx.module_name, ns_name)
        entity = CodeEntity(
            id=ns_id,
            name=ns_name,
            entity_type=EntityType.PACKAGE,
            repository=ctx.repository,
            file_path=ctx.file_path,
            line_number=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            language=self.language,
        )
        result.add_entity(entity)
        result.add_relationship(CodeRelationship(
            source_id=file_id,
            target_id=ns_id,
            relationship_type=RelationshipType.CONTAINS,
        ))

    # -- Interface -> INTERFACE entity --------------------------------------

    def _process_interface(self, node: Node, ctx: _ParsingContext, result: ParseResult, file_id: str) -> None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        iface_name = self._get_text(name_node, ctx.code)
        iface_id = ctx.entity_id(ctx.module_name, iface_name)
        entity = CodeEntity(
            id=iface_id,
            name=iface_name,
            entity_type=EntityType.INTERFACE,
            repository=ctx.repository,
            file_path=ctx.file_path,
            line_number=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            language=self.language,
            code=self._get_text(node, ctx.code),
        )
        result.add_entity(entity)
        result.add_relationship(CodeRelationship(
            source_id=file_id,
            target_id=iface_id,
            relationship_type=RelationshipType.CONTAINS,
        ))

    # -- Class -> CLASS entity + body walk -----------------------------------

    def _process_class(self, node: Node, ctx: _ParsingContext, result: ParseResult, file_id: str) -> None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        class_name = self._get_text(name_node, ctx.code)
        class_id = ctx.entity_id(ctx.module_name, class_name)
        entity = CodeEntity(
            id=class_id,
            name=class_name,
            entity_type=EntityType.CLASS,
            repository=ctx.repository,
            file_path=ctx.file_path,
            line_number=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            language=self.language,
            code=self._get_text(node, ctx.code),
        )
        result.add_entity(entity)
        result.add_relationship(CodeRelationship(
            source_id=file_id,
            target_id=class_id,
            relationship_type=RelationshipType.CONTAINS,
        ))

        # Walk class body
        body = node.child_by_field_name("body")
        if body:
            saved_class = ctx.current_class
            saved_class_id = ctx.current_class_full_id
            ctx.current_class = class_name
            ctx.current_class_full_id = class_id
            self._process_class_body(body, ctx, result, class_id)
            ctx.current_class = saved_class
            ctx.current_class_full_id = saved_class_id

    def _process_class_body(self, body: Node, ctx: _ParsingContext, result: ParseResult, class_id: str) -> None:
        for child in body.children:
            if child.type == "method_definition":
                self._process_class_method(child, ctx, result, class_id)

    def _process_class_method(self, node: Node, ctx: _ParsingContext, result: ParseResult, class_id: str) -> None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        method_name = self._get_text(name_node, ctx.code)
        is_constructor = method_name == "constructor"

        # Modifiers
        modifiers: list[str] = []
        for child in node.children:
            if child.type in ("public", "private", "protected", "static", "async", "readonly", "abstract"):
                modifiers.append(child.type)

        # Return type
        return_type = self._extract_return_type(node, ctx.code)

        # Build signature
        params = self._extract_parameters(node, ctx.code)
        param_str = ", ".join(
            f"{p['name']}: {p['type']}" if p.get("type") else p["name"] for p in params
        )
        signature = f"{method_name}({param_str})"
        if return_type:
            signature += f": {return_type}"

        method_id = ctx.entity_id(ctx.module_name, ctx.current_class, method_name)
        entity_type = EntityType.CONSTRUCTOR if is_constructor else EntityType.METHOD
        entity = CodeEntity(
            id=method_id,
            name=method_name,
            entity_type=entity_type,
            repository=ctx.repository,
            file_path=ctx.file_path,
            line_number=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            language=self.language,
            signature=signature,
            return_type=return_type,
            modifiers=modifiers,
            code=self._get_text(node, ctx.code),
        )
        result.add_entity(entity)

        rel_type = RelationshipType.HAS_CONSTRUCTOR if is_constructor else RelationshipType.HAS_METHOD
        result.add_relationship(CodeRelationship(
            source_id=class_id,
            target_id=method_id,
            relationship_type=rel_type,
        ))

        # Scan body for hook calls
        body = node.child_by_field_name("body")
        if body:
            self._extract_hook_calls(body, ctx, result, method_id)

    # -- Top-level function -> METHOD entity ---------------------------------

    def _process_function(self, node: Node, ctx: _ParsingContext, result: ParseResult, file_id: str) -> None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        func_name = self._get_text(name_node, ctx.code)
        func_id = ctx.entity_id(ctx.module_name, func_name)

        modifiers: list[str] = []
        for child in node.children:
            if child.type == "async":
                modifiers.append("async")

        return_type = self._extract_return_type(node, ctx.code)
        params = self._extract_parameters(node, ctx.code)
        param_str = ", ".join(
            f"{p['name']}: {p['type']}" if p.get("type") else p["name"] for p in params
        )
        signature = f"function {func_name}({param_str})"
        if return_type:
            signature += f": {return_type}"

        entity = CodeEntity(
            id=func_id,
            name=func_name,
            entity_type=EntityType.METHOD,
            repository=ctx.repository,
            file_path=ctx.file_path,
            line_number=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            language=self.language,
            signature=signature,
            return_type=return_type,
            modifiers=modifiers,
            code=self._get_text(node, ctx.code),
        )
        result.add_entity(entity)
        result.add_relationship(CodeRelationship(
            source_id=file_id,
            target_id=func_id,
            relationship_type=RelationshipType.CONTAINS,
        ))

        body = node.child_by_field_name("body")
        if body:
            self._extract_hook_calls(body, ctx, result, func_id)

    # -- Variable declaration (arrow functions) -> METHOD entity -------------

    def _process_variable_declaration(self, node: Node, ctx: _ParsingContext, result: ParseResult, file_id: str) -> None:
        for declarator in self._find_nodes(node, "variable_declarator"):
            name_node = declarator.child_by_field_name("name")
            value_node = declarator.child_by_field_name("value")
            if not name_node or not value_node:
                continue

            # We only care about arrow functions / plain functions assigned to a const.
            actual_func = value_node
            if value_node.type == "call_expression":
                # Could be an HOC wrapper -- skip for now unless wrapping an arrow/function.
                args_node = value_node.child_by_field_name("arguments")
                if args_node:
                    for arg in args_node.children:
                        if arg.type in ("arrow_function", "function"):
                            actual_func = arg
                            break
                if actual_func.type not in ("arrow_function", "function"):
                    continue
            elif value_node.type not in ("arrow_function", "function"):
                continue

            var_name = self._get_text(name_node, ctx.code)
            func_id = ctx.entity_id(ctx.module_name, var_name)

            modifiers: list[str] = []
            for child in actual_func.children:
                if child.type == "async":
                    modifiers.append("async")

            return_type = self._extract_return_type(actual_func, ctx.code)
            # Also check type annotation on the declarator itself.
            if not return_type:
                type_ann = declarator.child_by_field_name("type")
                if type_ann:
                    return_type = self._get_text(type_ann, ctx.code)

            params = self._extract_parameters(actual_func, ctx.code)
            param_str = ", ".join(
                f"{p['name']}: {p['type']}" if p.get("type") else p["name"] for p in params
            )
            signature = f"const {var_name} = ({param_str}) =>"
            if return_type:
                signature += f": {return_type}"

            entity = CodeEntity(
                id=func_id,
                name=var_name,
                entity_type=EntityType.METHOD,
                repository=ctx.repository,
                file_path=ctx.file_path,
                line_number=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                language=self.language,
                signature=signature,
                return_type=return_type,
                modifiers=modifiers,
                code=self._get_text(declarator, ctx.code),
            )
            result.add_entity(entity)
            result.add_relationship(CodeRelationship(
                source_id=file_id,
                target_id=func_id,
                relationship_type=RelationshipType.CONTAINS,
            ))

            body = actual_func.child_by_field_name("body")
            if body:
                self._extract_hook_calls(body, ctx, result, func_id)

    # -- Test stereotype detection ------------------------------------------

    def _check_test_expression(self, node: Node, ctx: _ParsingContext, result: ParseResult, file_id: str) -> None:
        """Check if an expression statement is a test call (describe/it/test)."""
        for call_node in self._find_nodes(node, "call_expression"):
            func_node = call_node.child_by_field_name("function")
            if not func_node:
                continue
            call_name: str | None = None
            if func_node.type == "identifier":
                call_name = self._get_text(func_node, ctx.code)
            elif func_node.type == "member_expression":
                obj = func_node.child_by_field_name("object")
                if obj:
                    call_name = self._get_text(obj, ctx.code)

            if call_name and call_name in TEST_CALL_NAMES:
                # Extract the description string (first argument)
                args = call_node.child_by_field_name("arguments")
                desc = call_name
                if args:
                    for arg in args.children:
                        if arg.type in ("string", "template_string"):
                            desc = self._get_text(arg, ctx.code).strip("'\"`")
                            break

                test_id = ctx.entity_id(ctx.module_name, call_name, desc.replace(" ", "_"))
                entity = CodeEntity(
                    id=test_id,
                    name=desc,
                    entity_type=EntityType.METHOD,
                    repository=ctx.repository,
                    file_path=ctx.file_path,
                    line_number=call_node.start_point[0] + 1,
                    line_end=call_node.end_point[0] + 1,
                    language=self.language,
                    stereotypes=["test"],
                )
                result.add_entity(entity)
                result.add_relationship(CodeRelationship(
                    source_id=file_id,
                    target_id=test_id,
                    relationship_type=RelationshipType.CONTAINS,
                ))
                # Only process the outermost test call at top level
                return

    # -- EXPORTS relationships (from collected names) -----------------------

    def _emit_export_relationships(self, ctx: _ParsingContext, result: ParseResult, file_id: str) -> None:
        for entity in result.entities:
            name = entity.name
            export_type, _ = ctx.is_exported(name)
            if export_type:
                result.add_relationship(CodeRelationship(
                    source_id=file_id,
                    target_id=entity.id,
                    relationship_type=RelationshipType.EXPORTS,
                    properties={"export_type": export_type},
                ))

    # -- Hook call extraction (USES_HOOK) -----------------------------------

    def _extract_hook_calls(self, node: Node, ctx: _ParsingContext, result: ParseResult, source_id: str) -> None:
        seen: set[str] = set()
        for call_node in self._find_nodes(node, "call_expression"):
            func_node = call_node.child_by_field_name("function")
            if not func_node or func_node.type != "identifier":
                continue
            call_name = self._get_text(func_node, ctx.code)
            if call_name in REACT_HOOKS or call_name.startswith("use"):
                if call_name not in seen:
                    seen.add(call_name)
                    hook_id = f"hook:{call_name}"
                    if hook_id not in ctx.emitted_hooks:
                        ctx.emitted_hooks.add(hook_id)
                        result.add_entity(CodeEntity(
                            id=hook_id,
                            name=call_name,
                            entity_type=EntityType.HOOK,
                            repository=ctx.repository,
                            file_path=ctx.file_path,
                            line_number=call_node.start_point[0] + 1,
                            line_end=call_node.end_point[0] + 1,
                            language=self.language,
                        ))
                    result.add_relationship(CodeRelationship(
                        source_id=source_id,
                        target_id=hook_id,
                        relationship_type=RelationshipType.USES_HOOK,
                    ))

    # -- Parameter / return-type extraction ----------------------------------

    def _extract_parameters(self, node: Node, code: bytes) -> list[dict[str, str]]:
        params: list[dict[str, str]] = []
        params_node = node.child_by_field_name("parameters")
        if not params_node:
            return params
        for child in params_node.children:
            if child.type in ("required_parameter", "optional_parameter"):
                pattern = child.child_by_field_name("pattern")
                type_node = child.child_by_field_name("type")
                if pattern:
                    name = self._get_text(pattern, code)
                    ptype = self._get_text(type_node, code) if type_node else ""
                    params.append({"name": name, "type": ptype})
            elif child.type == "identifier":
                params.append({"name": self._get_text(child, code), "type": ""})
        return params

    def _extract_return_type(self, node: Node, code: bytes) -> str | None:
        rt = node.child_by_field_name("return_type")
        if rt:
            text = self._get_text(rt, code)
            # Strip leading ": " if present
            if text.startswith(":"):
                text = text[1:].strip()
            return text
        return None
