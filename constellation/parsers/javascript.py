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
CALLABLE_SCOPE_BARRIERS = frozenset(
    {
        "arrow_function",
        "class",
        "class_body",
        "class_declaration",
        "function",
        "function_declaration",
        "generator_function",
        "method_definition",
    }
)
CALL_TARGET_WRAPPER_TYPES = frozenset(
    {
        "as_expression",
        "instantiation_expression",
        "non_null_expression",
        "parenthesized_expression",
        "satisfies_expression",
        "type_assertion",
    }
)


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
    module_callable_ids: dict[str, str] = field(default_factory=dict)
    imported_callable_ids: dict[str, str] = field(default_factory=dict)
    imported_namespace_modules: dict[str, str] = field(default_factory=dict)
    class_method_ids: dict[str, dict[str, str]] = field(default_factory=dict)
    class_static_method_ids: dict[str, dict[str, str]] = field(default_factory=dict)
    current_class_method_ids: dict[str, str] = field(default_factory=dict)
    current_class_static_method_ids: dict[str, str] = field(default_factory=dict)
    local_callable_scopes: list[dict[str, str | None]] = field(default_factory=list)
    local_instance_scopes: list[dict[str, str]] = field(default_factory=list)

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
        self._collect_imports(tree.root_node, ctx)
        self._collect_local_callables(tree.root_node, ctx)
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

    def _collect_imports(self, root: Node, ctx: _ParsingContext) -> None:
        for child in root.children:
            if child.type != "import_statement":
                continue

            source_node = child.child_by_field_name("source")
            if not source_node:
                for sub in child.children:
                    if sub.type == "string":
                        source_node = sub
                        break
            if not source_node:
                continue

            module_name = self._import_module_name(self._get_text(source_node, ctx.code).strip("'\""))

            clause_node = child.child_by_field_name("clause")
            if not clause_node:
                for sub in child.children:
                    if sub.type == "import_clause":
                        clause_node = sub
                        break
            if not clause_node:
                continue

            for sub in clause_node.children:
                if sub.type == "identifier":
                    continue
                if sub.type == "namespace_import":
                    alias_node = sub.child_by_field_name("name")
                    if not alias_node:
                        for grandchild in sub.children:
                            if grandchild.type == "identifier":
                                alias_node = grandchild
                                break
                    if alias_node:
                        ctx.imported_namespace_modules[self._get_text(alias_node, ctx.code)] = module_name
                    continue
                if sub.type != "named_imports":
                    continue
                for spec in sub.children:
                    if spec.type != "import_specifier":
                        continue
                    imported_node = spec.child_by_field_name("name")
                    if not imported_node:
                        for grandchild in spec.children:
                            if grandchild.type == "identifier":
                                imported_node = grandchild
                                break
                    if not imported_node:
                        continue
                    imported_name = self._get_text(imported_node, ctx.code)
                    alias_node = spec.child_by_field_name("alias")
                    if not alias_node:
                        identifiers = [grandchild for grandchild in spec.children if grandchild.type == "identifier"]
                        if len(identifiers) > 1:
                            alias_node = identifiers[-1]
                    local_name = imported_name
                    if alias_node:
                        local_name = self._get_text(alias_node, ctx.code)
                    ctx.imported_callable_ids[local_name] = ctx.entity_id(module_name, imported_name)

    def _collect_local_callables(self, root: Node, ctx: _ParsingContext) -> None:
        for child in root.children:
            if child.type == "export_statement":
                for sub in child.children:
                    self._collect_top_level_callables(sub, ctx)
            else:
                self._collect_top_level_callables(child, ctx)

    def _collect_top_level_callables(self, node: Node, ctx: _ParsingContext) -> None:
        if node.type == "function_declaration":
            name_node = node.child_by_field_name("name")
            if not name_node:
                return
            func_name = self._get_text(name_node, ctx.code)
            ctx.module_callable_ids[func_name] = ctx.entity_id(ctx.module_name, func_name)
            return

        if node.type in ("lexical_declaration", "variable_declaration"):
            for declarator in self._find_nodes(node, "variable_declarator"):
                name_node = declarator.child_by_field_name("name")
                value_node = declarator.child_by_field_name("value")
                if not name_node or name_node.type != "identifier" or not value_node:
                    continue
                if self._resolve_function_value(value_node) is None:
                    continue
                func_name = self._get_text(name_node, ctx.code)
                ctx.module_callable_ids[func_name] = ctx.entity_id(ctx.module_name, func_name)
            return

        if node.type != "class_declaration":
            return

        name_node = node.child_by_field_name("name")
        body = node.child_by_field_name("body")
        if not name_node or not body:
            return

        class_name = self._get_text(name_node, ctx.code)
        method_ids: dict[str, str] = {}
        static_method_ids: dict[str, str] = {}
        for child in body.children:
            if child.type != "method_definition":
                continue
            method_name_node = child.child_by_field_name("name")
            if not method_name_node:
                continue
            method_name = self._get_text(method_name_node, ctx.code)
            method_id = ctx.entity_id(ctx.module_name, class_name, method_name)
            method_ids[method_name] = method_id
            if any(grandchild.type == "static" for grandchild in child.children):
                static_method_ids[method_name] = method_id
        if method_ids:
            ctx.class_method_ids[class_name] = method_ids
        if static_method_ids:
            ctx.class_static_method_ids[class_name] = static_method_ids

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
            saved_class_method_ids = ctx.current_class_method_ids
            saved_class_static_method_ids = ctx.current_class_static_method_ids
            ctx.current_class = class_name
            ctx.current_class_full_id = class_id
            ctx.current_class_method_ids = ctx.class_method_ids.get(class_name, {})
            ctx.current_class_static_method_ids = ctx.class_static_method_ids.get(class_name, {})
            self._process_class_body(body, ctx, result, class_id)
            ctx.current_class = saved_class
            ctx.current_class_full_id = saved_class_id
            ctx.current_class_method_ids = saved_class_method_ids
            ctx.current_class_static_method_ids = saved_class_static_method_ids

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
            self._process_callable_body(
                body,
                ctx,
                result,
                method_id,
                [ctx.current_class, method_name],
            )

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
            self._process_callable_body(body, ctx, result, func_id, [func_name])

    # -- Variable declaration (arrow functions) -> METHOD entity -------------

    def _process_variable_declaration(self, node: Node, ctx: _ParsingContext, result: ParseResult, file_id: str) -> None:
        for declarator in self._find_nodes(node, "variable_declarator"):
            name_node = declarator.child_by_field_name("name")
            value_node = declarator.child_by_field_name("value")
            if not name_node or not value_node:
                continue

            # We only care about arrow functions / plain functions assigned to a const.
            actual_func = self._resolve_function_value(value_node)
            if actual_func is None:
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
                self._process_callable_body(body, ctx, result, func_id, [var_name])

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

    # -- General call extraction (CALLS / REFERENCE) ------------------------

    def _process_callable_body(
        self,
        body: Node,
        ctx: _ParsingContext,
        result: ParseResult,
        source_id: str,
        scope_parts: list[str],
    ) -> None:
        local_callable_ids = self._collect_scope_callable_ids(body, ctx, scope_parts)
        local_instance_ids = self._collect_scope_instance_ids(body, ctx)
        ctx.local_callable_scopes.append(local_callable_ids)
        ctx.local_instance_scopes.append(local_instance_ids)
        try:
            self._process_nested_callables(body, ctx, result, source_id, scope_parts)
            self._extract_calls(body, ctx, result, source_id, scope_parts)
            self._extract_hook_calls(body, ctx, result, source_id)
        finally:
            ctx.local_instance_scopes.pop()
            ctx.local_callable_scopes.pop()

    def _collect_scope_callable_ids(
        self,
        node: Node,
        ctx: _ParsingContext,
        scope_parts: list[str],
    ) -> dict[str, str | None]:
        callable_ids = self._collect_parameter_bindings(node, ctx.code)
        for current in node.children:
            if current.type == "function_declaration":
                name_node = current.child_by_field_name("name")
                if not name_node:
                    continue
                func_name = self._get_text(name_node, ctx.code)
                callable_ids[func_name] = ctx.entity_id(ctx.module_name, *scope_parts, func_name)
                continue

            if current.type == "class_declaration":
                name_node = current.child_by_field_name("name")
                if name_node:
                    callable_ids[self._get_text(name_node, ctx.code)] = None
                continue

            if current.type not in ("lexical_declaration", "variable_declaration"):
                continue

            for declarator in current.children:
                if declarator.type != "variable_declarator":
                    continue
                name_node = declarator.child_by_field_name("name")
                value_node = declarator.child_by_field_name("value")
                if not name_node or name_node.type != "identifier":
                    continue
                func_name = self._get_text(name_node, ctx.code)
                if not value_node:
                    callable_ids[func_name] = None
                    continue
                if self._resolve_function_value(value_node) is None:
                    callable_ids[func_name] = None
                    continue
                callable_ids[func_name] = ctx.entity_id(ctx.module_name, *scope_parts, func_name)
        return callable_ids

    def _collect_scope_instance_ids(self, node: Node, ctx: _ParsingContext) -> dict[str, str]:
        instance_ids: dict[str, str] = {}
        for current in node.children:
            if current.type not in ("lexical_declaration", "variable_declaration"):
                continue

            for declarator in current.children:
                if declarator.type != "variable_declarator":
                    continue
                name_node = declarator.child_by_field_name("name")
                value_node = declarator.child_by_field_name("value")
                if not name_node or name_node.type != "identifier" or not value_node:
                    continue
                if value_node.type != "new_expression":
                    continue

                constructor_node = value_node.child_by_field_name("constructor")
                if constructor_node is None:
                    continue
                constructor_node = self._unwrap_call_target(constructor_node)
                if constructor_node.type != "identifier":
                    continue

                class_name = self._get_text(constructor_node, ctx.code)
                if class_name not in ctx.class_method_ids:
                    continue
                instance_ids[self._get_text(name_node, ctx.code)] = class_name
        return instance_ids

    def _collect_parameter_bindings(self, body: Node, code: bytes) -> dict[str, str | None]:
        callable_ids: dict[str, str | None] = {}
        parent = body.parent
        if parent is None:
            return callable_ids

        parameters = parent.child_by_field_name("parameters")
        if parameters is None:
            return callable_ids

        for child in parameters.children:
            pattern = child
            if child.type in ("required_parameter", "optional_parameter"):
                next_pattern = child.child_by_field_name("pattern")
                if next_pattern is None:
                    continue
                pattern = next_pattern
            if pattern.type == "identifier":
                callable_ids[self._get_text(pattern, code)] = None
        return callable_ids

    def _process_nested_callables(
        self,
        node: Node,
        ctx: _ParsingContext,
        result: ParseResult,
        container_id: str,
        scope_parts: list[str],
    ) -> None:
        for current in self._iter_scope_nodes(node):
            if current.type == "function_declaration":
                self._process_nested_function(current, ctx, result, container_id, scope_parts)
                continue
            if current.type == "variable_declarator":
                self._process_nested_variable_callable(current, ctx, result, container_id, scope_parts)

    def _process_nested_function(
        self,
        node: Node,
        ctx: _ParsingContext,
        result: ParseResult,
        container_id: str,
        scope_parts: list[str],
    ) -> None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        func_name = self._get_text(name_node, ctx.code)
        func_id = ctx.entity_id(ctx.module_name, *scope_parts, func_name)

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

        result.add_entity(CodeEntity(
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
        ))
        result.add_relationship(CodeRelationship(
            source_id=container_id,
            target_id=func_id,
            relationship_type=RelationshipType.CONTAINS,
        ))

        body = node.child_by_field_name("body")
        if body:
            self._process_callable_body(
                body,
                ctx,
                result,
                func_id,
                [*scope_parts, func_name],
            )

    def _process_nested_variable_callable(
        self,
        declarator: Node,
        ctx: _ParsingContext,
        result: ParseResult,
        container_id: str,
        scope_parts: list[str],
    ) -> None:
        name_node = declarator.child_by_field_name("name")
        value_node = declarator.child_by_field_name("value")
        if not name_node or name_node.type != "identifier" or not value_node:
            return

        actual_func = self._resolve_function_value(value_node)
        if actual_func is None:
            return

        var_name = self._get_text(name_node, ctx.code)
        func_id = ctx.entity_id(ctx.module_name, *scope_parts, var_name)

        modifiers: list[str] = []
        for child in actual_func.children:
            if child.type == "async":
                modifiers.append("async")

        return_type = self._extract_return_type(actual_func, ctx.code)
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

        result.add_entity(CodeEntity(
            id=func_id,
            name=var_name,
            entity_type=EntityType.METHOD,
            repository=ctx.repository,
            file_path=ctx.file_path,
            line_number=declarator.start_point[0] + 1,
            line_end=declarator.end_point[0] + 1,
            language=self.language,
            signature=signature,
            return_type=return_type,
            modifiers=modifiers,
            code=self._get_text(declarator, ctx.code),
        ))
        result.add_relationship(CodeRelationship(
            source_id=container_id,
            target_id=func_id,
            relationship_type=RelationshipType.CONTAINS,
        ))

        body = actual_func.child_by_field_name("body")
        if body:
            self._process_callable_body(
                body,
                ctx,
                result,
                func_id,
                [*scope_parts, var_name],
            )

    def _iter_scope_nodes(self, node: Node) -> Iterator[Node]:
        stack = [node]
        while stack:
            current = stack.pop()
            yield current
            if current is not node and current.type in CALLABLE_SCOPE_BARRIERS:
                continue
            for child in reversed(current.children):
                stack.append(child)

    def _extract_calls(
        self,
        node: Node,
        ctx: _ParsingContext,
        result: ParseResult,
        source_id: str,
        scope_parts: list[str],
    ) -> None:
        self._extract_calls_in_scope(node, ctx, result, source_id, scope_parts, set(), is_scope_root=True)

    def _extract_calls_in_scope(
        self,
        node: Node,
        ctx: _ParsingContext,
        result: ParseResult,
        source_id: str,
        scope_parts: list[str],
        seen_targets: set[str],
        *,
        is_scope_root: bool = False,
    ) -> None:
        entered_scope = False
        if not is_scope_root and node.type == "statement_block":
            ctx.local_callable_scopes.append(self._collect_scope_callable_ids(node, ctx, scope_parts))
            ctx.local_instance_scopes.append(self._collect_scope_instance_ids(node, ctx))
            entered_scope = True

        try:
            if node.type == "call_expression":
                self._record_call(node, ctx, result, source_id, seen_targets)

            if not is_scope_root and node.type in CALLABLE_SCOPE_BARRIERS:
                return

            for child in node.children:
                self._extract_calls_in_scope(child, ctx, result, source_id, scope_parts, seen_targets)
        finally:
            if entered_scope:
                ctx.local_instance_scopes.pop()
                ctx.local_callable_scopes.pop()

    def _record_call(
        self,
        call_node: Node,
        ctx: _ParsingContext,
        result: ParseResult,
        source_id: str,
        seen_targets: set[str],
    ) -> None:
        func_node = call_node.child_by_field_name("function")
        if not func_node:
            return

        wrapped_func_node = func_node
        func_node = self._unwrap_call_target(func_node)
        called_symbol: str | None = None
        target_id: str | None = None
        receiver_text: str | None = None
        if func_node.type == "identifier":
            called_symbol = self._get_text(func_node, ctx.code)
            target_id = self._resolve_identifier_callable(called_symbol, ctx)
        elif func_node.type == "member_expression":
            object_node = func_node.child_by_field_name("object")
            if object_node:
                receiver_text = self._get_text(object_node, ctx.code)
            called_symbol, target_id = self._resolve_member_call(func_node, ctx)
        else:
            called_symbol = self._get_text(wrapped_func_node, ctx.code).strip()

        if not called_symbol:
            return

        if target_id is None:
            target_id = self._reference_target_id(source_id, called_symbol, call_node)
            if target_id not in seen_targets:
                properties = {"symbol": called_symbol}
                if receiver_text:
                    properties["receiver"] = receiver_text
                properties["enclosing_declaration_id"] = source_id
                properties["enclosing_declaration_name"] = self._enclosing_declaration_name(source_id)
                result.add_entity(CodeEntity(
                    id=target_id,
                    name=called_symbol,
                    entity_type=EntityType.REFERENCE,
                    repository=ctx.repository,
                    file_path=ctx.file_path,
                    line_number=call_node.start_point[0] + 1,
                    line_end=call_node.end_point[0] + 1,
                    language=self.language,
                    properties=properties,
                ))

        if target_id in seen_targets:
            return
        seen_targets.add(target_id)
        result.add_relationship(CodeRelationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type=RelationshipType.CALLS,
        ))

    def _resolve_identifier_callable(self, call_name: str, ctx: _ParsingContext) -> str | None:
        for scope in reversed(ctx.local_callable_scopes):
            if call_name in scope:
                return scope[call_name]

        target_id = ctx.module_callable_ids.get(call_name)
        if target_id is not None:
            return target_id

        return ctx.imported_callable_ids.get(call_name)

    def _resolve_member_call(self, node: Node, ctx: _ParsingContext) -> tuple[str | None, str | None]:
        object_node = node.child_by_field_name("object")
        property_node = node.child_by_field_name("property")
        if not object_node or not property_node:
            return None, None

        object_node = self._unwrap_call_target(object_node)
        object_text = self._get_text(object_node, ctx.code)
        property_name = self._get_text(property_node, ctx.code)
        called_symbol = f"{object_text}.{property_name}"

        if object_text == "this":
            return called_symbol, ctx.current_class_method_ids.get(property_name)
        for scope in reversed(ctx.local_instance_scopes):
            class_name = scope.get(object_text)
            if class_name is not None:
                return called_symbol, ctx.class_method_ids.get(class_name, {}).get(property_name)
        for scope in reversed(ctx.local_callable_scopes):
            if object_text in scope:
                return called_symbol, None
        if object_text == ctx.current_class:
            return called_symbol, ctx.current_class_static_method_ids.get(property_name)
        if object_text in ctx.class_static_method_ids:
            return called_symbol, ctx.class_static_method_ids.get(object_text, {}).get(property_name)
        imported_module = ctx.imported_namespace_modules.get(object_text)
        if imported_module:
            return called_symbol, ctx.entity_id(imported_module, property_name)
        return called_symbol, None

    def _reference_target_id(self, source_id: str, called_symbol: str, call_node: Node) -> str:
        line_number = call_node.start_point[0] + 1
        column_number = call_node.start_point[1] + 1
        return f"{source_id}::ref:{line_number}:{column_number}:{called_symbol}"

    @staticmethod
    def _enclosing_declaration_name(source_id: str) -> str:
        local_id = source_id.split("::", 1)[-1]
        return local_id.rsplit(".", 1)[-1]

    def _unwrap_call_target(self, node: Node) -> Node:
        current = node
        while current.type in CALL_TARGET_WRAPPER_TYPES:
            next_node = current.child_by_field_name("expression")
            if next_node is None:
                named_children = [child for child in current.children if child.is_named]
                if not named_children:
                    break
                next_node = named_children[-1] if current.type == "type_assertion" else named_children[0]
            if next_node == current:
                break
            current = next_node
        return current

    @staticmethod
    def _import_module_name(specifier: str) -> str:
        module_name = Path(specifier).stem
        if module_name == "index":
            parent_name = Path(specifier).parent.name
            if parent_name:
                return parent_name
        return module_name

    # -- Hook call extraction (USES_HOOK) -----------------------------------

    def _extract_hook_calls(self, node: Node, ctx: _ParsingContext, result: ParseResult, source_id: str) -> None:
        seen: set[str] = set()
        for call_node in self._iter_scope_nodes(node):
            if call_node.type != "call_expression":
                continue
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

    def _resolve_function_value(self, value_node: Node) -> Node | None:
        actual_func = value_node
        if value_node.type == "call_expression":
            args_node = value_node.child_by_field_name("arguments")
            if args_node:
                for arg in args_node.children:
                    if arg.type in ("arrow_function", "function"):
                        return arg
            return None
        if actual_func.type in ("arrow_function", "function"):
            return actual_func
        return None

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
