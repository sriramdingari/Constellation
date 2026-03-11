"""Tests for constellation.parsers.registry module."""

from pathlib import Path

import pytest

import constellation.parsers.registry as registry_module
from constellation.parsers.base import BaseParser, ParseResult
from constellation.parsers.dotnet import DotNetParser
from constellation.parsers.java import JavaParser
from constellation.parsers.javascript import JavaScriptParser
from constellation.parsers.python_parser import PythonParser
from constellation.parsers.registry import ParserRegistry, get_default_registry


# --- Fake parser for testing ---

class FakeParser(BaseParser):
    """A fake parser used in registry tests."""

    def __init__(self, language_name: str, extensions: list[str]):
        self._language_name = language_name
        self._extensions = extensions

    @property
    def language(self) -> str:
        return self._language_name

    @property
    def file_extensions(self) -> list[str]:
        return self._extensions

    def parse_file(self, file_path: Path, repository: str) -> ParseResult:
        return ParseResult(file_path=str(file_path), language=self.language)


# --- Helpers ---

def _python_parser() -> FakeParser:
    return FakeParser("python", [".py", ".pyi"])


def _java_parser() -> FakeParser:
    return FakeParser("java", [".java"])


def _typescript_parser() -> FakeParser:
    return FakeParser("typescript", [".ts", ".tsx"])


# --- Tests ---


@pytest.fixture
def reset_default_registry():
    original = registry_module._default_registry
    registry_module._default_registry = None
    try:
        yield
    finally:
        registry_module._default_registry = original


class TestParserRegistryRegisterAndLookup:
    """register() adds a parser; get_parser_for_file() retrieves it."""

    def test_register_and_lookup_by_extension(self):
        registry = ParserRegistry()
        py_parser = _python_parser()
        registry.register(py_parser)
        found = registry.get_parser_for_file(Path("src/main.py"))
        assert found is py_parser

    def test_lookup_alternate_extension(self):
        registry = ParserRegistry()
        py_parser = _python_parser()
        registry.register(py_parser)
        found = registry.get_parser_for_file(Path("stubs/types.pyi"))
        assert found is py_parser

    def test_lookup_unknown_extension_returns_none(self):
        registry = ParserRegistry()
        registry.register(_python_parser())
        assert registry.get_parser_for_file(Path("main.rb")) is None

    def test_lookup_no_extension_returns_none(self):
        registry = ParserRegistry()
        registry.register(_python_parser())
        assert registry.get_parser_for_file(Path("Makefile")) is None


class TestParserRegistrySupportedExtensions:
    """supported_extensions property lists all registered extensions."""

    def test_supported_extensions(self):
        registry = ParserRegistry()
        registry.register(_python_parser())
        registry.register(_java_parser())
        exts = registry.supported_extensions
        assert ".py" in exts
        assert ".pyi" in exts
        assert ".java" in exts

    def test_supported_extensions_empty_registry(self):
        registry = ParserRegistry()
        assert registry.supported_extensions == set() or registry.supported_extensions == []


class TestParserRegistryEmpty:
    """Empty registry returns None for any lookup."""

    def test_empty_registry_returns_none(self):
        registry = ParserRegistry()
        assert registry.get_parser_for_file(Path("anything.py")) is None

    def test_empty_registry_supported_extensions_empty(self):
        registry = ParserRegistry()
        assert len(registry.supported_extensions) == 0


class TestParserRegistryMultipleExtensions:
    """A parser with multiple extensions is reachable via each one."""

    def test_all_extensions_map_to_same_parser(self):
        registry = ParserRegistry()
        ts_parser = _typescript_parser()
        registry.register(ts_parser)
        assert registry.get_parser_for_file(Path("app.ts")) is ts_parser
        assert registry.get_parser_for_file(Path("component.tsx")) is ts_parser


class TestParserRegistryOverride:
    """Later registration overrides earlier for the same extension."""

    def test_override_extension(self):
        registry = ParserRegistry()
        parser_a = FakeParser("lang-a", [".xx"])
        parser_b = FakeParser("lang-b", [".xx"])
        registry.register(parser_a)
        registry.register(parser_b)
        found = registry.get_parser_for_file(Path("file.xx"))
        assert found is parser_b


class TestDefaultRegistry:
    """The lazy default registry should register the built-in parser set."""

    def test_registers_all_builtin_extensions(self, reset_default_registry):
        registry = get_default_registry()

        assert {".py", ".java", ".js", ".jsx", ".ts", ".tsx", ".cs"} <= (
            registry.supported_extensions
        )

    def test_looks_up_builtin_parsers(self, reset_default_registry):
        registry = get_default_registry()

        assert isinstance(registry.get_parser_for_file(Path("main.py")), PythonParser)
        assert isinstance(registry.get_parser_for_file(Path("App.java")), JavaParser)
        assert isinstance(
            registry.get_parser_for_file(Path("component.tsx")),
            JavaScriptParser,
        )
        assert isinstance(registry.get_parser_for_file(Path("Program.cs")), DotNetParser)

    def test_returns_singleton_instance(self, reset_default_registry):
        registry = get_default_registry()

        assert get_default_registry() is registry
