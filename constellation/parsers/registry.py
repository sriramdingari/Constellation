"""Parser registry for mapping file extensions to language parsers."""

from __future__ import annotations

from pathlib import Path

from constellation.parsers.base import BaseParser


class ParserRegistry:
    """Registry that maps file extensions to parser instances.

    Attributes:
        _parsers: Mapping of language name to parser instance.
        _extension_map: Mapping of file extension to parser instance.
    """

    def __init__(self) -> None:
        self._parsers: dict[str, BaseParser] = {}
        self._extension_map: dict[str, BaseParser] = {}

    def register(self, parser: BaseParser) -> None:
        """Register a parser, mapping each of its extensions.

        If an extension is already registered, the new parser overrides it.
        """
        self._parsers[parser.language] = parser
        for ext in parser.file_extensions:
            self._extension_map[ext] = parser

    def get_parser_for_file(self, file_path: Path) -> BaseParser | None:
        """Return the parser that handles *file_path*, or ``None``."""
        return self._extension_map.get(file_path.suffix)

    @property
    def supported_extensions(self) -> set[str]:
        """Set of all currently registered file extensions."""
        return set(self._extension_map.keys())


# ---------------------------------------------------------------------------
# Default registry singleton
# ---------------------------------------------------------------------------

_default_registry: ParserRegistry | None = None


def _register_default_parsers(registry: ParserRegistry) -> None:
    """Register the four built-in language parsers (deferred imports)."""
    try:
        from constellation.parsers.java_parser import JavaParser
        registry.register(JavaParser())
    except ImportError:
        pass

    try:
        from constellation.parsers.kotlin_parser import KotlinParser
        registry.register(KotlinParser())
    except ImportError:
        pass

    try:
        from constellation.parsers.typescript_parser import TypeScriptParser
        registry.register(TypeScriptParser())
    except ImportError:
        pass

    try:
        from constellation.parsers.python_parser import PythonParser
        registry.register(PythonParser())
    except ImportError:
        pass


def get_default_registry() -> ParserRegistry:
    """Return the default :class:`ParserRegistry` (lazy singleton).

    On first call, creates the registry and registers all built-in parsers.
    Subsequent calls return the same instance.
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = ParserRegistry()
        _register_default_parsers(_default_registry)
    return _default_registry
