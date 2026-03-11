"""Base parser abstract class and ParseResult dataclass."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from constellation.models import CodeEntity, CodeRelationship


@dataclass
class ParseResult:
    """Result of parsing a single source file.

    Attributes:
        file_path: Path to the parsed file.
        language: Programming language of the file.
        entities: Code entities discovered during parsing.
        relationships: Relationships between entities.
        errors: Error messages encountered during parsing.
    """

    file_path: str
    language: str
    entities: list[CodeEntity] = field(default_factory=list)
    relationships: list[CodeRelationship] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """True if parsing produced no errors."""
        return len(self.errors) == 0

    def add_entity(self, entity: CodeEntity) -> None:
        """Append a code entity to the result."""
        self.entities.append(entity)

    def add_relationship(self, relationship: CodeRelationship) -> None:
        """Append a relationship to the result."""
        self.relationships.append(relationship)

    def add_error(self, error: str) -> None:
        """Record a parsing error."""
        self.errors.append(error)


class BaseParser(ABC):
    """Abstract base class that all language parsers must implement."""

    @property
    @abstractmethod
    def language(self) -> str:
        """The programming language this parser handles (e.g. 'python')."""
        ...

    @property
    @abstractmethod
    def file_extensions(self) -> list[str]:
        """File extensions this parser can handle (e.g. ['.py', '.pyi'])."""
        ...

    @abstractmethod
    def parse_file(self, file_path: Path, repository: str) -> ParseResult:
        """Parse a source file and return structured results.

        Args:
            file_path: Path to the source file.
            repository: Name/identifier of the repository the file belongs to.

        Returns:
            A ParseResult containing entities, relationships, and any errors.
        """
        ...

    def can_parse(self, file_path: Path) -> bool:
        """Check whether this parser supports the given file.

        Matching is based on the file's suffix against ``file_extensions``.
        """
        return file_path.suffix in self.file_extensions
