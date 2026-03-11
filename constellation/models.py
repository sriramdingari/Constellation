from enum import Enum
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    FILE = "File"
    PACKAGE = "Package"
    CLASS = "Class"
    INTERFACE = "Interface"
    METHOD = "Method"
    CONSTRUCTOR = "Constructor"
    FIELD = "Field"


class RelationshipType(str, Enum):
    CONTAINS = "CONTAINS"
    IN_PACKAGE = "IN_PACKAGE"
    HAS_METHOD = "HAS_METHOD"
    HAS_CONSTRUCTOR = "HAS_CONSTRUCTOR"
    HAS_FIELD = "HAS_FIELD"
    DECLARES = "DECLARES"
    EXTENDS = "EXTENDS"
    IMPLEMENTS = "IMPLEMENTS"
    OVERRIDES = "OVERRIDES"
    CALLS = "CALLS"
    USES_TYPE = "USES_TYPE"
    IMPORTS = "IMPORTS"
    EXPORTS = "EXPORTS"
    DEPENDS_ON = "DEPENDS_ON"
    PRODUCES_BEAN = "PRODUCES_BEAN"
    THROWS = "THROWS"
    RENDERS = "RENDERS"
    USES_HOOK = "USES_HOOK"
    PROVIDES_CONTEXT = "PROVIDES_CONTEXT"
    CONSUMES_CONTEXT = "CONSUMES_CONTEXT"


class CodeEntity(BaseModel):
    id: str
    name: str
    entity_type: EntityType
    repository: str
    file_path: str
    line_number: int
    line_end: int | None = None
    language: str
    code: str | None = None
    signature: str | None = None
    return_type: str | None = None
    docstring: str | None = None
    modifiers: list[str] = Field(default_factory=list)
    stereotypes: list[str] = Field(default_factory=list)
    properties: dict = Field(default_factory=dict)
    content_hash: str | None = None
    embedding: list[float] | None = None


class CodeRelationship(BaseModel):
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    properties: dict = Field(default_factory=dict)
