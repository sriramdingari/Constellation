from abc import ABC, abstractmethod

from constellation.models import CodeEntity, EntityType

EMBEDDABLE_TYPES = {EntityType.METHOD, EntityType.CLASS, EntityType.INTERFACE, EntityType.CONSTRUCTOR}
MAX_EMBEDDING_TEXT_LENGTH = 2500


def is_embeddable(entity_type: EntityType) -> bool:
    return entity_type in EMBEDDABLE_TYPES


def prepare_embedding_text(entity: CodeEntity) -> str:
    parts = [f"[{entity.entity_type.value}] {entity.id}"]
    if entity.signature:
        parts.append(f"Signature: {entity.signature}")
    if entity.stereotypes:
        parts.append(f"Stereotypes: {', '.join(entity.stereotypes)}")
    if entity.docstring:
        parts.append(f"Docstring: {entity.docstring}")
    if entity.code:
        parts.append(f"Code: {entity.code}")
    text = "\n".join(parts)
    if len(text) > MAX_EMBEDDING_TEXT_LENGTH:
        text = text[:MAX_EMBEDDING_TEXT_LENGTH]
    return text


class BaseEmbeddingProvider(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def dimensions(self) -> int: ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
