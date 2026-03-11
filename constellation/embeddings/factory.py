from constellation.config import Settings
from constellation.embeddings.base import BaseEmbeddingProvider


def create_embedding_provider(provider_name: str, settings: Settings) -> BaseEmbeddingProvider:
    if provider_name == "openai":
        from constellation.embeddings.openai import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            base_url=settings.openai_base_url or None,
        )
    elif provider_name == "ollama":
        from constellation.embeddings.ollama import OllamaEmbeddingProvider

        return OllamaEmbeddingProvider(
            base_url=settings.ollama_base_url,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider_name}")
