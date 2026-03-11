from constellation.config import Settings
from constellation.embeddings.base import BaseEmbeddingProvider


def create_embedding_provider(provider_name: str, settings: Settings) -> BaseEmbeddingProvider:
    if provider_name == "openai":
        from constellation.embeddings.openai import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.resolved_embedding_model("openai"),
            dimensions=settings.resolved_embedding_dimensions("openai"),
            base_url=settings.openai_base_url or None,
        )
    elif provider_name == "ollama":
        from constellation.embeddings.ollama import OllamaEmbeddingProvider

        return OllamaEmbeddingProvider(
            base_url=settings.ollama_base_url,
            model=settings.resolved_embedding_model("ollama"),
            dimensions=settings.resolved_embedding_dimensions("ollama"),
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider_name}")
