from pydantic_settings import BaseSettings

DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_EMBEDDING_DIMENSIONS = 1536
DEFAULT_OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_EMBEDDING_DIMENSIONS = 768


class Settings(BaseSettings):
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "constellation"
    redis_url: str = "redis://localhost:6379"
    embedding_provider: str = "openai"
    embedding_model: str = DEFAULT_OPENAI_EMBEDDING_MODEL
    embedding_dimensions: int = DEFAULT_OPENAI_EMBEDDING_DIMENSIONS
    openai_api_key: str = ""
    openai_base_url: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = DEFAULT_OLLAMA_EMBEDDING_MODEL
    ollama_embedding_dimensions: int = DEFAULT_OLLAMA_EMBEDDING_DIMENSIONS
    embedding_batch_size: int = 8
    entity_batch_size: int = 100

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def resolved_embedding_model(self, provider: str | None = None) -> str:
        provider_name = provider or self.embedding_provider
        if provider_name == "openai":
            return self.embedding_model
        if provider_name == "ollama":
            return self.ollama_embedding_model
        raise ValueError(f"Unknown embedding provider: {provider_name}")

    def resolved_embedding_dimensions(self, provider: str | None = None) -> int:
        provider_name = provider or self.embedding_provider
        if provider_name == "openai":
            return self.embedding_dimensions
        if provider_name == "ollama":
            return self.ollama_embedding_dimensions
        raise ValueError(f"Unknown embedding provider: {provider_name}")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
