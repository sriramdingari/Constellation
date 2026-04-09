import os
from unittest.mock import patch

import pytest

from constellation.config import Settings, get_settings


def _settings_without_env_file(**overrides):
    return Settings(_env_file=None, **overrides)


class TestSettings:
    def test_default_neo4j_uri(self):
        s = _settings_without_env_file()
        assert s.neo4j_uri == "bolt://localhost:7687"

    def test_default_neo4j_user(self):
        s = _settings_without_env_file()
        assert s.neo4j_user == "neo4j"

    def test_default_neo4j_password(self):
        s = _settings_without_env_file()
        assert s.neo4j_password == "constellation"

    def test_default_redis_url(self):
        s = _settings_without_env_file()
        assert s.redis_url == "redis://localhost:6379"

    def test_default_embedding_provider(self):
        s = _settings_without_env_file()
        assert s.embedding_provider == "openai"

    def test_default_embedding_model(self):
        s = _settings_without_env_file()
        assert s.embedding_model == "text-embedding-3-small"

    def test_default_embedding_dimensions(self):
        s = _settings_without_env_file()
        assert s.embedding_dimensions == 1536

    def test_default_ollama_base_url(self):
        s = _settings_without_env_file()
        assert s.ollama_base_url == "http://localhost:11434"

    def test_default_ollama_embedding_model(self):
        s = _settings_without_env_file()
        assert s.ollama_embedding_model == "nomic-embed-text"

    def test_default_ollama_embedding_dimensions(self):
        s = _settings_without_env_file()
        assert s.ollama_embedding_dimensions == 768

    def test_default_embedding_batch_size(self):
        s = _settings_without_env_file()
        assert s.embedding_batch_size == 8

    def test_default_entity_batch_size(self):
        s = _settings_without_env_file()
        assert s.entity_batch_size == 100

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_openai_api_key_defaults_empty(self):
        s = _settings_without_env_file()
        assert s.openai_api_key == ""

    def test_default_github_token(self):
        s = _settings_without_env_file()
        assert s.github_token == ""

    @patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_test_token"}, clear=False)
    def test_github_token_override_from_env(self):
        s = _settings_without_env_file()
        assert s.github_token == "ghp_test_token"

    @patch.dict(os.environ, {"NEO4J_URI": "bolt://custom:7687"})
    def test_override_from_env(self):
        s = _settings_without_env_file()
        assert s.neo4j_uri == "bolt://custom:7687"

    @patch.dict(os.environ, {"EMBEDDING_PROVIDER": "ollama"})
    def test_override_embedding_provider(self):
        s = _settings_without_env_file()
        assert s.embedding_provider == "ollama"

    def test_resolved_openai_embedding_settings(self):
        s = _settings_without_env_file()
        assert s.resolved_embedding_model() == "text-embedding-3-small"
        assert s.resolved_embedding_dimensions() == 1536

    @patch.dict(os.environ, {"EMBEDDING_PROVIDER": "ollama"}, clear=False)
    def test_resolved_ollama_embedding_settings(self):
        s = _settings_without_env_file()
        assert s.resolved_embedding_model() == "nomic-embed-text"
        assert s.resolved_embedding_dimensions() == 768

    @patch.dict(
        os.environ,
        {
            "EMBEDDING_PROVIDER": "ollama",
            "OLLAMA_EMBEDDING_MODEL": "mxbai-embed-large",
            "OLLAMA_EMBEDDING_DIMENSIONS": "1024",
        },
        clear=False,
    )
    def test_ollama_overrides_from_env(self):
        s = _settings_without_env_file()
        assert s.ollama_embedding_model == "mxbai-embed-large"
        assert s.ollama_embedding_dimensions == 1024
        assert s.resolved_embedding_model() == "mxbai-embed-large"
        assert s.resolved_embedding_dimensions() == 1024


class TestStorageBackend:
    def test_settings_default_storage_backend(self):
        s = _settings_without_env_file()
        assert s.storage_backend == "neo4j"
        assert s.postgres_dsn == ""

    def test_settings_postgres_backend(self):
        s = _settings_without_env_file(
            storage_backend="postgres",
            postgres_dsn="postgresql://user:pass@localhost/db",
        )
        assert s.storage_backend == "postgres"
        assert s.postgres_dsn == "postgresql://user:pass@localhost/db"

    def test_settings_invalid_backend_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            _settings_without_env_file(storage_backend="sqlite")

    def test_settings_postgres_backend_requires_non_empty_dsn(self):
        """storage_backend=postgres with empty postgres_dsn must raise."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="postgres_dsn"):
            _settings_without_env_file(storage_backend="postgres", postgres_dsn="")

    def test_settings_postgres_backend_accepts_non_empty_dsn(self):
        """storage_backend=postgres with a real DSN must construct cleanly."""
        s = _settings_without_env_file(
            storage_backend="postgres",
            postgres_dsn="postgresql://user:pass@localhost:5432/db",
        )
        assert s.storage_backend == "postgres"
        assert s.postgres_dsn == "postgresql://user:pass@localhost:5432/db"

    def test_settings_neo4j_backend_does_not_require_dsn(self):
        """storage_backend=neo4j must not require postgres_dsn."""
        s = _settings_without_env_file(storage_backend="neo4j", postgres_dsn="")
        assert s.storage_backend == "neo4j"


class TestGetSettings:
    def test_returns_settings_instance(self):
        import constellation.config
        constellation.config._settings = None
        s = get_settings()
        assert isinstance(s, Settings)

    def test_caches_instance(self):
        import constellation.config
        constellation.config._settings = None
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
