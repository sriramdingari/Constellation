import os
from unittest.mock import patch

from constellation.config import Settings, get_settings


class TestSettings:
    def test_default_neo4j_uri(self):
        s = Settings()
        assert s.neo4j_uri == "bolt://localhost:7687"

    def test_default_neo4j_user(self):
        s = Settings()
        assert s.neo4j_user == "neo4j"

    def test_default_neo4j_password(self):
        s = Settings()
        assert s.neo4j_password == "constellation"

    def test_default_redis_url(self):
        s = Settings()
        assert s.redis_url == "redis://localhost:6379"

    def test_default_embedding_provider(self):
        s = Settings()
        assert s.embedding_provider == "openai"

    def test_default_embedding_model(self):
        s = Settings()
        assert s.embedding_model == "text-embedding-3-small"

    def test_default_embedding_dimensions(self):
        s = Settings()
        assert s.embedding_dimensions == 1536

    def test_default_ollama_base_url(self):
        s = Settings()
        assert s.ollama_base_url == "http://localhost:11434"

    def test_default_embedding_batch_size(self):
        s = Settings()
        assert s.embedding_batch_size == 8

    def test_default_entity_batch_size(self):
        s = Settings()
        assert s.entity_batch_size == 100

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_openai_api_key_defaults_empty(self):
        s = Settings(_env_file=None)
        assert s.openai_api_key == ""

    @patch.dict(os.environ, {"NEO4J_URI": "bolt://custom:7687"})
    def test_override_from_env(self):
        s = Settings()
        assert s.neo4j_uri == "bolt://custom:7687"

    @patch.dict(os.environ, {"EMBEDDING_PROVIDER": "ollama"})
    def test_override_embedding_provider(self):
        s = Settings()
        assert s.embedding_provider == "ollama"


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
