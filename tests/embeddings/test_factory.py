import sys
from unittest.mock import MagicMock, patch

import pytest

from constellation.embeddings.factory import create_embedding_provider


def _make_settings(**overrides):
    settings = MagicMock()
    settings.embedding_model = overrides.get("embedding_model", "text-embedding-3-small")
    settings.embedding_dimensions = overrides.get("embedding_dimensions", 1536)
    settings.openai_api_key = overrides.get("openai_api_key", "sk-test-key")
    settings.openai_base_url = overrides.get("openai_base_url", "")
    settings.ollama_base_url = overrides.get("ollama_base_url", "http://localhost:11434")
    settings.ollama_embedding_model = overrides.get(
        "ollama_embedding_model", "nomic-embed-text"
    )
    settings.ollama_embedding_dimensions = overrides.get(
        "ollama_embedding_dimensions", 768
    )

    def _resolved_embedding_model(provider):
        if provider == "ollama":
            return settings.ollama_embedding_model
        return settings.embedding_model

    def _resolved_embedding_dimensions(provider):
        if provider == "ollama":
            return settings.ollama_embedding_dimensions
        return settings.embedding_dimensions

    settings.resolved_embedding_model = MagicMock(
        side_effect=_resolved_embedding_model
    )
    settings.resolved_embedding_dimensions = MagicMock(
        side_effect=_resolved_embedding_dimensions
    )
    return settings


@pytest.fixture()
def mock_openai_provider():
    mock_cls = MagicMock()
    mock_module = MagicMock()
    mock_module.OpenAIEmbeddingProvider = mock_cls
    with patch.dict(sys.modules, {"constellation.embeddings.openai": mock_module}):
        yield mock_cls


@pytest.fixture()
def mock_ollama_provider():
    mock_cls = MagicMock()
    mock_module = MagicMock()
    mock_module.OllamaEmbeddingProvider = mock_cls
    with patch.dict(sys.modules, {"constellation.embeddings.ollama": mock_module}):
        yield mock_cls


class TestCreateEmbeddingProviderOpenAI:
    def test_returns_openai_provider_instance(self, mock_openai_provider):
        mock_instance = MagicMock()
        mock_openai_provider.return_value = mock_instance

        result = create_embedding_provider("openai", _make_settings())

        assert result is mock_instance

    def test_passes_api_key_from_settings(self, mock_openai_provider):
        settings = _make_settings(openai_api_key="sk-custom-key")

        create_embedding_provider("openai", settings)

        mock_openai_provider.assert_called_once_with(
            api_key="sk-custom-key",
            model="text-embedding-3-small",
            dimensions=1536,
            base_url=None,
        )

    def test_passes_model_from_settings(self, mock_openai_provider):
        settings = _make_settings(embedding_model="text-embedding-ada-002")

        create_embedding_provider("openai", settings)

        _, kwargs = mock_openai_provider.call_args
        assert kwargs["model"] == "text-embedding-ada-002"

    def test_passes_dimensions_from_settings(self, mock_openai_provider):
        settings = _make_settings(embedding_dimensions=768)

        create_embedding_provider("openai", settings)

        _, kwargs = mock_openai_provider.call_args
        assert kwargs["dimensions"] == 768


class TestCreateEmbeddingProviderOllama:
    def test_returns_ollama_provider_instance(self, mock_ollama_provider):
        mock_instance = MagicMock()
        mock_ollama_provider.return_value = mock_instance

        result = create_embedding_provider("ollama", _make_settings())

        assert result is mock_instance

    def test_passes_base_url_from_settings(self, mock_ollama_provider):
        settings = _make_settings(ollama_base_url="http://custom:11434")

        create_embedding_provider("ollama", settings)

        mock_ollama_provider.assert_called_once_with(
            base_url="http://custom:11434",
            model="nomic-embed-text",
            dimensions=768,
        )

    def test_passes_model_from_settings(self, mock_ollama_provider):
        settings = _make_settings(ollama_embedding_model="mxbai-embed-large")

        create_embedding_provider("ollama", settings)

        _, kwargs = mock_ollama_provider.call_args
        assert kwargs["model"] == "mxbai-embed-large"

    def test_passes_dimensions_from_settings(self, mock_ollama_provider):
        settings = _make_settings(ollama_embedding_dimensions=384)

        create_embedding_provider("ollama", settings)

        _, kwargs = mock_ollama_provider.call_args
        assert kwargs["dimensions"] == 384


class TestCreateEmbeddingProviderUnknown:
    def test_raises_value_error_for_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown embedding provider: unknown"):
            create_embedding_provider("unknown", _make_settings())

    def test_raises_value_error_with_provider_name_in_message(self):
        with pytest.raises(ValueError, match="some_weird_provider"):
            create_embedding_provider("some_weird_provider", _make_settings())
