import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from constellation.embeddings.ollama import OllamaEmbeddingProvider


class TestModelNameProperty:
    def test_returns_configured_model(self):
        provider = OllamaEmbeddingProvider(model="mxbai-embed-large")
        assert provider.model_name == "mxbai-embed-large"

    def test_returns_default_model(self):
        provider = OllamaEmbeddingProvider()
        assert provider.model_name == "nomic-embed-text"


class TestDimensionsProperty:
    def test_returns_configured_dimensions(self):
        provider = OllamaEmbeddingProvider(dimensions=384)
        assert provider.dimensions == 384

    def test_returns_default_dimensions(self):
        provider = OllamaEmbeddingProvider()
        assert provider.dimensions == 768


class TestEmbedBatch:
    @pytest.mark.asyncio
    async def test_empty_list_returns_empty(self):
        provider = OllamaEmbeddingProvider()
        result = await provider.embed_batch([])
        assert result == []

    @pytest.mark.asyncio
    @patch("constellation.embeddings.ollama.httpx.AsyncClient")
    async def test_returns_embeddings_from_api(self, mock_async_client_cls):
        provider = OllamaEmbeddingProvider()

        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        mock_async_client_cls.return_value.__aenter__.return_value = mock_client

        result = await provider.embed_batch(["hello"])

        assert result == [[0.1, 0.2, 0.3]]

    @pytest.mark.asyncio
    @patch("constellation.embeddings.ollama.httpx.AsyncClient")
    async def test_batch_sent_in_single_request(self, mock_async_client_cls):
        """All texts go in one POST to /api/embed, not one per text."""
        provider = OllamaEmbeddingProvider(model="nomic-embed-text")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        mock_async_client_cls.return_value.__aenter__.return_value = mock_client

        result = await provider.embed_batch(["alpha", "beta"])

        assert mock_client.post.call_count == 1
        assert result == [[0.1, 0.2], [0.3, 0.4]]

    @pytest.mark.asyncio
    @patch("constellation.embeddings.ollama.httpx.AsyncClient")
    async def test_sends_correct_request_body(self, mock_async_client_cls):
        provider = OllamaEmbeddingProvider(
            base_url="http://myhost:11434", model="nomic-embed-text"
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.5]]}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        mock_async_client_cls.return_value.__aenter__.return_value = mock_client

        await provider.embed_batch(["test text"])

        mock_client.post.assert_called_once_with(
            "http://myhost:11434/api/embed",
            json={"model": "nomic-embed-text", "input": ["test text"]},
        )

    @pytest.mark.asyncio
    @patch("constellation.embeddings.ollama.httpx.AsyncClient")
    async def test_correct_return_shape_for_multiple_inputs(self, mock_async_client_cls):
        provider = OllamaEmbeddingProvider(dimensions=4)
        texts = ["alpha", "beta", "gamma"]

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "embeddings": [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 1.0, 1.1, 1.2],
            ],
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        mock_async_client_cls.return_value.__aenter__.return_value = mock_client

        result = await provider.embed_batch(texts)

        assert len(result) == 3
        assert all(len(emb) == 4 for emb in result)
        assert mock_client.post.call_count == 1
