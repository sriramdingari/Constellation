import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from openai import APIConnectionError, APIStatusError
from tenacity import wait_none

from constellation.embeddings.openai import OpenAIEmbeddingProvider


def _make_embedding_response(embeddings: list[list[float]]):
    """Build a mock response matching the OpenAI embeddings API shape."""
    data = []
    for i, emb in enumerate(embeddings):
        item = MagicMock()
        item.embedding = emb
        item.index = i
        data.append(item)
    response = MagicMock()
    response.data = data
    return response


@pytest.fixture(autouse=True)
def _disable_retry_wait():
    """Disable tenacity wait for all tests so retries don't block."""
    original_wait = OpenAIEmbeddingProvider.embed_batch.retry.wait
    OpenAIEmbeddingProvider.embed_batch.retry.wait = wait_none()
    yield
    OpenAIEmbeddingProvider.embed_batch.retry.wait = original_wait


class TestModelNameProperty:
    def test_returns_configured_model(self):
        provider = OpenAIEmbeddingProvider(api_key="test-key", model="text-embedding-3-large")
        assert provider.model_name == "text-embedding-3-large"

    def test_returns_default_model(self):
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        assert provider.model_name == "text-embedding-3-small"


class TestDimensionsProperty:
    def test_returns_configured_dimensions(self):
        provider = OpenAIEmbeddingProvider(api_key="test-key", dimensions=768)
        assert provider.dimensions == 768

    def test_returns_default_dimensions(self):
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        assert provider.dimensions == 1536


class TestEmbedBatch:
    @pytest.mark.asyncio
    async def test_returns_embeddings_from_api(self):
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        expected_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_response = _make_embedding_response(expected_embeddings)

        provider._client = AsyncMock()
        provider._client.embeddings.create = AsyncMock(return_value=mock_response)

        result = await provider.embed_batch(["hello", "world"])

        assert result == expected_embeddings

    @pytest.mark.asyncio
    async def test_calls_api_with_correct_parameters(self):
        provider = OpenAIEmbeddingProvider(
            api_key="test-key", model="text-embedding-3-large", dimensions=3072
        )
        mock_response = _make_embedding_response([[0.1] * 3072])

        provider._client = AsyncMock()
        provider._client.embeddings.create = AsyncMock(return_value=mock_response)

        await provider.embed_batch(["test text"])

        provider._client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-large",
            input=["test text"],
            dimensions=3072,
        )

    @pytest.mark.asyncio
    async def test_empty_list_returns_empty(self):
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        # Should not call the API at all
        provider._client = AsyncMock()

        result = await provider.embed_batch([])

        assert result == []
        provider._client.embeddings.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_correct_return_shape_for_multiple_inputs(self):
        provider = OpenAIEmbeddingProvider(api_key="test-key", dimensions=4)
        texts = ["alpha", "beta", "gamma"]
        embeddings = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]]
        mock_response = _make_embedding_response(embeddings)

        provider._client = AsyncMock()
        provider._client.embeddings.create = AsyncMock(return_value=mock_response)

        result = await provider.embed_batch(texts)

        assert len(result) == 3
        assert all(len(emb) == 4 for emb in result)


class TestRetryBehavior:
    @pytest.mark.asyncio
    async def test_retries_on_transient_error(self):
        provider = OpenAIEmbeddingProvider(api_key="test-key")

        success_response = _make_embedding_response([[0.1, 0.2]])

        mock_request = MagicMock()
        transient_error = APIConnectionError(request=mock_request)

        provider._client = AsyncMock()
        provider._client.embeddings.create = AsyncMock(
            side_effect=[transient_error, success_response]
        )

        result = await provider.embed_batch(["retry me"])

        assert result == [[0.1, 0.2]]
        assert provider._client.embeddings.create.call_count == 2


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_raises_after_all_retries_exhausted(self):
        provider = OpenAIEmbeddingProvider(api_key="test-key")

        mock_request = MagicMock()
        persistent_error = APIConnectionError(request=mock_request)

        provider._client = AsyncMock()
        provider._client.embeddings.create = AsyncMock(side_effect=persistent_error)

        with pytest.raises(APIConnectionError):
            await provider.embed_batch(["fail"])

        assert provider._client.embeddings.create.call_count == 3
