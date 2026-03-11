"""Tests for API routes."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from constellation.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_redis():
    with patch("constellation.api.routes.redis") as mock_r:
        mock_instance = MagicMock()
        mock_r.from_url.return_value = mock_instance
        mock_instance.exists.return_value = False
        mock_instance.ping.return_value = True
        yield mock_instance


@pytest.fixture
def mock_task():
    with patch("constellation.api.routes.index_repository") as mock_t:
        mock_result = MagicMock()
        mock_result.id = "task-abc-123"
        mock_t.delay.return_value = mock_result
        yield mock_t


@pytest.fixture
def mock_graph_client():
    mock_client = AsyncMock()
    mock_client.connect = AsyncMock()
    mock_client.close = AsyncMock()
    mock_client.get_repository = AsyncMock(return_value=None)
    mock_client.list_repositories = AsyncMock(return_value=[])
    mock_client.delete_repository = AsyncMock()
    with patch(
        "constellation.api.routes.GraphClient", return_value=mock_client
    ):
        yield mock_client


class TestIndexRepo:
    def test_dispatches_task_returns_202(self, client, mock_redis, mock_task):
        response = client.post(
            "/repositories/index",
            json={"source": "/path/to/repo"},
        )
        assert response.status_code == 202
        data = response.json()
        assert data["job_id"] == "task-abc-123"
        assert data["status"] == "queued"
        mock_task.delay.assert_called_once_with(
            "/path/to/repo", None, None, False
        )

    def test_github_url_derives_name(self, client, mock_redis, mock_task):
        response = client.post(
            "/repositories/index",
            json={"source": "https://github.com/owner/repo"},
        )
        assert response.status_code == 202
        data = response.json()
        assert data["repository"] == "owner/repo"

    def test_explicit_name_used(self, client, mock_redis, mock_task):
        response = client.post(
            "/repositories/index",
            json={"source": "/path/to/repo", "name": "custom-name"},
        )
        assert response.status_code == 202
        data = response.json()
        assert data["repository"] == "custom-name"

    def test_409_when_lock_exists(self, client, mock_redis, mock_task):
        mock_redis.exists.return_value = True
        response = client.post(
            "/repositories/index",
            json={"source": "/path/to/repo"},
        )
        assert response.status_code == 409
        assert "already in progress" in response.json()["detail"]


class TestListRepositories:
    def test_returns_list(self, client, mock_graph_client):
        mock_graph_client.list_repositories.return_value = [
            {
                "name": "repo-a",
                "source": "/path/a",
                "last_indexed_at": "2025-01-01T00:00:00Z",
                "last_commit_sha": "abc",
                "entity_count": 10,
            },
            {
                "name": "repo-b",
                "source": "/path/b",
                "last_indexed_at": None,
                "last_commit_sha": None,
                "entity_count": 0,
            },
        ]
        response = client.get("/repositories")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["name"] == "repo-a"
        assert data[1]["name"] == "repo-b"


class TestGetRepository:
    def test_returns_repo_info(self, client, mock_graph_client):
        mock_graph_client.get_repository.return_value = {
            "name": "my-repo",
            "source": "/path/to/repo",
            "last_indexed_at": "2025-01-01T00:00:00Z",
            "last_commit_sha": "abc123",
            "entity_count": 42,
        }
        response = client.get("/repositories/my-repo")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "my-repo"
        assert data["entity_count"] == 42

    def test_404_when_not_found(self, client, mock_graph_client):
        mock_graph_client.get_repository.return_value = None
        response = client.get("/repositories/nonexistent")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestDeleteRepository:
    def test_returns_204(self, client, mock_graph_client):
        mock_graph_client.get_repository.return_value = {
            "name": "my-repo",
            "source": "/path/to/repo",
        }
        response = client.delete("/repositories/my-repo")
        assert response.status_code == 204
        mock_graph_client.delete_repository.assert_called_once_with("my-repo")

    def test_404_when_not_found(self, client, mock_graph_client):
        mock_graph_client.get_repository.return_value = None
        response = client.delete("/repositories/nonexistent")
        assert response.status_code == 404


class TestGetJobStatus:
    def _mock_async_result(self, state, result=None, info=None):
        mock_result = MagicMock()
        mock_result.state = state
        mock_result.result = result
        mock_result.info = info
        return mock_result

    def test_pending_returns_queued(self, client):
        with patch("constellation.api.routes.AsyncResult") as mock_ar:
            mock_ar.return_value = self._mock_async_result("PENDING")
            response = client.get("/jobs/some-id")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "some-id"
        assert data["status"] == "queued"

    def test_started_returns_in_progress(self, client):
        with patch("constellation.api.routes.AsyncResult") as mock_ar:
            mock_ar.return_value = self._mock_async_result("STARTED")
            response = client.get("/jobs/some-id")
        assert response.status_code == 200
        assert response.json()["status"] == "in_progress"

    def test_progress_state_with_data(self, client):
        with patch("constellation.api.routes.AsyncResult") as mock_ar:
            mock_ar.return_value = self._mock_async_result(
                "PROGRESS",
                info={
                    "files_total": 100,
                    "files_processed": 50,
                    "entities_found": 200,
                },
            )
            response = client.get("/jobs/some-id")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "in_progress"
        assert data["progress"]["files_total"] == 100
        assert data["progress"]["files_processed"] == 50
        assert data["progress"]["entities_found"] == 200

    def test_success_returns_completed(self, client):
        with patch("constellation.api.routes.AsyncResult") as mock_ar:
            mock_ar.return_value = self._mock_async_result(
                "SUCCESS",
                result={"entities_created": 42, "files_processed": 10},
            )
            response = client.get("/jobs/some-id")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["result"] == {"entities_created": 42, "files_processed": 10}

    def test_failure_returns_failed(self, client):
        with patch("constellation.api.routes.AsyncResult") as mock_ar:
            mock_result = self._mock_async_result("FAILURE")
            mock_result.result = Exception("Connection refused")
            mock_ar.return_value = mock_result
            response = client.get("/jobs/some-id")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert "Connection refused" in data["error"]


class TestHealth:
    def test_returns_health_status(self, client, mock_graph_client):
        with patch("constellation.api.routes.redis") as mock_redis_mod:
            mock_redis_instance = MagicMock()
            mock_redis_mod.from_url.return_value = mock_redis_instance
            mock_redis_instance.ping.return_value = True
            response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["neo4j"] == "connected"
        assert data["redis"] == "connected"
