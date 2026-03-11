"""Tests for Celery worker tasks."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from constellation.indexer.pipeline import IndexingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_indexing_result(**overrides) -> IndexingResult:
    defaults = dict(
        repository="user/repo",
        files_total=10,
        files_processed=8,
        files_skipped=2,
        entities_created=25,
        relationships_created=15,
        errors=[],
    )
    defaults.update(overrides)
    return IndexingResult(**defaults)


def _make_mock_self(retries=0):
    """Create a mock Celery task self (bound task).

    The mock simulates a Celery Task instance for ``bind=True`` tasks,
    providing ``request.retries``, ``update_state``, ``retry``, and
    ``MaxRetriesExceededError``.
    """
    mock_self = MagicMock()
    mock_self.request = MagicMock()
    mock_self.request.retries = retries
    mock_self.update_state = MagicMock()
    mock_self.retry = MagicMock(side_effect=Exception("retry-sentinel"))
    mock_self.MaxRetriesExceededError = type(
        "MaxRetriesExceededError", (Exception,), {}
    )
    return mock_self


def _make_mock_lock(acquirable=True):
    """Create a mock Redis lock."""
    lock = MagicMock()
    lock.acquire = MagicMock(return_value=acquirable)
    lock.release = MagicMock()
    return lock


def _make_mock_redis(lock=None):
    """Create a mock Redis client."""
    mock_redis = MagicMock()
    if lock is None:
        lock = _make_mock_lock()
    mock_redis.lock = MagicMock(return_value=lock)
    return mock_redis


def _call_task(mock_self, **kwargs):
    """Call the index_repository task's underlying function directly.

    Because index_repository is decorated with ``@celery_app.task(bind=True)``,
    calling it normally goes through Celery's __call__ machinery. We access
    the raw function via ``__func__`` to bypass bound-method self injection,
    then pass our ``mock_self`` explicitly.
    """
    from constellation.worker.tasks import index_repository

    fn = getattr(index_repository.run, "__func__", index_repository.run)
    return fn(mock_self, **kwargs)


# ---------------------------------------------------------------------------
# Test: pipeline dispatched with correct arguments
# ---------------------------------------------------------------------------


class TestPipelineDispatch:
    @patch("constellation.worker.tasks._run_indexing", new_callable=AsyncMock)
    @patch("constellation.worker.tasks.redis")
    def test_dispatches_pipeline_with_correct_arguments(
        self,
        mock_redis_mod,
        mock_run_indexing,
    ):
        """index_repository passes source, name, exclude_patterns, reindex, and
        a progress callback to _run_indexing."""
        result = _make_indexing_result()
        mock_run_indexing.return_value = result

        mock_redis_client = _make_mock_redis()
        mock_redis_mod.from_url.return_value = mock_redis_client

        mock_self = _make_mock_self()

        _call_task(
            mock_self,
            source="/path/to/repo",
            name="my-repo",
            exclude_patterns=["*.log"],
            reindex=True,
        )

        mock_run_indexing.assert_called_once()
        call_args = mock_run_indexing.call_args
        assert call_args[0][0] == "/path/to/repo"
        assert call_args[0][1] == "my-repo"
        assert call_args[0][2] == ["*.log"]
        assert call_args[0][3] is True
        # The fifth argument should be a callable (progress_callback)
        assert callable(call_args[0][4])

    @patch("constellation.worker.tasks._run_indexing", new_callable=AsyncMock)
    @patch("constellation.worker.tasks.redis")
    def test_dispatches_with_default_arguments(
        self,
        mock_redis_mod,
        mock_run_indexing,
    ):
        """When name/exclude_patterns/reindex not provided, defaults are used."""
        result = _make_indexing_result(repository="repo")
        mock_run_indexing.return_value = result

        mock_redis_client = _make_mock_redis()
        mock_redis_mod.from_url.return_value = mock_redis_client

        mock_self = _make_mock_self()

        _call_task(mock_self, source="/path/to/repo")

        mock_run_indexing.assert_called_once()
        call_args = mock_run_indexing.call_args
        assert call_args[0][0] == "/path/to/repo"
        assert call_args[0][1] is None  # name defaults to None
        assert call_args[0][2] is None  # exclude_patterns defaults to None
        assert call_args[0][3] is False  # reindex defaults to False


# ---------------------------------------------------------------------------
# Test: result dict shape
# ---------------------------------------------------------------------------


class TestResultShape:
    @patch("constellation.worker.tasks._run_indexing", new_callable=AsyncMock)
    @patch("constellation.worker.tasks.redis")
    def test_returns_correct_result_dict(
        self,
        mock_redis_mod,
        mock_run_indexing,
    ):
        """index_repository returns a dict with all IndexingResult fields."""
        result = _make_indexing_result(
            repository="my-repo",
            files_total=10,
            files_processed=8,
            files_skipped=2,
            entities_created=25,
            relationships_created=15,
            errors=["some error"],
        )
        mock_run_indexing.return_value = result

        mock_redis_client = _make_mock_redis()
        mock_redis_mod.from_url.return_value = mock_redis_client

        mock_self = _make_mock_self()

        returned = _call_task(mock_self, source="/path/to/repo")

        assert isinstance(returned, dict)
        assert returned["repository"] == "my-repo"
        assert returned["files_total"] == 10
        assert returned["files_processed"] == 8
        assert returned["files_skipped"] == 2
        assert returned["entities_created"] == 25
        assert returned["relationships_created"] == 15
        assert returned["errors"] == ["some error"]

    @patch("constellation.worker.tasks._run_indexing", new_callable=AsyncMock)
    @patch("constellation.worker.tasks.redis")
    def test_result_has_all_expected_keys(
        self,
        mock_redis_mod,
        mock_run_indexing,
    ):
        """Result dict contains exactly the expected keys."""
        result = _make_indexing_result()
        mock_run_indexing.return_value = result

        mock_redis_client = _make_mock_redis()
        mock_redis_mod.from_url.return_value = mock_redis_client

        mock_self = _make_mock_self()

        returned = _call_task(mock_self, source="/path/to/repo")

        expected_keys = {
            "repository",
            "files_total",
            "files_processed",
            "files_skipped",
            "entities_created",
            "relationships_created",
            "errors",
        }
        assert set(returned.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Test: Redis lock acquired before indexing
# ---------------------------------------------------------------------------


class TestRedisLockAcquired:
    @patch("constellation.worker.tasks._run_indexing", new_callable=AsyncMock)
    @patch("constellation.worker.tasks.redis")
    def test_lock_acquired_before_indexing(
        self,
        mock_redis_mod,
        mock_run_indexing,
    ):
        """Redis lock is acquired before _run_indexing is called."""
        call_order = []
        lock = _make_mock_lock()
        lock.acquire = MagicMock(
            return_value=True,
            side_effect=lambda **kwargs: call_order.append("lock_acquire")
            or True,
        )
        mock_redis_client = _make_mock_redis(lock=lock)
        mock_redis_mod.from_url.return_value = mock_redis_client

        result = _make_indexing_result()

        async def track_run(*args, **kwargs):
            call_order.append("run_indexing")
            return result

        mock_run_indexing.side_effect = track_run

        mock_self = _make_mock_self()
        _call_task(mock_self, source="/path/to/repo", name="my-repo")

        assert call_order == ["lock_acquire", "run_indexing"]

    @patch("constellation.worker.tasks._run_indexing", new_callable=AsyncMock)
    @patch("constellation.worker.tasks.redis")
    def test_lock_key_contains_repo_name(
        self,
        mock_redis_mod,
        mock_run_indexing,
    ):
        """Lock key includes the repository name for per-repo locking."""
        lock = _make_mock_lock()
        mock_redis_client = _make_mock_redis(lock=lock)
        mock_redis_mod.from_url.return_value = mock_redis_client

        result = _make_indexing_result()
        mock_run_indexing.return_value = result

        mock_self = _make_mock_self()
        _call_task(mock_self, source="/path/to/repo", name="my-repo")

        mock_redis_client.lock.assert_called_once()
        lock_key = mock_redis_client.lock.call_args[0][0]
        assert "my-repo" in lock_key


# ---------------------------------------------------------------------------
# Test: Redis lock prevents concurrent indexing
# ---------------------------------------------------------------------------


class TestRedisLockConcurrency:
    @patch("constellation.worker.tasks._run_indexing", new_callable=AsyncMock)
    @patch("constellation.worker.tasks.redis")
    def test_raises_when_lock_not_acquired(
        self,
        mock_redis_mod,
        mock_run_indexing,
    ):
        """When lock.acquire returns False, raises RuntimeError."""
        lock = _make_mock_lock(acquirable=False)
        mock_redis_client = _make_mock_redis(lock=lock)
        mock_redis_mod.from_url.return_value = mock_redis_client

        mock_self = _make_mock_self()

        with pytest.raises(
            RuntimeError, match="[Ii]ndexing already in progress"
        ):
            _call_task(mock_self, source="/path/to/repo", name="my-repo")

        mock_run_indexing.assert_not_called()


# ---------------------------------------------------------------------------
# Test: Redis lock released after success
# ---------------------------------------------------------------------------


class TestLockReleasedOnSuccess:
    @patch("constellation.worker.tasks._run_indexing", new_callable=AsyncMock)
    @patch("constellation.worker.tasks.redis")
    def test_lock_released_after_success(
        self,
        mock_redis_mod,
        mock_run_indexing,
    ):
        """Lock is released after successful indexing."""
        lock = _make_mock_lock()
        mock_redis_client = _make_mock_redis(lock=lock)
        mock_redis_mod.from_url.return_value = mock_redis_client

        result = _make_indexing_result()
        mock_run_indexing.return_value = result

        mock_self = _make_mock_self()
        _call_task(mock_self, source="/path/to/repo", name="my-repo")

        lock.release.assert_called_once()


# ---------------------------------------------------------------------------
# Test: Redis lock released after failure
# ---------------------------------------------------------------------------


class TestLockReleasedOnFailure:
    @patch("constellation.worker.tasks._run_indexing", new_callable=AsyncMock)
    @patch("constellation.worker.tasks.redis")
    def test_lock_released_after_failure(
        self,
        mock_redis_mod,
        mock_run_indexing,
    ):
        """Lock is released even when indexing fails and retries are exhausted."""
        lock = _make_mock_lock()
        mock_redis_client = _make_mock_redis(lock=lock)
        mock_redis_mod.from_url.return_value = mock_redis_client

        mock_run_indexing.side_effect = RuntimeError("Pipeline crashed")

        mock_self = _make_mock_self(retries=2)
        # On max retries exceeded, retry raises MaxRetriesExceededError,
        # which causes the except branch to re-raise the original exc
        mock_self.retry = MagicMock(
            side_effect=mock_self.MaxRetriesExceededError("max retries")
        )

        with pytest.raises(RuntimeError, match="Pipeline crashed"):
            _call_task(mock_self, source="/path/to/repo", name="my-repo")

        lock.release.assert_called_once()

    @patch("constellation.worker.tasks._run_indexing", new_callable=AsyncMock)
    @patch("constellation.worker.tasks.redis")
    def test_lock_released_even_if_release_raises_lock_not_owned(
        self,
        mock_redis_mod,
        mock_run_indexing,
    ):
        """Lock release gracefully handles LockNotOwnedError."""
        import redis as real_redis

        lock = _make_mock_lock()
        lock.release = MagicMock(
            side_effect=real_redis.exceptions.LockNotOwnedError("not owned")
        )
        mock_redis_client = _make_mock_redis(lock=lock)
        mock_redis_mod.from_url.return_value = mock_redis_client
        # Expose the real exceptions module through the mock
        mock_redis_mod.exceptions = real_redis.exceptions

        result = _make_indexing_result()
        mock_run_indexing.return_value = result

        mock_self = _make_mock_self()

        # Should NOT raise despite LockNotOwnedError
        returned = _call_task(
            mock_self, source="/path/to/repo", name="my-repo"
        )
        assert returned is not None


# ---------------------------------------------------------------------------
# Test: Progress callback calls self.update_state
# ---------------------------------------------------------------------------


class TestProgressCallback:
    @patch("constellation.worker.tasks._run_indexing", new_callable=AsyncMock)
    @patch("constellation.worker.tasks.redis")
    def test_progress_callback_updates_state(
        self,
        mock_redis_mod,
        mock_run_indexing,
    ):
        """Progress callback calls self.update_state with PROGRESS state."""
        lock = _make_mock_lock()
        mock_redis_client = _make_mock_redis(lock=lock)
        mock_redis_mod.from_url.return_value = mock_redis_client

        result = _make_indexing_result()
        captured_callback = None

        async def capture_run(
            source, name, exclude_patterns, reindex, progress_callback
        ):
            nonlocal captured_callback
            captured_callback = progress_callback
            # Simulate progress updates
            progress_callback(10, 3, 5)
            progress_callback(10, 7, 12)
            return result

        mock_run_indexing.side_effect = capture_run

        mock_self = _make_mock_self()
        _call_task(mock_self, source="/path/to/repo", name="my-repo")

        assert captured_callback is not None
        assert mock_self.update_state.call_count == 2

        # First call
        first_call = mock_self.update_state.call_args_list[0]
        assert first_call[1]["state"] == "PROGRESS"
        assert first_call[1]["meta"]["files_total"] == 10
        assert first_call[1]["meta"]["files_processed"] == 3
        assert first_call[1]["meta"]["entities_found"] == 5

        # Second call
        second_call = mock_self.update_state.call_args_list[1]
        assert second_call[1]["state"] == "PROGRESS"
        assert second_call[1]["meta"]["files_total"] == 10
        assert second_call[1]["meta"]["files_processed"] == 7
        assert second_call[1]["meta"]["entities_found"] == 12


# ---------------------------------------------------------------------------
# Test: Retry on exception
# ---------------------------------------------------------------------------


class TestRetryOnException:
    @patch("constellation.worker.tasks._run_indexing", new_callable=AsyncMock)
    @patch("constellation.worker.tasks.redis")
    def test_retry_called_on_exception(
        self,
        mock_redis_mod,
        mock_run_indexing,
    ):
        """When _run_indexing raises, self.retry is called with exponential countdown."""
        lock = _make_mock_lock()
        mock_redis_client = _make_mock_redis(lock=lock)
        mock_redis_mod.from_url.return_value = mock_redis_client

        original_error = ValueError("something went wrong")
        mock_run_indexing.side_effect = original_error

        mock_self = _make_mock_self(retries=0)
        # retry raises its own exception to signal Celery to reschedule
        retry_exc = Exception("celery-retry")
        mock_self.retry = MagicMock(side_effect=retry_exc)

        with pytest.raises(Exception, match="celery-retry"):
            _call_task(mock_self, source="/path/to/repo", name="my-repo")

        mock_self.retry.assert_called_once()
        call_kwargs = mock_self.retry.call_args[1]
        assert call_kwargs["exc"] is original_error
        # countdown = 2^retries * 10, retries=0 => 2^0 * 10 = 10
        assert call_kwargs["countdown"] == 10

    @patch("constellation.worker.tasks._run_indexing", new_callable=AsyncMock)
    @patch("constellation.worker.tasks.redis")
    def test_retry_countdown_increases_with_retries(
        self,
        mock_redis_mod,
        mock_run_indexing,
    ):
        """Exponential backoff: countdown = 2^retries * 10."""
        lock = _make_mock_lock()
        mock_redis_client = _make_mock_redis(lock=lock)
        mock_redis_mod.from_url.return_value = mock_redis_client

        mock_run_indexing.side_effect = ValueError("error")

        mock_self = _make_mock_self(retries=1)
        retry_exc = Exception("celery-retry")
        mock_self.retry = MagicMock(side_effect=retry_exc)

        with pytest.raises(Exception, match="celery-retry"):
            _call_task(mock_self, source="/path/to/repo", name="my-repo")

        call_kwargs = mock_self.retry.call_args[1]
        # countdown = 2^1 * 10 = 20
        assert call_kwargs["countdown"] == 20

    @patch("constellation.worker.tasks._run_indexing", new_callable=AsyncMock)
    @patch("constellation.worker.tasks.redis")
    def test_original_exception_raised_when_max_retries_exceeded(
        self,
        mock_redis_mod,
        mock_run_indexing,
    ):
        """When max retries exceeded, the original exception is re-raised."""
        lock = _make_mock_lock()
        mock_redis_client = _make_mock_redis(lock=lock)
        mock_redis_mod.from_url.return_value = mock_redis_client

        original_error = ValueError("persistent failure")
        mock_run_indexing.side_effect = original_error

        mock_self = _make_mock_self(retries=2)
        mock_self.retry = MagicMock(
            side_effect=mock_self.MaxRetriesExceededError("max retries")
        )

        with pytest.raises(ValueError, match="persistent failure"):
            _call_task(mock_self, source="/path/to/repo", name="my-repo")


# ---------------------------------------------------------------------------
# Test: repo name derivation from source
# ---------------------------------------------------------------------------


class TestRepoNameDerivation:
    @patch("constellation.worker.tasks._run_indexing", new_callable=AsyncMock)
    @patch("constellation.worker.tasks.redis")
    def test_derives_repo_name_from_source_when_name_not_given(
        self,
        mock_redis_mod,
        mock_run_indexing,
    ):
        """When name is None, repo name is derived from source for the lock key."""
        lock = _make_mock_lock()
        mock_redis_client = _make_mock_redis(lock=lock)
        mock_redis_mod.from_url.return_value = mock_redis_client

        result = _make_indexing_result(repository="repo")
        mock_run_indexing.return_value = result

        mock_self = _make_mock_self()
        _call_task(mock_self, source="/path/to/repo")

        # Lock key should contain derived repo name
        lock_key = mock_redis_client.lock.call_args[0][0]
        assert "repo" in lock_key

    @patch("constellation.worker.tasks._run_indexing", new_callable=AsyncMock)
    @patch("constellation.worker.tasks.redis")
    def test_uses_explicit_name_for_lock_key(
        self,
        mock_redis_mod,
        mock_run_indexing,
    ):
        """When name is provided, it is used in the lock key."""
        lock = _make_mock_lock()
        mock_redis_client = _make_mock_redis(lock=lock)
        mock_redis_mod.from_url.return_value = mock_redis_client

        result = _make_indexing_result(repository="custom-name")
        mock_run_indexing.return_value = result

        mock_self = _make_mock_self()
        _call_task(mock_self, source="/path/to/repo", name="custom-name")

        lock_key = mock_redis_client.lock.call_args[0][0]
        assert "custom-name" in lock_key
