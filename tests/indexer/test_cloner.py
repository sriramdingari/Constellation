import base64
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from constellation.indexer.cloner import clone_repository, get_commit_sha, cleanup_clone


class TestCloneRepository:
    @patch("constellation.indexer.cloner.tempfile.mkdtemp", return_value="/tmp/constellation_abc123")
    @patch("constellation.indexer.cloner.subprocess.run")
    def test_clone_calls_git_clone_depth_1(self, mock_run, mock_mkdtemp):
        result = clone_repository("https://github.com/user/repo")

        mock_run.assert_called_once_with(
            ["git", "clone", "--depth", "1", "https://github.com/user/repo", "/tmp/constellation_abc123"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert result == Path("/tmp/constellation_abc123")

    @patch("constellation.indexer.cloner.tempfile.mkdtemp", return_value="/tmp/constellation_abc123")
    @patch("constellation.indexer.cloner.subprocess.run")
    def test_clone_returns_path(self, mock_run, mock_mkdtemp):
        result = clone_repository("https://github.com/user/repo")
        assert isinstance(result, Path)
        assert str(result) == "/tmp/constellation_abc123"

    @patch("constellation.indexer.cloner.shutil.rmtree")
    @patch("constellation.indexer.cloner.tempfile.mkdtemp", return_value="/tmp/constellation_abc123")
    @patch("constellation.indexer.cloner.subprocess.run", side_effect=subprocess.CalledProcessError(128, "git"))
    def test_clone_raises_on_invalid_url(self, mock_run, mock_mkdtemp, mock_rmtree):
        with pytest.raises(subprocess.CalledProcessError):
            clone_repository("https://invalid-url/not-a-repo")

        mock_rmtree.assert_called_once_with("/tmp/constellation_abc123", ignore_errors=True)

    @patch("constellation.indexer.cloner.tempfile.mkdtemp", return_value="/tmp/constellation_abc123")
    @patch("constellation.indexer.cloner.subprocess.run")
    def test_clone_uses_git_config_env_for_github_token(self, mock_run, mock_mkdtemp):
        clone_repository(
            "https://github.com/user/repo",
            github_token="ghp_test_token",
        )

        args, kwargs = mock_run.call_args
        assert args[0] == [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/user/repo",
            "/tmp/constellation_abc123",
        ]
        assert kwargs["check"] is True
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True

        env = kwargs["env"]
        expected = base64.b64encode(
            b"x-access-token:ghp_test_token"
        ).decode()
        assert env["GIT_CONFIG_COUNT"] == "1"
        assert env["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraHeader"
        assert env["GIT_CONFIG_VALUE_0"] == f"Authorization: Basic {expected}"
        assert "ghp_test_token" not in " ".join(args[0])

    @patch("constellation.indexer.cloner.tempfile.mkdtemp", return_value="/tmp/constellation_abc123")
    @patch("constellation.indexer.cloner.subprocess.run")
    def test_clone_does_not_use_token_for_non_github_https_url(self, mock_run, mock_mkdtemp):
        clone_repository(
            "https://gitlab.com/user/repo",
            github_token="ghp_test_token",
        )

        _, kwargs = mock_run.call_args
        assert "env" not in kwargs

    @patch("constellation.indexer.cloner.tempfile.mkdtemp", return_value="/tmp/constellation_abc123")
    @patch("constellation.indexer.cloner.subprocess.run")
    def test_clone_normalizes_github_ssh_url_when_token_present(self, mock_run, mock_mkdtemp):
        clone_repository(
            "git@github.com:user/repo.git",
            github_token="ghp_test_token",
        )

        args, kwargs = mock_run.call_args
        assert args[0] == [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/user/repo.git",
            "/tmp/constellation_abc123",
        ]
        assert "env" in kwargs

    @patch("constellation.indexer.cloner.shutil.rmtree")
    @patch("constellation.indexer.cloner.tempfile.mkdtemp", return_value="/tmp/constellation_abc123")
    @patch("constellation.indexer.cloner.subprocess.run", side_effect=subprocess.CalledProcessError(128, "git"))
    def test_authenticated_clone_cleans_up_on_failure(self, mock_run, mock_mkdtemp, mock_rmtree):
        with pytest.raises(subprocess.CalledProcessError):
            clone_repository(
                "https://github.com/user/private-repo",
                github_token="ghp_test_token",
            )

        mock_rmtree.assert_called_once_with("/tmp/constellation_abc123", ignore_errors=True)


class TestGetCommitSha:
    @patch("constellation.indexer.cloner.subprocess.run")
    def test_returns_sha_for_git_repo(self, mock_run):
        mock_run.return_value = MagicMock(stdout="abc123def456\n")
        result = get_commit_sha(Path("/some/repo"))

        mock_run.assert_called_once_with(
            ["git", "rev-parse", "HEAD"],
            cwd="/some/repo",
            check=True,
            capture_output=True,
            text=True,
        )
        assert result == "abc123def456"

    @patch("constellation.indexer.cloner.subprocess.run", side_effect=subprocess.CalledProcessError(128, "git"))
    def test_returns_none_for_non_git_directory(self, mock_run):
        result = get_commit_sha(Path("/not/a/repo"))
        assert result is None

    @patch("constellation.indexer.cloner.subprocess.run", side_effect=FileNotFoundError)
    def test_returns_none_when_git_not_found(self, mock_run):
        result = get_commit_sha(Path("/some/path"))
        assert result is None


class TestCleanupClone:
    @patch("constellation.indexer.cloner.shutil.rmtree")
    def test_removes_directory(self, mock_rmtree):
        cleanup_clone(Path("/tmp/constellation_abc123"))
        mock_rmtree.assert_called_once_with("/tmp/constellation_abc123", ignore_errors=True)

    @patch("constellation.indexer.cloner.shutil.rmtree")
    def test_no_error_if_already_removed(self, mock_rmtree):
        # ignore_errors=True means no exception even if path is gone
        cleanup_clone(Path("/tmp/already_gone"))
        mock_rmtree.assert_called_once_with("/tmp/already_gone", ignore_errors=True)
