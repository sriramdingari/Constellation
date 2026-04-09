import base64
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _normalize_github_url(url: str, github_token: str | None) -> str:
    if github_token and url.startswith("git@github.com:"):
        return "https://github.com/" + url.removeprefix("git@github.com:")
    return url


def _github_auth_env(github_token: str) -> dict[str, str]:
    encoded = base64.b64encode(
        f"x-access-token:{github_token}".encode()
    ).decode()
    return {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "http.https://github.com/.extraHeader",
        "GIT_CONFIG_VALUE_0": f"Authorization: Basic {encoded}",
    }


def clone_repository(url: str, github_token: str | None = None) -> Path:
    """Shallow clone a git repository and return path to the clone."""
    tmp_dir = tempfile.mkdtemp(prefix="constellation_")
    clone_url = _normalize_github_url(url, github_token)
    run_kwargs: dict[str, object] = {
        "check": True,
        "capture_output": True,
        "text": True,
    }
    if github_token and clone_url.startswith("https://github.com/"):
        run_kwargs["env"] = {
            **os.environ,
            **_github_auth_env(github_token),
        }

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", clone_url, tmp_dir],
            **run_kwargs,
        )
    except subprocess.CalledProcessError:
        # Clean up temp dir on failure
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return Path(tmp_dir)


def get_commit_sha(repo_path: Path) -> str | None:
    """Get HEAD commit SHA for a git repo, or None if not a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def cleanup_clone(path: Path) -> None:
    """Remove a cloned repository directory. No error if already gone."""
    shutil.rmtree(str(path), ignore_errors=True)
