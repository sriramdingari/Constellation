import shutil
import subprocess
import tempfile
from pathlib import Path


def clone_repository(url: str) -> Path:
    """Shallow clone a git repository and return path to the clone."""
    tmp_dir = tempfile.mkdtemp(prefix="constellation_")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, tmp_dir],
            check=True,
            capture_output=True,
            text=True,
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
