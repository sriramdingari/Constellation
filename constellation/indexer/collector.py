import hashlib
from fnmatch import fnmatch
from pathlib import Path

DEFAULT_EXCLUSIONS = frozenset({
    "node_modules", "venv", "__pycache__", "build", "dist", "target",
    ".git", "**/generated-sources/**", "**/migrations/**", ".venv",
    "*.min.js", "*.min.css",
})


def is_github_url(source: str) -> bool:
    return source.startswith("https://github.com/") or source.startswith("git@github.com:")


def derive_repo_name(source: str) -> str:
    if is_github_url(source):
        path = source.replace("https://github.com/", "").replace("git@github.com:", "")
        path = path.rstrip("/").removesuffix(".git")
        return path
    return Path(source).name


def is_excluded(relative_path: Path, exclusion_patterns: frozenset[str]) -> bool:
    path_str = str(relative_path)
    parts = relative_path.parts
    for pattern in exclusion_patterns:
        if "**" in pattern or "*" in pattern:
            if fnmatch(path_str, pattern):
                return True
        else:
            if pattern in parts:
                return True
    return False


def compute_file_hash(file_path: Path) -> str:
    h = hashlib.md5()
    h.update(file_path.read_bytes())
    return h.hexdigest()


def collect_files(
    root: Path,
    supported_extensions: set[str],
    exclude_patterns: frozenset[str] | None = None,
) -> list[Path]:
    if exclude_patterns is None:
        exclude_patterns = DEFAULT_EXCLUSIONS
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in supported_extensions:
            continue
        relative = path.relative_to(root)
        if is_excluded(relative, exclude_patterns):
            continue
        files.append(path)
    return files
