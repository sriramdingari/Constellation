import pytest
from pathlib import Path

from constellation.indexer.collector import (
    is_github_url,
    derive_repo_name,
    is_excluded,
    compute_file_hash,
    collect_files,
    DEFAULT_EXCLUSIONS,
)


class TestIsGithubUrl:
    def test_https_github_url(self):
        assert is_github_url("https://github.com/user/repo") is True

    def test_https_github_url_with_git_suffix(self):
        assert is_github_url("https://github.com/user/repo.git") is True

    def test_ssh_github_url(self):
        assert is_github_url("git@github.com:user/repo") is True

    def test_local_absolute_path(self):
        assert is_github_url("/local/path") is False

    def test_relative_path(self):
        assert is_github_url("./relative") is False

    def test_gitlab_url(self):
        assert is_github_url("https://gitlab.com/user/repo") is False


class TestDeriveRepoName:
    def test_local_path(self):
        assert derive_repo_name("/home/dev/my-project") == "my-project"

    def test_github_https_url(self):
        assert derive_repo_name("https://github.com/user/repo") == "user/repo"

    def test_github_url_with_git_suffix(self):
        assert derive_repo_name("https://github.com/user/repo.git") == "user/repo"

    def test_github_url_with_trailing_slash(self):
        assert derive_repo_name("https://github.com/user/repo/") == "user/repo"


class TestIsExcluded:
    def test_node_modules(self):
        assert is_excluded(Path("node_modules/foo.js"), DEFAULT_EXCLUSIONS) is True

    def test_venv(self):
        assert is_excluded(Path("venv/lib/site.py"), DEFAULT_EXCLUSIONS) is True

    def test_dot_git(self):
        assert is_excluded(Path(".git/config"), DEFAULT_EXCLUSIONS) is True

    def test_pycache(self):
        assert is_excluded(Path("__pycache__/foo.pyc"), DEFAULT_EXCLUSIONS) is True

    def test_build(self):
        assert is_excluded(Path("build/output.jar"), DEFAULT_EXCLUSIONS) is True

    def test_allowed_source_file(self):
        assert is_excluded(Path("src/main.py"), DEFAULT_EXCLUSIONS) is False

    def test_custom_glob_pattern(self):
        patterns = frozenset({"**/test/**"})
        assert is_excluded(Path("src/test/foo.java"), patterns) is True

    def test_min_js_glob(self):
        assert is_excluded(Path("app.min.js"), DEFAULT_EXCLUSIONS) is True


class TestComputeFileHash:
    def test_returns_32_char_hex(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("hello world")
        result = compute_file_hash(f)
        assert len(result) == 32
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic(self, tmp_path):
        f = tmp_path / "same.txt"
        f.write_text("deterministic content")
        assert compute_file_hash(f) == compute_file_hash(f)

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("content A")
        f2.write_text("content B")
        assert compute_file_hash(f1) != compute_file_hash(f2)


class TestCollectFiles:
    def test_finds_supported_extensions(self, tmp_path):
        (tmp_path / "main.py").write_text("print('hi')")
        (tmp_path / "App.java").write_text("class App {}")
        result = collect_files(tmp_path, {".py", ".java"})
        names = {p.name for p in result}
        assert names == {"main.py", "App.java"}

    def test_excludes_non_supported_extensions(self, tmp_path):
        (tmp_path / "main.py").write_text("print('hi')")
        (tmp_path / "readme.txt").write_text("hello")
        result = collect_files(tmp_path, {".py"})
        names = {p.name for p in result}
        assert "readme.txt" not in names
        assert "main.py" in names

    def test_excludes_excluded_directories(self, tmp_path):
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "lib.js").write_text("module.exports = {}")
        (tmp_path / "app.js").write_text("console.log('hi')")
        result = collect_files(tmp_path, {".js"})
        names = {p.name for p in result}
        assert "lib.js" not in names
        assert "app.js" in names

    def test_empty_directory(self, tmp_path):
        result = collect_files(tmp_path, {".py"})
        assert result == []

    def test_nested_directories(self, tmp_path):
        nested = tmp_path / "src" / "pkg"
        nested.mkdir(parents=True)
        (nested / "deep.py").write_text("x = 1")
        result = collect_files(tmp_path, {".py"})
        assert len(result) == 1
        assert result[0].name == "deep.py"

    def test_uses_default_exclusions_when_none(self, tmp_path):
        vc = tmp_path / "__pycache__"
        vc.mkdir()
        (vc / "cached.pyc").write_text("bytecode")
        (tmp_path / "app.py").write_text("x = 1")
        result = collect_files(tmp_path, {".py", ".pyc"})
        names = {p.name for p in result}
        assert "cached.pyc" not in names
        assert "app.py" in names


class TestDefaultExclusions:
    @pytest.mark.parametrize("expected", [
        "node_modules", "venv", "__pycache__", "build", "dist",
        "target", ".git", ".venv",
    ])
    def test_contains_directory_exclusion(self, expected):
        assert expected in DEFAULT_EXCLUSIONS

    @pytest.mark.parametrize("expected", [
        "**/generated-sources/**", "**/migrations/**",
    ])
    def test_contains_glob_exclusion(self, expected):
        assert expected in DEFAULT_EXCLUSIONS

    def test_contains_minified_file_exclusions(self):
        assert "*.min.js" in DEFAULT_EXCLUSIONS
        assert "*.min.css" in DEFAULT_EXCLUSIONS
