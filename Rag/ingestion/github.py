"""Read lightweight, public GitHub repository evidence without executing it."""

from pathlib import PurePosixPath
from typing import Any
from urllib.parse import quote, urlparse

import requests

from ..config import GITHUB_API_URL, MAX_DOCUMENT_CHARS, MAX_FILES_PER_REPOSITORY, REQUEST_TIMEOUT_SECONDS
from .loader import Document


class GitHubRepositoryError(Exception):
    """A linked GitHub repository cannot be inspected publicly."""


def github_repository_from_url(value: str) -> str | None:
    parsed = urlparse(value.strip())
    if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return None
    return f"{parts[0]}/{parts[1].removesuffix('.git')}"


def _relevant(path: str) -> bool:
    file_path = PurePosixPath(path)
    name = file_path.name.lower()
    if any(part.startswith(".") for part in file_path.parts) or file_path.suffix.lower() in {".bin", ".pt", ".pth", ".onnx", ".safetensors", ".zip", ".png", ".jpg"}:
        return False
    if name in {"readme.md", "readme", "model-card.md", "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg", "environment.yml", "environment.yaml", "config.json", "inference.py", "predict.py", "demo.py"}:
        return True
    return file_path.suffix == ".py" and any(part.lower() in {"examples", "example", "scripts", "src", "demo", "inference"} for part in file_path.parts)


class GitHubLoader:
    def __init__(self, session: requests.Session | None = None, timeout: int = REQUEST_TIMEOUT_SECONDS) -> None:
        self.session, self.timeout = session or requests.Session(), timeout
        self.session.headers.setdefault("Accept", "application/vnd.github+json")

    def _json(self, url: str) -> Any:
        try:
            response = self.session.get(url, timeout=self.timeout)
        except requests.RequestException as error:
            raise GitHubRepositoryError("Unable to reach the linked GitHub repository.") from error
        if response.status_code in {401, 403}:
            raise GitHubRepositoryError("Linked GitHub repository is private, rate-limited, or unavailable.")
        if response.status_code == 404:
            raise GitHubRepositoryError("Linked GitHub repository was not found.")
        response.raise_for_status()
        return response.json()

    def load(self, repository_url: str) -> list[Document]:
        repo_id = github_repository_from_url(repository_url)
        if not repo_id:
            raise ValueError("Not a GitHub repository URL.")
        repo = self._json(f"{GITHUB_API_URL}/repos/{repo_id}")
        branch = repo.get("default_branch") or "main"
        tree = self._json(f"{GITHUB_API_URL}/repos/{repo_id}/git/trees/{quote(branch, safe='')}?recursive=1")
        docs = [Document(f"https://github.com/{repo_id}", str(repo), "github_metadata", path="github/repository.json", origin="linked_github_repository")]
        relevant = [entry["path"] for entry in tree.get("tree", []) if entry.get("type") == "blob" and _relevant(entry.get("path", ""))][:MAX_FILES_PER_REPOSITORY]
        for path in relevant:
            encoded = quote(path, safe="/")
            url = f"https://raw.githubusercontent.com/{repo_id}/{branch}/{encoded}"
            try:
                response = self.session.get(url, timeout=self.timeout)
                if response.status_code != 200:
                    continue
                content = response.text[:MAX_DOCUMENT_CHARS]
            except requests.RequestException:
                continue
            docs.append(Document(url, content, PurePosixPath(path).name.lower(), path=f"github/{path}", origin="linked_github_repository"))
        return docs
