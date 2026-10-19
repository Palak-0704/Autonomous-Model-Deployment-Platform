"""Collect public model-card, metadata, and dependency files from Hugging Face."""

from dataclasses import dataclass
import json
from pathlib import PurePosixPath
import re
from typing import Any
from urllib.parse import quote, urlparse

import requests

from ..config import (
	HF_API_URL,
	HF_RAW_URL,
	MAX_DOCUMENT_CHARS,
	MAX_FILES_PER_REPOSITORY,
	MAX_TOTAL_DOCUMENT_CHARS,
	RELEVANT_FILE_NAMES,
	RELEVANT_DOCUMENT_SUFFIXES,
	RELEVANT_SOURCE_DIRECTORIES,
	RELEVANT_SOURCE_NAMES,
	REQUEST_TIMEOUT_SECONDS,
)


@dataclass
class Document:
	"""A small document with enough provenance for later retrieval."""

	source: str
	content: str
	document_type: str
	path: str = ""
	origin: str = "model_repository"


# Hugging Face repository IDs are plain ``owner/name`` tokens. Rejecting anything else
# keeps encoded dots/slashes, NUL bytes, and spaces out of the URLs built from the ID.
_REPOSITORY_NAME_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
_RESERVED_ROUTES = frozenset({
	"models", "datasets", "spaces", "api", "docs", "blog", "settings", "login", "join",
	"pricing", "papers", "collections", "organizations", "tasks", "chat", "new", "enterprise",
	"learn", "posts", "welcome",
})


class HuggingFaceAuthenticationError(Exception):
	"""Raised when Hugging Face requires or rejects repository authentication."""


class HuggingFaceRepositoryError(Exception):
	"""Raised when a repository cannot be accessed for a non-authentication reason."""


def repository_id_from_url(repository_url: str) -> str:
	"""Return ``owner/name`` from a Hugging Face model or repository URL."""
	parsed = urlparse(repository_url)
	if parsed.netloc not in {"huggingface.co", "www.huggingface.co"}:
		raise ValueError("repository_url must be a public huggingface.co URL")

	parts = [part for part in parsed.path.split("/") if part]
	if len(parts) < 1:
		raise ValueError("repository_url must point to a Hugging Face model repository")
	# For 'gpt2', parts is ['gpt2'], so we return it directly as the repo_id.
	if parts[0].lower() in _RESERVED_ROUTES:
		raise ValueError("repository_url must point directly to a Hugging Face model repository")
	repo_parts = parts[:2]
	if not all(_REPOSITORY_NAME_PART.fullmatch(part) for part in repo_parts):
		raise ValueError("repository_url contains characters that are not valid in a Hugging Face repository ID")
	return "/".join(repo_parts)


def _read_limited_text(response: Any, max_chars: int = MAX_DOCUMENT_CHARS) -> str:
	"""Read at most ``max_chars`` characters from a streamed response.

	Reading stops once enough bytes (UTF-8 is at most 4 bytes per character) have been
	seen, so a very large text file does not have to be downloaded or held in memory.
	"""
	limit = max_chars * 4
	chunks: list[bytes] = []
	size = 0
	for chunk in response.iter_content(chunk_size=65536):
		chunks.append(chunk)
		size += len(chunk)
		if size >= limit:
			break
	return b"".join(chunks)[:limit].decode("utf-8", errors="replace")[:max_chars]


class HuggingFaceLoader:
	"""Download lightweight repository information through public HTTP endpoints."""

	def __init__(self, timeout: int = REQUEST_TIMEOUT_SECONDS, token: str = "") -> None:
		self.timeout = timeout
		self.token = token.strip()
		self.session = requests.Session()
		if self.token:
			self.session.headers.update({"Authorization": f"Bearer {self.token}"})

	def _get_json(self, url: str) -> Any:
		try:
			response = self.session.get(url, timeout=self.timeout)
		except requests.RequestException as error:
			raise HuggingFaceRepositoryError("Unable to reach the Hugging Face repository.") from error
		if response.status_code in {401, 403}:
			raise HuggingFaceAuthenticationError("Hugging Face authentication required or rejected.")
		if response.status_code == 404:
			raise HuggingFaceRepositoryError("Hugging Face repository was not found.")
		try:
			response.raise_for_status()
			return response.json()
		except (requests.HTTPError, ValueError) as error:
			raise HuggingFaceRepositoryError(
				f"Hugging Face returned an unusable response (HTTP {response.status_code})."
			) from error

	def _get_text(self, url: str) -> str:
		try:
			# stream=True so an oversized file is never downloaded in full.
			response = self.session.get(url, timeout=self.timeout, stream=True)
		except requests.RequestException as error:
			raise HuggingFaceRepositoryError("Unable to download a repository file.") from error
		try:
			if response.status_code in {401, 403}:
				raise HuggingFaceAuthenticationError("Hugging Face authentication required or rejected.")
			if response.status_code == 404:
				raise HuggingFaceRepositoryError("A referenced repository file was not found.")
			try:
				response.raise_for_status()
				return _read_limited_text(response)
			except (requests.RequestException, ValueError) as error:
				raise HuggingFaceRepositoryError(
					f"Unable to read a repository file (HTTP {response.status_code})."
				) from error
		finally:
			response.close()

	@staticmethod
	def _is_relevant(path: str) -> bool:
		file_path = PurePosixPath(path)
		name = file_path.name.lower()

		# Explicitly exclude common large/binary files
		if any(name.endswith(ext) for ext in {".safetensors", ".bin", ".onnx", ".ckpt", ".pt", ".h5"}):
			return False
		if name in {"vocab.json", "merges.txt", "tokenizer.json", "added_tokens.json", "spiece.model"}:
			return False

		if name in {item.lower() for item in RELEVANT_FILE_NAMES}:
			return True
		if name in {item.lower() for item in RELEVANT_SOURCE_NAMES}:
			return True
		if file_path.suffix.lower() in RELEVANT_DOCUMENT_SUFFIXES:
			return True
		return file_path.suffix.lower() == ".py" and any(
			part.lower() in RELEVANT_SOURCE_DIRECTORIES for part in file_path.parts
		)

	def load(self, repository_url: str) -> list[Document]:
		"""Load repository metadata and relevant text/configuration files."""
		repo_id = repository_id_from_url(repository_url)
		metadata = self._get_json(f"{HF_API_URL}/models/{repo_id}")
		if metadata.get("gated") and not self.token:
			raise HuggingFaceAuthenticationError(
				"Hugging Face authentication required. Please provide a valid Hugging Face Access Token."
			)
		documents = [
			Document(
				source=f"{HF_API_URL}/models/{repo_id}",
				content=json.dumps(metadata, indent=2),
				document_type="model_metadata",
			)
		]

		total_chars = len(documents[0].content)
		# Apply the file cap to relevant files only. Capping the raw sibling list first
		# could discard README/config files in repositories with many weight shards.
		relevant_paths = [
			sibling["rfilename"]
			for sibling in metadata.get("siblings", [])
			if sibling.get("rfilename") and self._is_relevant(sibling["rfilename"])
		][:MAX_FILES_PER_REPOSITORY]
		for path in relevant_paths:
			raw_url = f"{HF_RAW_URL}/{repo_id}/resolve/main/{quote(path, safe='/')}"
			try:
				content = self._get_text(raw_url)
			except HuggingFaceAuthenticationError:
				raise
			except HuggingFaceRepositoryError:
				# A missing optional file should not prevent the model card loading.
				continue
			if total_chars + len(content) > MAX_TOTAL_DOCUMENT_CHARS:
				break
			total_chars += len(content)
			documents.append(
				Document(
					source=raw_url,
					content=content,
					document_type=PurePosixPath(path).name.lower(),
					path=path,
				)
			)
		# Model cards and metadata may link a project repository.  GitHub content
		# is extra evidence, never a source of executable instructions.
		from .github import GitHubLoader, GitHubRepositoryError, github_repository_from_url
		# Third-party hosts (GitHub, framework sites) must never see the Hugging Face
		# access token, so they get their own session with no Authorization header.
		external_session = requests.Session()
		links = _github_links(metadata, documents)
		for link in links[:1]:
			try:
				documents.extend(GitHubLoader(external_session, self.timeout).load(link))
			except (GitHubRepositoryError, requests.RequestException):
				# The Hugging Face repository remains usable when an optional link fails.
				continue
		# Framework documentation is compatibility evidence, never model-specific
		# evidence.  It is collected only for the declared library.
		from .documentation import FrameworkDocumentationLoader
		documents.extend(FrameworkDocumentationLoader(external_session, self.timeout).load(metadata))
		return documents


def load_repository(repository_url: str, token: str = "") -> list[Document]:
	"""Convenience function for the common one-shot ingestion workflow."""
	return HuggingFaceLoader(token=token).load(repository_url)


def _github_links(metadata: Any, documents: list[Document]) -> list[str]:
	"""Find only explicitly present GitHub repository links."""
	import re
	values = [json.dumps(metadata)] + [document.content for document in documents if document.path.lower().startswith("readme")]
	links: list[str] = []
	for value in values:
		for url in re.findall(r"https?://(?:www\.)?github\.com/[\w.-]+/[\w.-]+(?:/[^\s)\]}'\"]*)?", value):
			base = urlparse(url).scheme + "://" + urlparse(url).netloc + "/" + "/".join(urlparse(url).path.split("/")[:3])
			if base not in links:
				links.append(base)
	return links



