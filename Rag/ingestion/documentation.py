"""Bounded collection of official, framework-level compatibility sources."""

import re
from typing import Any
import requests

from ..config import MAX_DOCUMENT_CHARS, REQUEST_TIMEOUT_SECONDS
from .loader import Document

# These are framework-level sources, not model-specific facts. Keys are found
# only from repository-declared library/tags; no model ID is special-cased.
OFFICIAL_FRAMEWORK_DOCS = {
    "transformers": ("https://huggingface.co/docs/transformers/installation", "https://raw.githubusercontent.com/huggingface/transformers/main/setup.py"),
    "pytorch": ("https://pytorch.org/get-started/locally/", "https://raw.githubusercontent.com/pytorch/pytorch/main/setup.py"),
    "tensorflow": ("https://www.tensorflow.org/install", "https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/pip_package/setup.py"),
    "jax": ("https://docs.jax.dev/en/latest/installation.html", "https://raw.githubusercontent.com/jax-ml/jax/main/setup.py"),
    "onnxruntime": ("https://onnxruntime.ai/docs/install/", "https://raw.githubusercontent.com/microsoft/onnxruntime/main/setup.py"),
}
_ALIASES = {"transformers": "transformers", "torch": "pytorch", "pytorch": "pytorch", "tensorflow": "tensorflow", "tf": "tensorflow", "jax": "jax", "flax": "jax", "onnx": "onnxruntime", "onnxruntime": "onnxruntime", "onnx-runtime": "onnxruntime"}


def declared_frameworks(metadata: dict[str, Any]) -> list[str]:
    values = [metadata.get("library_name", ""), *metadata.get("tags", [])]
    config = metadata.get("config", {})
    if isinstance(config, dict): values.extend(config.keys())
    found = []
    for value in values:
        framework = _ALIASES.get(str(value).casefold())
        if framework and framework not in found: found.append(framework)
    return found

class FrameworkDocumentationLoader:
    def __init__(self, session: requests.Session, timeout: int = REQUEST_TIMEOUT_SECONDS) -> None:
        self.session, self.timeout = session, timeout

    def load(self, metadata: dict[str, Any]) -> list[Document]:
        documents = []
        for framework in declared_frameworks(metadata):
            for index, url in enumerate(OFFICIAL_FRAMEWORK_DOCS[framework]):
                try:
                    response = self.session.get(url, timeout=self.timeout)
                    if response.status_code != 200: continue
                except requests.RequestException:
                    continue
                text = response.text if url.endswith(".py") else re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", response.text))
                documents.append(Document(url, text[:MAX_DOCUMENT_CHARS], "framework_documentation", path=f"framework_docs/{framework}/{'installation' if index == 0 else 'setup.py'}", origin="official_framework_documentation"))
        return documents
