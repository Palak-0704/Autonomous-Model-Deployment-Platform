"""Embedding-based semantic retrieval over extracted repository evidence."""

from functools import lru_cache
import re
from typing import Any

from ..config import DEFAULT_TOP_K
from ..extraction.extractor import ModelKnowledge


EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# Calibrated below the unrelated-evidence range for this compact model while
# retaining short factual snippets such as package names.
# Calibrated below the unrelated-evidence range for this compact model while
# retaining short factual snippets such as package names and task labels.
MIN_SIMILARITY = 0.22
NOT_FOUND_MESSAGE = "Information not found in available sources"
# Source-quality preference for the supplied model repository. It only orders candidates
# that already passed MIN_SIMILARITY, and is small enough that a clearly more relevant
# passage from another source still ranks first. Not empirically calibrated.
MODEL_REPOSITORY_TIE_BREAK = 0.05



@lru_cache(maxsize=1)
def _embedding_model() -> Any:
    """Load one small sentence model lazily; no database or remote index is used."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Semantic retrieval requires sentence-transformers. "
            "Install the RAG requirements before creating KnowledgeRetriever."
        ) from error
    return SentenceTransformer(EMBEDDING_MODEL_ID)


class KnowledgeRetriever:
    """Rank repository evidence by sentence-embedding cosine similarity."""

    def __init__(self, knowledge: ModelKnowledge | dict[str, Any]) -> None:
        self.knowledge = knowledge.to_dict() if isinstance(knowledge, ModelKnowledge) else knowledge
        self._candidates = self._build_candidates()
        self._embeddings: Any | None = None

    @staticmethod
    def _content(value: Any) -> str:
        if isinstance(value, list):
            return "\n".join(str(item) for item in value)
        if isinstance(value, dict):
            return "; ".join(f"{key}: {item}" for key, item in value.items())
        return str(value)

    def _build_candidates(self) -> list[dict[str, Any]]:
        """Build only provenance-backed evidence documents for semantic search."""
        evidence = self.knowledge.get("evidence", {})
        candidates = []
        for field, value in self.knowledge.items():
            if field in {"evidence", "source_documents"} or not value or not evidence.get(field):
                continue
            # Embed each extracted evidence snippet, rather than a field name or
            # a large concatenation of unrelated snippets. This lets natural
            # paraphrases match the actual repository statement/code.
            for item in evidence[field]:
                raw_value = item.get("value", "")
                # Extraction occasionally records a collection of facts under
                # one field. Index each fact separately so a Python/CUDA query
                # cannot return every unrelated value from that collection.
                values = raw_value if isinstance(raw_value, list) else [raw_value]
                for value in values:
                    text = self._content(value).strip()
                    if not text:
                        continue
                    # Field names are generic evidence metadata, not question
                    # routing. They give terse facts such as ``fill-mask`` a
                    # meaningful semantic context while preserving the exact
                    # original value returned to callers.
                    # ``model_specific_requirements`` is a provenance bucket,
                    # not a statement that every value is a hardware/runtime
                    # requirement.  Calling it a generic fact avoids making a
                    # dependency such as ``transformers`` match a question
                    # about required hardware.
                    if field == "model_specific_requirements":
                        label = "model-specific fact"
                    elif field == "required_files":
                        # A file being collected is not a requirement in the
                        # ordinary-language sense (for example, it is not a
                        # hardware requirement).
                        label = "repository file"
                    else:
                        label = field.replace("_", " ")
                    embedding_text = f"{label}: {text}"
                    source_item = dict(item)
                    source_item["value"] = text
                    candidates.append({"field": field, "text": embedding_text, "source": [source_item], "result_text": text})
        return candidates

    def _candidate_embeddings(self) -> Any:
        if self._embeddings is None:
            self._embeddings = _embedding_model().encode(
                [candidate["text"] for candidate in self._candidates],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            # Some test/deterministic encoders return a single vector as 1-D.
            # Normal sentence-transformers returns 2-D for a list, but keeping
            # this normalisation makes the retriever's contract explicit.
            if getattr(self._embeddings, "ndim", 1) == 1:
                self._embeddings = self._embeddings[None, :]
        return self._embeddings

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        """Return only semantically relevant, provenance-backed repository facts."""
        if not query or top_k <= 0 or not self._candidates:
            return []

        model = _embedding_model()
        query_embedding = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        # L2-normalized vector dot products are cosine similarities.
        raw_scores = self._candidate_embeddings() @ query_embedding
        # The supplied model card/config is normally more specific than an
        # upstream framework repository. This is a source-quality preference,
        # applied independently of the question wording.
        scores = raw_scores.copy()
        for index, candidate in enumerate(self._candidates):
            origin = candidate["source"][0].get("origin", "")
            if origin == "model_repository":
                # Prefer direct model-card/config evidence over an upstream
                # framework repository when both are plausible. This remains
                # independent of intent/category wording.
                scores[index] += MODEL_REPOSITORY_TIE_BREAK
        ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)
        results = []
        seen_values = set()
        for index, score in ranked:
            # Provenance is a tie-breaker for relevant evidence only.  It must
            # never turn a semantically unrelated model-card sentence into an
            # answer simply because that sentence is model-specific.
            if float(raw_scores[index]) < MIN_SIMILARITY:
                continue
            if len(results) >= min(top_k, 8):
                break
            candidate = self._candidates[index]
            content_key = re.sub(r"\s+", " ", candidate["result_text"].casefold())
            if content_key in seen_values:
                continue
            seen_values.add(content_key)
            results.append({
                "field": candidate["field"],
                "value": candidate["result_text"][:4000],
                "source": candidate["source"],
                "score": round(float(raw_scores[index]), 3),
                "evidence_type": "semantic",
            })
        return results

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        """Alias for retrieve."""
        return self.retrieve(query, top_k)
