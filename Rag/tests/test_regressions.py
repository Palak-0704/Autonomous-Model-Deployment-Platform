"""Offline, deterministic regression tests for the Rag audit fixes.

No network access and no embedding-model download is used. The fake embeddings below
exercise the retriever's *logic* (thresholding, tie-breaking, provenance); they say
nothing about how well the real MiniLM model separates relevant from irrelevant text.
"""

import re
import unittest
from unittest.mock import Mock, patch

import numpy as np
import requests

from Rag.config import MAX_FILES_PER_REPOSITORY
from Rag.extraction.extractor import extract_knowledge
from Rag.ingestion.loader import (
    Document,
    HuggingFaceLoader,
    HuggingFaceRepositoryError,
    _read_limited_text,
    repository_id_from_url,
)
from Rag.retrieval.retriever import KnowledgeRetriever


class ConceptEmbedding:
    """Each concept group is one orthogonal dimension; unmatched text is a zero vector."""

    GROUPS = (
        ("gpu", "cpu", "vram", "memory", "hardware", "accelerator"),
        ("package", "dependency", "dependencies", "transformers", "torch", "library", "libraries"),
        ("mask", "masked", "fill", "pretrained", "checkpoint"),
        ("weather", "forecast", "rain"),
    )

    def encode(self, values, **_):
        single = isinstance(values, str)
        vectors = []
        for text in [values] if single else values:
            words = re.findall(r"[a-z]+", text.lower())  # whole words only, never substrings
            vector = np.array([float(sum(words.count(word) for word in group)) for group in self.GROUPS])
            norm = np.linalg.norm(vector)
            vectors.append(vector / norm if norm else vector)
        return vectors[0] if single else np.array(vectors)


class VectorEmbedding:
    """Maps a marker found in the text to an explicit 2-D vector (to control cosines)."""

    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, values, **_):
        single = isinstance(values, str)
        out = []
        for text in [values] if single else values:
            out.append(next((vec for marker, vec in self.vectors.items() if marker in text.lower()), [0.0, 0.0]))
        return np.array(out[0] if single else out)


def _evidence(source, path, origin, value):
    return {"source": source, "path": path, "origin": origin, "value": value}


class LoaderRegressionTests(unittest.TestCase):
    def test_hf_token_is_not_sent_to_third_party_hosts(self):
        seen = {}

        class SpyGitHub:
            def __init__(self, session, timeout):
                seen["github"] = dict(session.headers)

            def load(self, link):
                return []

        class SpyDocs:
            def __init__(self, session, timeout):
                seen["docs"] = dict(session.headers)

            def load(self, metadata):
                return []

        loader = HuggingFaceLoader(token="hf_SECRET")
        metadata = {"id": "o/m", "siblings": [{"rfilename": "README.md"}]}
        with patch.object(HuggingFaceLoader, "_get_json", return_value=metadata), \
             patch.object(HuggingFaceLoader, "_get_text", return_value="see https://github.com/foo/bar"), \
             patch("Rag.ingestion.github.GitHubLoader", SpyGitHub), \
             patch("Rag.ingestion.documentation.FrameworkDocumentationLoader", SpyDocs):
            loader.load("https://huggingface.co/o/m")
        self.assertEqual(loader.session.headers["Authorization"], "Bearer hf_SECRET")
        self.assertNotIn("Authorization", seen["github"])
        self.assertNotIn("Authorization", seen["docs"])

    def test_malformed_repository_ids_are_rejected(self):
        for url in (
            "https://huggingface.co/..%2f..%2fapi/x",
            "https://huggingface.co/org/mo del",
            "https://huggingface.co/org/a%00b",
            "https://huggingface.co/org/../x",
            "https://huggingface.co/blog/some-post",
            "https://huggingface.co/settings/tokens",
            "https://user:pw@huggingface.co/org/model",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                repository_id_from_url(url)

    def test_valid_repository_ids_still_work(self):
        self.assertEqual(repository_id_from_url("https://huggingface.co/gpt2"), "gpt2")
        self.assertEqual(repository_id_from_url("https://huggingface.co/org/my-model_v1.0/tree/main"), "org/my-model_v1.0")
        self.assertEqual(repository_id_from_url("https://www.huggingface.co/org/model?x=1"), "org/model")

    def _load_with_siblings(self, siblings):
        fetched = []

        def fake_text(self, url):
            fetched.append(url)
            return "text"

        with patch.object(HuggingFaceLoader, "_get_json", return_value={"id": "o/m", "siblings": siblings}), \
             patch.object(HuggingFaceLoader, "_get_text", fake_text), \
             patch("Rag.ingestion.github.GitHubLoader"), \
             patch("Rag.ingestion.documentation.FrameworkDocumentationLoader") as docs:
            docs.return_value.load.return_value = []
            HuggingFaceLoader().load("https://huggingface.co/o/m")
        return fetched

    def test_relevant_files_are_not_dropped_by_irrelevant_files_before_them(self):
        siblings = [{"rfilename": f"shard-{i:03d}.bin"} for i in range(60)] + [{"rfilename": "README.md"}, {"rfilename": "config.json"}]
        fetched = self._load_with_siblings(siblings)
        self.assertTrue(any(url.endswith("/README.md") for url in fetched))
        self.assertTrue(any(url.endswith("/config.json") for url in fetched))

    def test_file_cap_still_bounds_relevant_files(self):
        siblings = [{"rfilename": f"docs/page{i}.md"} for i in range(MAX_FILES_PER_REPOSITORY + 25)]
        self.assertLessEqual(len(self._load_with_siblings(siblings)), MAX_FILES_PER_REPOSITORY)

    def test_file_names_are_url_quoted(self):
        fetched = self._load_with_siblings([{"rfilename": "docs/read me?.md"}])
        self.assertEqual(len(fetched), 1)
        self.assertNotIn(" ", fetched[0])
        self.assertNotIn("?", fetched[0])

    def test_streamed_read_stops_after_the_limit(self):
        class Endless:
            chunks = 0

            def iter_content(self, chunk_size):
                while True:
                    self.chunks += 1
                    yield b"a" * chunk_size

        response = Endless()
        text = _read_limited_text(response, max_chars=1000)
        self.assertEqual(len(text), 1000)
        self.assertEqual(response.chunks, 1)

    def test_http_error_on_a_file_becomes_a_repository_error_and_closes_the_response(self):
        loader = HuggingFaceLoader()
        response = Mock(status_code=500)
        response.raise_for_status.side_effect = requests.HTTPError("server error")
        loader.session = Mock()
        loader.session.get.return_value = response
        with self.assertRaises(HuggingFaceRepositoryError):
            loader._get_text("https://huggingface.co/o/m/resolve/main/README.md")
        response.close.assert_called_once()

    def test_successful_file_read_returns_text(self):
        loader = HuggingFaceLoader()
        response = Mock(status_code=200)
        response.iter_content.return_value = [b"hello ", b"world"]
        loader.session = Mock()
        loader.session.get.return_value = response
        self.assertEqual(loader._get_text("https://huggingface.co/o/m/resolve/main/README.md"), "hello world")


class ExtractionRegressionTests(unittest.TestCase):
    def test_standard_library_imports_are_not_dependencies(self):
        readme = "# Usage\n```python\nimport argparse\nimport logging\nimport time\nimport torch\n```\n"
        knowledge = extract_knowledge([Document("u", readme, "readme.md", "README.md")])
        self.assertIn("torch", knowledge.dependencies)
        for stdlib in ("argparse", "logging", "time"):
            self.assertNotIn(stdlib, knowledge.dependencies)

    def test_checkpoint_transformers_version_is_not_a_requirement_or_conflict(self):
        documents = [
            Document("api", '{"id": "o/m", "config": {"transformers_version": "4.21.0"}}', "model_metadata"),
            Document("req", "transformers>=4.40\n", "requirements.txt", "requirements.txt"),
        ]
        knowledge = extract_knowledge(documents)
        self.assertEqual(knowledge.dependency_versions["transformers"], ">=4.40")
        self.assertEqual(knowledge.conflicts, {})
        self.assertIn("transformers_version=4.21.0", knowledge.configuration_requirements)

    def test_explicit_hardware_requirement_is_extracted(self):
        for line in ("This model requires a GPU with 16 GB of VRAM.", "At least 8 GB of RAM is recommended."):
            with self.subTest(line=line):
                knowledge = extract_knowledge([Document("u", line, "readme.md", "README.md")])
                self.assertEqual(knowledge.gpu_cpu_requirements, [line])

    def test_capability_statements_are_not_hardware_requirements(self):
        knowledge = extract_knowledge([Document("u", "This model supports CPU inference.", "readme.md", "README.md")])
        self.assertEqual(knowledge.gpu_cpu_requirements, [])

    def test_framework_documentation_is_not_a_model_hardware_requirement(self):
        document = Document(
            "https://pytorch.org/x", "PyTorch requires a GPU with enough memory for CUDA builds.",
            "framework_documentation", "framework_docs/pytorch/installation", "official_framework_documentation",
        )
        self.assertEqual(extract_knowledge([document]).gpu_cpu_requirements, [])

    def test_model_card_content_beyond_the_first_snippets_is_indexed(self):
        paragraphs = [f"Boilerplate paragraph number {i} that is long enough to be kept as a snippet." for i in range(12)]
        paragraphs.append("Intended use: this checkpoint is meant for masked language modeling and feature extraction.")
        knowledge = extract_knowledge([Document("u", "\n\n".join(paragraphs), "readme.md", "README.md")])
        self.assertTrue(any("Intended use" in snippet for snippet in knowledge.document_evidence))

    def test_source_code_comments_are_not_prose_evidence(self):
        source = Document(
            "https://raw/x.py", "# TODO: this file is now mostly used for benchmarking, remodel it in consequence.\nclass A: pass",
            "x.py", "github/examples/x.py", "linked_github_repository",
        )
        self.assertEqual(extract_knowledge([source]).document_evidence, [])

    def test_trust_remote_code_claim_is_accurate_and_scoped_to_the_model_repository(self):
        own = extract_knowledge([Document("u", "Do not set trust_remote_code=True.", "readme.md", "README.md")])
        self.assertEqual(len(own.special_requirements), 1)
        self.assertIn("mentions trust_remote_code", own.special_requirements[0])
        self.assertNotIn("is required", own.special_requirements[0])
        linked = extract_knowledge([Document("g", "from_pretrained(x, trust_remote_code=True)", "readme.md", "github/README.md", "linked_github_repository")])
        self.assertEqual(linked.special_requirements, [])

    def test_conflicts_carry_evidence_for_every_declaring_source(self):
        documents = [
            Document("a", "torch==2.0", "requirements.txt", "requirements.txt"),
            Document("b", "torch==2.1", "requirements.txt", "github/requirements.txt"),
        ]
        knowledge = extract_knowledge(documents)
        self.assertEqual({item["path"] for item in knowledge.evidence["conflicts"]}, {"requirements.txt", "github/requirements.txt"})
        self.assertTrue(all("==2.0" in item["value"] and "==2.1" in item["value"] for item in knowledge.evidence["conflicts"]))


class RetrievalRegressionTests(unittest.TestCase):
    def _retrieve(self, knowledge, query, embedding=None, **kwargs):
        with patch("Rag.retrieval.retriever._embedding_model", return_value=embedding or ConceptEmbedding()):
            return KnowledgeRetriever(knowledge).retrieve(query, **kwargs)

    def test_documented_hardware_requirement_is_retrieved_with_provenance(self):
        text = "This model requires a GPU with 16 GB VRAM."
        knowledge = {
            "gpu_cpu_requirements": [text], "dependencies": ["transformers"],
            "evidence": {
                "gpu_cpu_requirements": [_evidence("https://hf/readme", "README.md", "model_repository", text)],
                "dependencies": [_evidence("https://hf/req", "requirements.txt", "model_repository", "transformers")],
            },
        }
        result = self._retrieve(knowledge, "What accelerator does it need?", top_k=1)
        self.assertEqual(result[0]["field"], "gpu_cpu_requirements")
        self.assertEqual(result[0]["source"][0]["path"], "README.md")

    def test_undocumented_hardware_is_not_answered_with_dependency_or_config_text(self):
        knowledge = {
            "dependencies": ["transformers", "torch"], "configuration_requirements": ["torch_dtype=float32"],
            "evidence": {
                "dependencies": [_evidence("s", "requirements.txt", "model_repository", "transformers"), _evidence("s", "requirements.txt", "model_repository", "torch")],
                "configuration_requirements": [_evidence("s", "config.json", "model_repository", "torch_dtype=float32")],
            },
        }
        self.assertEqual(self._retrieve(knowledge, "What hardware is required?"), [])

    def test_unrelated_question_returns_no_evidence(self):
        text = "This checkpoint is pretrained with a masked language modeling objective."
        knowledge = {"document_evidence": [text], "evidence": {"document_evidence": [_evidence("s", "README.md", "model_repository", text)]}}
        self.assertEqual(self._retrieve(knowledge, "Will it rain tomorrow? What is the weather forecast?"), [])

    def test_unseen_paraphrase_finds_the_relevant_snippet(self):
        text = "This checkpoint is pretrained with a mask-filling objective."
        other = "Install with the transformers library."
        knowledge = {"document_evidence": [text, other], "evidence": {"document_evidence": [
            _evidence("s", "README.md", "model_repository", text), _evidence("s", "README.md", "model_repository", other)]}}
        result = self._retrieve(knowledge, "Can it fill in a missing word?", top_k=1)
        self.assertEqual(result[0]["value"], text)

    def test_clearly_more_relevant_evidence_from_another_source_beats_a_weak_model_repository_hit(self):
        embedding = VectorEmbedding({"query": [1.0, 0.0], "strong linked": [1.0, 0.0], "weak model": [0.7, 0.714]})
        knowledge = {"document_evidence": ["strong linked", "weak model"], "evidence": {"document_evidence": [
            _evidence("g", "github/README.md", "linked_github_repository", "strong linked"),
            _evidence("h", "README.md", "model_repository", "weak model")]}}
        result = self._retrieve(knowledge, "query", embedding, top_k=2)
        self.assertEqual([item["value"] for item in result], ["strong linked", "weak model"])
        # The reported score is the real cosine similarity, not a boosted ranking score.
        self.assertAlmostEqual(result[0]["score"], 1.0, places=2)
        self.assertAlmostEqual(result[1]["score"], 0.7, places=2)

    def test_model_repository_preference_cannot_lift_a_candidate_over_the_relevance_threshold(self):
        embedding = VectorEmbedding({"query": [1.0, 0.0], "just below": [0.2, 0.98]})
        knowledge = {"document_evidence": ["just below"], "evidence": {"document_evidence": [
            _evidence("h", "README.md", "model_repository", "just below")]}}
        self.assertEqual(self._retrieve(knowledge, "query", embedding), [])

    def test_conflicting_declarations_are_retrievable_from_every_source(self):
        documents = [
            Document("a", "torch==2.0", "requirements.txt", "requirements.txt"),
            Document("b", "torch==2.1", "requirements.txt", "github/requirements.txt"),
        ]
        knowledge = extract_knowledge(documents)
        results = self._retrieve(knowledge, "Which library versions are declared?", top_k=8)
        conflicts = [item for item in results if item["field"] == "conflicts"]
        self.assertEqual({item["source"][0]["path"] for item in conflicts}, {"requirements.txt", "github/requirements.txt"})


if __name__ == "__main__":
    unittest.main()
