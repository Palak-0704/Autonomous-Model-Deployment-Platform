import unittest
from unittest.mock import patch
import numpy as np

from Rag.retrieval.retriever import KnowledgeRetriever


class FakeEmbeddingModel:
    def encode(self, values, **_):
        if isinstance(values, str): values = [values]
        vectors = []
        for value in values:
            text = value.lower()
            if any(word in text for word in ("weather", "hardware", "accelerator")):
                vectors.append([0., 0.])
            elif any(word in text for word in ("transformers", "dependency", "package", "library")):
                vectors.append([1., 0.])
            else:
                vectors.append([0., 1.])
        return np.array(vectors[0] if len(vectors) == 1 else vectors)


class RetrievalTests(unittest.TestCase):
    def test_returns_matched_snippet_not_everything_in_source(self):
        knowledge = {"dependencies": ["transformers>=4.0"], "loading_code": ["model = AutoModel.from_pretrained('demo')"], "evidence": {
            "dependencies": [{"source": "x", "path": "README.md", "value": "transformers>=4.0"}],
            "loading_code": [{"source": "x", "path": "README.md", "value": "model = AutoModel.from_pretrained('demo')"}],
        }}
        with patch("Rag.retrieval.retriever._embedding_model", return_value=FakeEmbeddingModel()):
            result = KnowledgeRetriever(knowledge).retrieve("Which dependency is required?", top_k=1)
        self.assertEqual(result[0]["field"], "dependencies")
        self.assertNotIn("from_pretrained", result[0]["value"])
        self.assertEqual(result[0]["source"][0]["path"], "README.md")

    def test_list_evidence_is_split_into_individual_candidates(self):
        knowledge = {"python_requirements": ["python>=3.10", "torch>=2.0"], "evidence": {
            "python_requirements": [{"source": "x", "path": "requirements.txt", "value": ["python>=3.10", "torch>=2.0"]}],
        }}
        candidates = KnowledgeRetriever(knowledge)._candidates
        self.assertEqual([candidate["result_text"] for candidate in candidates], ["python>=3.10", "torch>=2.0"])
        self.assertEqual([candidate["source"][0]["value"] for candidate in candidates], ["python>=3.10", "torch>=2.0"])

    def test_absent_question_is_empty(self):
        with patch("Rag.retrieval.retriever._embedding_model", return_value=FakeEmbeddingModel()):
            # Model-source preference must not promote a zero-similarity
            # passage above the relevance threshold.
            result = KnowledgeRetriever({"task": "fill-mask", "evidence": {"task": [{"path": "metadata", "origin": "model_repository", "value": "fill-mask"}]}}).retrieve("weather forecast")
        self.assertEqual(result, [])

    def test_missing_hardware_evidence_is_not_inferred_from_unrelated_fields(self):
        with patch("Rag.retrieval.retriever._embedding_model", return_value=FakeEmbeddingModel()):
            result = KnowledgeRetriever({"task": "fill-mask", "evidence": {"task": [{"path": "metadata", "value": "fill-mask"}]}}).retrieve("What hardware is required?")
        self.assertEqual(result, [])

    def test_metadata_buckets_do_not_claim_to_be_requirements(self):
        knowledge = {
            "model_specific_requirements": ["transformers"],
            "required_files": ["config.json"],
            "evidence": {
                "model_specific_requirements": [{"path": "README.md", "value": "transformers"}],
                "required_files": [{"path": "config.json", "value": "config.json"}],
            },
        }
        candidates = KnowledgeRetriever(knowledge)._candidates
        self.assertIn("model-specific fact: transformers", [item["text"] for item in candidates])
        self.assertIn("repository file: config.json", [item["text"] for item in candidates])


    def test_framework_compatibility_has_attributed_cross_source_evidence(self):
        knowledge = {"framework_compatibility": ["Python >=3.10"], "evidence": {"framework_compatibility": [{"source": "https://official/docs", "path": "framework_docs/transformers/installation", "origin": "official_framework_documentation", "value": "Python >=3.10"}]}}
        with patch("Rag.retrieval.retriever._embedding_model", return_value=FakeEmbeddingModel()):
            result = KnowledgeRetriever(knowledge).retrieve("What Python compatibility does the framework document?")
        self.assertEqual(result[0]["source"][0]["origin"], "official_framework_documentation")

    def test_unseen_natural_language_phrase_retrieves_document_evidence(self):
        knowledge = {"document_evidence": ["This checkpoint was pretrained on masked language modeling using English text."], "evidence": {"document_evidence": [{"source": "https://hf/readme", "path": "README.md", "origin": "model_repository", "value": "This checkpoint was pretrained on masked language modeling using English text."}]}}
        with patch("Rag.retrieval.retriever._embedding_model", return_value=FakeEmbeddingModel()):
            result = KnowledgeRetriever(knowledge).retrieve("What sort of material taught this checkpoint?", top_k=1)
        self.assertEqual(result[0]["field"], "document_evidence")
        self.assertEqual(result[0]["source"][0]["path"], "README.md")

    def test_model_repository_evidence_beats_generic_linked_repository(self):
        knowledge = {"document_evidence": ["The model fills a missing word marked with a mask.", "Generic framework configuration documentation."], "evidence": {"document_evidence": [
            {"source": "https://hf/readme", "path": "README.md", "origin": "model_repository", "value": "The model fills a missing word marked with a mask."},
            {"source": "https://github.com/upstream", "path": "github/README.md", "origin": "linked_github_repository", "value": "Generic framework configuration documentation."},
        ]}}
        with patch("Rag.retrieval.retriever._embedding_model", return_value=FakeEmbeddingModel()):
            result = KnowledgeRetriever(knowledge).retrieve("Can it fill a missing word?", top_k=1)
        self.assertEqual(result[0]["source"][0]["origin"], "model_repository")


if __name__ == "__main__": unittest.main()
