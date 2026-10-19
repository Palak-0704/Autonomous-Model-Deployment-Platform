import unittest
from unittest.mock import Mock

from Rag.extraction.extractor import extract_knowledge
from Rag.ingestion.github import GitHubLoader, github_repository_from_url
from Rag.ingestion.loader import Document, HuggingFaceLoader, repository_id_from_url
from Rag.ingestion.documentation import FrameworkDocumentationLoader, declared_frameworks


class IngestionAndExtractionTests(unittest.TestCase):
    def test_url_parsing_rejects_non_model_routes(self):
        self.assertEqual(repository_id_from_url("https://huggingface.co/org/model/?a=1"), "org/model")
        with self.assertRaises(ValueError): repository_id_from_url("https://huggingface.co/models")
        with self.assertRaises(ValueError): repository_id_from_url("https://example.com/org/model")

    def test_github_url_parsing(self):
        self.assertEqual(github_repository_from_url("https://github.com/org/project/tree/main"), "org/project")
        self.assertIsNone(github_repository_from_url("https://example.com/org/project"))

    def test_extracts_manifest_and_code_with_provenance(self):
        docs = [
            Document("https://hf/api", '{"id":"org/demo","pipeline_tag":"text-generation","library_name":"transformers","tags":["pytorch"]}', "model_metadata"),
            Document("https://hf/readme", "# Usage\n```python\nfrom transformers import pipeline\ngenerator = pipeline('text-generation', model='org/demo')\n```", "readme.md", "README.md"),
            Document("https://hf/requirements", "transformers>=4.40\ntorch==2.2.0\n", "requirements.txt", "requirements.txt"),
        ]
        knowledge = extract_knowledge(docs, "https://huggingface.co/org/demo")
        self.assertEqual(knowledge.model_id, "org/demo")
        self.assertEqual(knowledge.dependency_versions["transformers"], ">=4.40")
        self.assertIn("torch", knowledge.dependencies)
        self.assertTrue(knowledge.inference_code)
        self.assertIn("requirements.txt", {item["path"] for item in knowledge.evidence["dependencies"]})

    def test_general_document_snippets_are_preserved_with_provenance(self):
        document = Document("https://hf/readme", "This model was trained to classify satellite images into land-cover classes.", "readme.md", "README.md")
        knowledge = extract_knowledge([document])
        self.assertIn("satellite images", knowledge.document_evidence[0])
        self.assertEqual(knowledge.evidence["document_evidence"][0]["source"], "https://hf/readme")

    def test_conflicting_dependency_versions_are_retained(self):
        docs = [Document("a", "torch==2.0", "requirements.txt", "requirements.txt"), Document("b", "torch==2.1", "requirements.txt", "github/requirements.txt")]
        knowledge = extract_knowledge(docs)
        self.assertEqual(knowledge.conflicts["dependency_versions.torch"], ["==2.0", "==2.1"])

    def test_cuda_example_is_not_a_hardware_requirement(self):
        knowledge = extract_knowledge([Document("x", 'model.to("cuda")', "readme.md", "README.md")])
        self.assertFalse(knowledge.hardware_requirements)
        self.assertFalse(knowledge.gpu_cpu_requirements)

    def test_framework_evidence_is_separate_from_model_requirements(self):
        docs = [
            Document("https://hf/model", "transformers", "readme.md", "README.md"),
            Document("https://hf/docs", "Python requires >=3.10", "framework_documentation", "framework_docs/transformers/installation", "official_framework_documentation"),
        ]
        knowledge = extract_knowledge(docs)
        self.assertIn("Python requires >=3.10", knowledge.framework_compatibility)
        self.assertNotIn("Python requires >=3.10", knowledge.model_specific_requirements)
        item = knowledge.evidence["python_requirements"][0]
        self.assertEqual(item["origin"], "official_framework_documentation")

    def test_framework_setup_python_range_is_extracted(self):
        document = Document("https://official/setup.py", "SUPPORTED_PYTHON_VERSIONS = (10, 14)", "framework_documentation", "framework_docs/transformers/setup.py", "official_framework_documentation")
        knowledge = extract_knowledge([document])
        self.assertIn("SUPPORTED_PYTHON_VERSIONS = (10, 14) (CPython 3.10 through 3.14)", knowledge.framework_compatibility)

    def test_pep440_constraints_and_markers_are_preserved(self):
        docs = [Document("https://example/requirements", '\n'.join((
            'python>=3.10,<3.13', 'torch~=2.2', 'tensorflow>=2.15; python_version < "3.13"', 'jax>=0.4.20',
        )), "requirements.txt", "requirements.txt")]
        knowledge = extract_knowledge(docs)
        self.assertEqual(knowledge.dependency_versions["python"], ">=3.10,<3.13")
        self.assertEqual(knowledge.dependency_versions["torch"], "~=2.2")
        self.assertEqual(knowledge.dependency_versions["tensorflow"], '>=2.15; python_version < "3.13"')
        self.assertEqual(knowledge.dependency_versions["jax"], ">=0.4.20")

    def test_github_loader_ignores_binary_and_preserves_path(self):
        session = Mock()
        response = Mock(status_code=200); response.json.side_effect = [
            {"default_branch": "main"},
            {"tree": [{"path": "README.md", "type": "blob"}, {"path": "weights.bin", "type": "blob"}]},
        ]
        raw = Mock(status_code=200, text="# Installation\npip install demo")
        session.get.side_effect = [response, response, raw]
        docs = GitHubLoader(session=session).load("https://github.com/org/project")
        self.assertEqual([doc.path for doc in docs], ["github/repository.json", "github/README.md"])
        self.assertIn("raw.githubusercontent.com/org/project/main/README.md", docs[1].source)

    def test_declared_frameworks_detects_all_supported_sources(self):
        metadata = {"library_name": "transformers", "tags": ["pytorch", "tensorflow", "jax", "onnx"]}
        self.assertEqual(declared_frameworks(metadata), ["transformers", "pytorch", "tensorflow", "jax", "onnxruntime"])

    def test_official_framework_documents_keep_attribution(self):
        session = Mock(); session.get.side_effect = [Mock(status_code=200, text="Install"), Mock(status_code=200, text="python_requires='>=3.10'")] * 4
        docs = FrameworkDocumentationLoader(session).load({"tags": ["pytorch", "tensorflow", "jax", "onnx"]})
        self.assertEqual(len(docs), 8)
        self.assertTrue(all(doc.origin == "official_framework_documentation" for doc in docs))
        self.assertIn("framework_docs/onnxruntime/setup.py", [doc.path for doc in docs])


if __name__ == "__main__": unittest.main()
