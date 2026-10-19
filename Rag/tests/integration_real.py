"""Real network RAG evaluation. Not part of default unit-test discovery."""

import json
from pathlib import Path

from Rag.extraction.extractor import extract_knowledge
from Rag.ingestion.loader import load_repository
from Rag.retrieval.retriever import KnowledgeRetriever


CASES = (
    ("https://huggingface.co/distilbert/distilroberta-base", "What purpose is this checkpoint designed for?", ("mask", "fill-mask")),
    ("https://huggingface.co/openai/whisper-tiny", "Can it turn spoken audio into written words?", ("speech recognition", "transcrib")),
	("https://huggingface.co/google/vit-base-patch16-224", "Can I provide a photograph and receive a category?", ("image classification", "image-classification")),
)


def main() -> None:
    report = {"kind": "real network integration", "cases": []}
    for url, question, expected in CASES:
        documents = load_repository(url)
        knowledge = extract_knowledge(documents, url)
        retriever = KnowledgeRetriever(knowledge)
        results = retriever.retrieve(question, top_k=3)
        evidence = "\n".join(item["value"] for item in results)
        probes = {}
        for probe in ("Which dependencies are documented?", "How is the model loaded?", "How is inference performed?", "What hardware is required?", "What is the weather forecast?"):
            found = retriever.retrieve(probe, top_k=2)
            probes[probe] = {"fields": [item["field"] for item in found], "evidence": "\n".join(item["value"] for item in found)[:800]}
        report["cases"].append({
            "url": url, "question": question, "expected": expected,
            "paths": [document.path or document.document_type for document in documents],
            "github_paths": [document.path for document in documents if document.path.startswith("github/")],
			"official_framework_paths": [document.path for document in documents if document.origin == "official_framework_documentation"],
			"framework_compatibility": knowledge.framework_compatibility,
            "task": knowledge.task, "result_fields": [item["field"] for item in results],
            "retrieved_evidence": evidence[:2000],
            "probes": probes,
            "pass": any(term.casefold() in evidence.casefold() for term in expected),
        })
    output = Path("reports/rag_verification/real_integration.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not all(case["pass"] for case in report["cases"]):
        raise SystemExit("One or more real retrieval checks failed.")


if __name__ == "__main__":
    main()
