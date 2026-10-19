"""Real public-repository test for semantic RAG retrieval."""

from ..extraction.extractor import extract_knowledge
from ..ingestion.loader import load_repository
from .retriever import KnowledgeRetriever


LOADING_QUESTIONS = (
    "How is the model loaded?",
    "How do I initialize the model?",
    "What is the loading procedure?",
)
OTHER_QUESTIONS = (
    "How is inference performed?",
    "What input does the model accept?",
    "What does the model return?",
    "What libraries are required?",
)


def main() -> None:
    repository_url = "https://huggingface.co/openai-community/gpt2"
    knowledge = extract_knowledge(load_repository(repository_url), repository_url=repository_url)
    retriever = KnowledgeRetriever(knowledge)

    for question in LOADING_QUESTIONS:
        results = retriever.retrieve(question, top_k=5)
        values = "\n".join(item["value"] for item in results)
        assert "GPT2Tokenizer.from_pretrained" in values, (question, results)
        assert "GPT2Model.from_pretrained" in values, (question, results)
        print(f"PASS loading: {question}")

    for question in OTHER_QUESTIONS:
        results = retriever.retrieve(question, top_k=3)
        assert results, question
        if question == "What libraries are required?":
            assert "transformers" in "\n".join(item["value"] for item in results).casefold()
        print(f"PASS evidence: {question} -> {[item['field'] for item in results]}")

    assert retriever.retrieve("What is today's weather forecast?", top_k=3) == []
    print("Real GPT-2 semantic retrieval test passed.")


if __name__ == "__main__":
    main()
