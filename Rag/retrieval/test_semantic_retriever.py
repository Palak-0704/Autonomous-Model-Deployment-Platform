"""Fast offline checks for embedding-based retrieval plumbing."""

from unittest.mock import patch

import numpy as np

from .retriever import KnowledgeRetriever


class FakeEmbeddingModel:
    """Deterministic semantic-space stand-in; avoids model download in unit tests."""

    def encode(self, texts, **_: object):
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for text in texts:
            lowered = text.casefold()
            if any(word in lowered for word in ("load", "initializ", "procedure", "from_pretrained")):
                vectors.append([1.0, 0.0, 0.0])
            elif any(word in lowered for word in ("inference", "generate", "predict")):
                vectors.append([0.0, 1.0, 0.0])
            elif any(word in lowered for word in ("weather", "recipe", "unrelated")):
                vectors.append([0.0, 0.0, 1.0])
            else:
                vectors.append([0.0, 0.0, 0.0])
        return np.array(vectors[0] if len(vectors) == 1 else vectors)


def main() -> None:
    knowledge = {
        "loading_code": "GPT2Tokenizer.from_pretrained('gpt2')\nGPT2Model.from_pretrained('gpt2')",
        "inference_code": "generator = pipeline('text-generation', model='gpt2')",
        "evidence": {
            "loading_code": [{"path": "README.md", "value": "GPT2Model.from_pretrained('gpt2')"}],
            "inference_code": [{"path": "README.md", "value": "pipeline('text-generation')"}],
        },
    }
    with patch("Rag.retrieval.retriever._embedding_model", return_value=FakeEmbeddingModel()):
        retriever = KnowledgeRetriever(knowledge)
        for question in ("How is the model loaded?", "How do I initialize the model?", "What is the loading procedure?"):
            result = retriever.retrieve(question)
            assert result and result[0]["field"] == "loading_code", question
        assert retriever.retrieve("What is today's weather?") == []
    print("Semantic retriever unit tests passed.")


if __name__ == "__main__":
    main()
