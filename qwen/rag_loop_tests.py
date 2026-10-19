"""Offline tests for duplicate and unavailable-answer loop control."""

from unittest.mock import patch

from .rag_loop import NOT_SPECIFIED, run_question_answer_loop


class FakeRag:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def ask(self, question: str) -> str:
        self.calls.append(question)
        return NOT_SPECIFIED


def main() -> None:
    rag = FakeRag()
    decisions = iter([
        ("ask", "What Python version is required?"),
        ("ask", "What Python version is required?"),
        ("ask", "What Python version is required?"),
    ])
    with patch("qwen.rag_loop.RepositoryRag.from_repository", return_value=rag), \
         patch("qwen.rag_loop._next_action", side_effect=lambda *_: next(decisions)), \
         patch("qwen.rag_loop.create_deployment_plan_with_response", return_value=("{}", {})):
        interactions, reason, _, _ = run_question_answer_loop("https://huggingface.co/demo/model")
    assert rag.calls == ["What Python version is required?"]
    assert len(interactions) == 3
    assert reason == "controller completed after repeated duplicate questions"
    print("Duplicate-question and unavailable-answer loop tests passed.")


if __name__ == "__main__":
    main()
