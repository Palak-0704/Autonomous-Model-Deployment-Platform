"""Offline tests for the Gradio callback; no Qwen model or UI server is started."""

from unittest.mock import patch

from .test import generate_code


def main() -> None:
    interactions = [("How is the model loaded?", "README: model = GPT2Model.from_pretrained('gpt2')")]
    with patch("qwen.test.run_question_answer_loop", return_value=(interactions, "complete", "{}", {})), \
         patch("qwen.test.create_evidence_only_code", return_value="model = GPT2Model.from_pretrained('gpt2')") as code:
        result = generate_code("https://huggingface.co/openai-community/gpt2", "Show loading code")
    assert result == "model = GPT2Model.from_pretrained('gpt2')"
    assert code.call_args.args[0] == "Show loading code"
    print("Gradio callback test passed: only final code is returned.")


if __name__ == "__main__":
    main()
