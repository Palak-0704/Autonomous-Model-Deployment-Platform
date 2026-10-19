"""Colab GPU demonstration of the Qwen <-> RAG question-answer loop."""

import json

import torch

from .qwen_client import QWEN_MODEL_ID, load_qwen
from .rag_loop import run_question_answer_loop


def main() -> None:
    repository_url = "https://huggingface.co/openai-community/gpt2"
    if not torch.cuda.is_available():
        raise RuntimeError("No NVIDIA GPU is active in the Colab kernel.")
    print(f"REPOSITORY URL: {repository_url}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading directly from Hugging Face: {QWEN_MODEL_ID}")
    load_qwen()
    interaction, completion_reason, raw_response, plan = run_question_answer_loop(repository_url)
    for number, (question, answer) in enumerate(interaction, start=1):
        print(f"\n===== INTERACTION {number} =====")
        print("QWEN QUESTION:")
        print(question)
        print("RAG ANSWER:")
        print(answer)
    print(f"\nCOMPLETION DECISION: {completion_reason}.")
    print("\nFINAL QWEN RAW RESPONSE:")
    print(raw_response)
    print("\nFINAL DEPLOYMENT PLAN:")
    print(json.dumps(plan, indent=2, ensure_ascii=True))
    print("\nQwen <-> RAG question-answer loop PASSED.")


if __name__ == "__main__":
    main()
