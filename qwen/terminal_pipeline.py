"""Terminal-only CUDA test for the complete RAG -> Qwen evidence pipeline."""

import json
import sys
import time
from pathlib import Path

# Add project root to path for relative imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from qwen.decision import _parse_json, unsupported_plan_values
from qwen.qwen_client import QWEN_MODEL_ID, load_qwen
from qwen.rag_loop import NOT_SPECIFIED, run_question_answer_loop


DEFAULT_REPOSITORY_URL = "https://huggingface.co/openai-community/gpt2"
DEFAULT_REQUEST = "Create an evidence-only environment setup, installation, loading, and inference plan."


def main(repository_url: str = DEFAULT_REPOSITORY_URL, user_request: str = DEFAULT_REQUEST) -> None:
    """Load Qwen once, run the internal loop, and print its evidence trail."""
    print("[1] Terminal pipeline started")
    start_time = time.time()
    if not torch.cuda.is_available():
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        print(
            "ERROR: This pipeline requires an NVIDIA CUDA runtime. "
            f"Detected GPU: {device}."
        )
        return

    print(f"[2] RAG/Qwen initialization started")
    print(f"REPOSITORY URL: {repository_url}")
    # print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading Qwen once: {QWEN_MODEL_ID}")
    load_qwen()
    print("Qwen loaded and bound; RAG/Qwen requests will reuse this instance.")

    interactions, completion_reason, raw_response, safe_plan = run_question_answer_loop(
        repository_url, user_request
    )
    print(f"[3] RAG/Qwen loop completed in {time.time() - start_time:.2f}s")
    for index, (question, answer) in enumerate(interactions, start=1):
        print(f"\n===== RAG / QWEN INTERACTION {index} =====")
        print("QWEN QUESTION:")
        print(question)
        print("RAG ANSWER / EVIDENCE:")
        print(answer)

    final_evidence = "\n\n".join(
        f"QWEN QUESTION:\n{question}\nRAG ANSWER:\n{answer}"
        for question, answer in interactions
    ) or NOT_SPECIFIED
    print(f"\nCOMPLETION DECISION: {completion_reason}")
    print("\nFINAL QWEN RAW RESPONSE:")
    print(raw_response)
    print("\nFINAL VALIDATED DEPLOYMENT PLAN:")
    print(json.dumps(safe_plan, indent=2, ensure_ascii=True))

    try:
        unsupported = unsupported_plan_values(_parse_json(raw_response), final_evidence)
    except ValueError:
        unsupported = {"response": "Qwen did not return parseable plan JSON"}
    print("\nUNSUPPORTED FACT AUDIT:")
    print(json.dumps(unsupported or {"result": "No unsupported facts generated."}, indent=2))
    if unsupported:
        raise AssertionError("Qwen generated unsupported facts; sanitizer replaced them in the final plan.")
    print("\nRAG -> Qwen terminal pipeline PASSED.")


if __name__ == "__main__":
    main(*sys.argv[1:3])
