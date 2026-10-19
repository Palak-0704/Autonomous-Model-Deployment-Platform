"""Small allowlisted Transformers adapters; arbitrary repository code is never imported."""

from typing import Any

from .plan import SUPPORTED_TASKS


class ModelAdapter:
    def __init__(self, model_id: str, task: str) -> None:
        self.model_id, self.task, self.pipeline = model_id, task, None

    def load(self) -> None:
        if self.task not in SUPPORTED_TASKS:
            raise ValueError(f"No safe adapter for task '{self.task}'.")
        try:
            from transformers import pipeline
        except ImportError as error:
            raise RuntimeError("transformers is not installed in the active runtime.") from error
        # Never enable trust_remote_code from repository text.
        self.pipeline = pipeline(self.task, model=self.model_id, trust_remote_code=False)

    def smoke_input(self) -> Any:
        samples = {
            "text-generation": "Hello, my name is", "text2text-generation": "summarize: A short test sentence.",
            "fill-mask": "The capital of France is <mask>.", "text-classification": "This is a helpful product.",
            "token-classification": "Alice lives in London.", "question-answering": {"question": "What is the answer?", "context": "The answer is a smoke test."},
            "feature-extraction": "A small inference smoke test.",
        }
        if self.task in {"image-classification", "automatic-speech-recognition"}:
            raise ValueError(f"{self.task} needs a user-provided image or audio sample; automatic smoke input is unsafe.")
        return samples[self.task]

    def predict(self, value: Any) -> Any:
        if self.pipeline is None:
            raise RuntimeError("Target model is not loaded.")
        return self.pipeline(value)
