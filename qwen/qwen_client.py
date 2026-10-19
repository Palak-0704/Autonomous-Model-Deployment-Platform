"""Lazy Hugging Face client for Qwen2.5-Coder-3B-Instruct."""

from functools import lru_cache
import gc
import importlib
import os
from typing import Any


QWEN_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
_ACTIVE_QWEN: tuple[Any, Any] | None = None


class QwenClientError(RuntimeError):
    """Base error for Qwen loading and inference failures."""


class QwenAuthenticationError(QwenClientError):
    """The Hugging Face model requires authentication or access approval."""


class QwenHardwareError(QwenClientError):
    """The selected model/runtime cannot run on the available hardware."""


def _runtime_modules() -> tuple[Any, Any]:
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as error:
        raise QwenClientError(
            "Qwen dependencies are missing. Install torch, transformers, and accelerate."
        ) from error
    return torch, transformers


@lru_cache(maxsize=1)
def load_qwen() -> tuple[Any, Any]:
    """Download from Hugging Face on first use and cache tokenizer/model once."""
    torch, transformers = _runtime_modules()
    
    if not torch.cuda.is_available():
        raise QwenHardwareError(
            "Qwen requires a CUDA GPU for this project. "
            "Run the pipeline in a Google Colab GPU runtime."
        )

    token = os.getenv("HF_TOKEN") or None
    load_options: dict[str, Any] = {
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
        "token": token,
    }

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            QWEN_MODEL_ID,
            token=token,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_ID,
            **load_options,
        )
    except Exception as error:
        message = str(error).lower()
        if any(term in message for term in ("401", "403", "gated", "authentication", "token")):
            raise QwenAuthenticationError(
                f"Hugging Face authentication or access approval is required for {QWEN_MODEL_ID}."
            ) from error
        if any(term in message for term in ("out of memory", "memory", "cuda", "device")):
            raise QwenHardwareError(
                f"The selected Qwen model could not run with the available hardware/runtime: {error}"
            ) from error
        raise QwenClientError(
            f"Hugging Face could not load {QWEN_MODEL_ID}: {error}"
        ) from error

    model.eval()
    bind_loaded_qwen(tokenizer, model)
    return tokenizer, model


def bind_loaded_qwen(tokenizer: Any, model: Any) -> None:
    """Bind an already-loaded pair for request handlers in this Python process."""
    global _ACTIVE_QWEN
    _ACTIVE_QWEN = (tokenizer, model)


def active_qwen() -> tuple[Any, Any]:
    """Return the notebook-bound Qwen pair without ever loading a new model."""
    if _ACTIVE_QWEN is None:
        raise QwenClientError(
            "Qwen is not bound. Load Qwen once in the notebook and call "
            "bind_loaded_qwen() before starting Gradio."
        )
    return _ACTIVE_QWEN

def generate_response(prompt: str, max_new_tokens: int = 256) -> str:
    """Generate a response from the cached Qwen model."""
    if not prompt.strip():
        raise ValueError("Qwen prompt cannot be empty.")
    torch, _ = _runtime_modules()
    # Gradio callbacks use only the exact pair preloaded by the notebook.
    tokenizer, model = active_qwen()
    inputs: Any = None
    output: Any = None
    new_tokens: Any = None
    response = ""
    try:
        # Qwen is an instruction model.  Use its repository-provided chat
        # template when available so the response is a direct assistant turn.
        if getattr(tokenizer, "chat_template", None):
            rendered_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered_prompt = prompt
        inputs = tokenizer(rendered_prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                # The short control/code responses do not need a KV cache.
                # Disabling it prevents long RAG prompts from growing GPU memory.
                use_cache=False,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    except Exception as error:
        raise QwenClientError(f"Qwen inference failed: {error}") from error
    finally:
        # Keep the bound model/tokenizer resident; release only request tensors.
        inputs = None
        output = None
        new_tokens = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return response
