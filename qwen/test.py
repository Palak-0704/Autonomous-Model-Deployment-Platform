"""Simple Gradio UI for the internal RAG <-> Qwen evidence-only code flow."""

from typing import Any

from .decision import create_evidence_only_code
from .qwen_client import bind_loaded_qwen
from .rag_loop import NOT_SPECIFIED, run_question_answer_loop


DEFAULT_REQUEST = "Show repository-documented model loading and inference code."


def generate_code(repository_url: str, query: str = "") -> str:
    """Run RAG <-> Qwen internally and return only final validated code."""
    request = query.strip() or DEFAULT_REQUEST
    try:
        interactions, _, _, _ = run_question_answer_loop(repository_url.strip(), request)
    except Exception as error:
        return f"# RAG/Qwen request failed: {error}"

    evidence = "\n\n".join(
        f"QWEN QUESTION:\n{question}\nRAG ANSWER:\n{answer}"
        for question, answer in interactions
    ) or NOT_SPECIFIED
    try:
        return create_evidence_only_code(request, repository_url.strip(), evidence)
    except Exception as error:
        return f"# Qwen code generation failed: {error}"


def create_app():
    """Create the minimal UI; no generated code is executed."""
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError("Gradio is required to launch this UI.") from error
    with gr.Blocks(title="RAG to Qwen Code Test") as app:
        gr.Markdown("# RAG ↔ Qwen code test")
        repository_url = gr.Textbox(
            label="Hugging Face model/repository URL",
            value="https://huggingface.co/openai-community/gpt2",
        )
        query = gr.Textbox(label="Optional request", placeholder=DEFAULT_REQUEST)
        submit = gr.Button("Generate evidence-only code")
        output = gr.Code(label="Qwen final generated code", language="python")
        submit.click(generate_code, inputs=[repository_url, query], outputs=output)
    return app


def launch_colab(tokenizer: Any, model: Any):
    """Launch the UI only from a CUDA-enabled Colab runtime."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Connect this notebook to a Colab GPU runtime.")
    # This binds the notebook's already-loaded model. No callback performs a
    # second from_pretrained call or creates another Qwen model instance.
    bind_loaded_qwen(tokenizer, model)
    print(f"Colab GPU: {torch.cuda.get_device_name(0)}")
    # One GPU generation at a time prevents simultaneous browser requests from
    # duplicating temporary inference memory while retaining the bound model.
    return create_app().queue(default_concurrency_limit=1).launch(share=True)


if __name__ == "__main__":
    from .qwen_client import load_qwen

    launch_colab(*load_qwen())
