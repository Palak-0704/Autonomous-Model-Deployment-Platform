# Autonomous ML Model Deployment and Testing Platform

Local development is safe on a CPU-only machine: repository inspection does not download target-model weights or Qwen. Run `./.venv/Scripts/python.exe app.py` and use **Inspect and validate plan** first. The download/load button is intentionally explicit; use it only for a small CPU-capable model. Heavy Qwen work belongs in Colab.

The existing Colab Sync configuration automatically mirrors this workspace to `/content/Project -2` (excluding `.venv`). In Colab, select a GPU runtime, wait for sync, then run:

```python
%cd "/content/Project -2"
!pip install -r Rag/requirement.txt accelerate
!python -m qwen.terminal_pipeline
```

Qwen is loaded once per Colab kernel. It requires CUDA, but no longer needs a particular GPU model; a compatible CUDA GPU is sufficient. Local target-model execution remains opt-in and only uses the allowlisted Transformers adapters. Custom repository code is blocked. The target-model UI currently accepts text for supported text tasks; image and audio task adapters are planned but intentionally not represented by a misleading textbox.
