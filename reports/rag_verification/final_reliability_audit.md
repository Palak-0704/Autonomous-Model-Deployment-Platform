# RAG final reliability audit

Date: 2026-10-14

## Scope inspected

The complete `Rag/` pipeline was inspected: Hugging Face ingestion (`ingestion/loader.py`), linked GitHub and official framework-documentation collection (`ingestion/github.py`, `ingestion/documentation.py`), extraction (`extraction/extractor.py`), semantic retrieval (`retrieval/retriever.py`), the Gradio test interface (`test.py`), unit tests, semantic-retriever tests, and the real-network integration runner.

The pipeline remains: repository URL -> bounded text ingestion -> provenance backed extraction -> semantic evidence retrieval. It does not run repository code, download model weights, use a vector database, or generate answers with an LLM.

## Commands actually executed

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s . -p "test*.py" -v
.\.venv\Scripts\python.exe -m Rag.retrieval.test_semantic_retriever
.\.venv\Scripts\python.exe -m Rag.tests.integration_real
```

Results:

* Unit discovery: **18/18 passed**.
* Semantic retriever unit script: **passed**.
* Real network integration: its three primary task checks **passed** and rewrote `real_integration.json` at 2026-10-14 15:01 local time.

## Real repositories and evidence collected

| Repository | Task query | Primary check | Examples of retained sources |
| --- | --- | --- | --- |
| `distilbert/distilroberta-base` | What purpose is this checkpoint designed for? | Pass (`fill-mask`) | HF `README.md`, `config.json`, model metadata; linked Transformers GitHub files; official Transformers, PyTorch, TensorFlow, and JAX documentation |
| `openai/whisper-tiny` | Can it turn spoken audio into written words? | Pass (ASR/transcription) | HF `README.md`, `config.json`, processor files; linked Whisper `README.md`, `pyproject.toml`, and `requirements.txt`; official framework documentation |
| `google/vit-base-patch16-224` | Can I provide a photograph and receive a category? | Pass (`image-classification`) | HF `README.md`, `config.json`, preprocessing config; linked ViT `README.md`, `setup.py`, and JAX requirements; official framework documentation |

The complete source paths, URLs, extracted compatibility evidence, top-three results, and probe outputs are in `real_integration.json`.

## Fix made during this audit

The retriever added a `+0.45` source preference to model-repository evidence before evaluating the semantic relevance threshold. That could promote a semantically unrelated model-card passage to a result solely because it was from the model repository.

`retrieval/retriever.py` now applies `MIN_SIMILARITY` to the unadjusted cosine score. Provenance preference is used only to order candidates that have already met that relevance test. A regression test covers a zero-similarity, model-repository candidate for an absent question.

## Findings and release decision

Extraction and provenance checks passed: version strings and markers are kept as source text, framework compatibility is retained separately from model-specific requirements, and conflicting dependency declarations are not silently reconciled.

The current implementation is **not ready to be frozen as a reliable open-ended-question component**. The real integration report still shows two important retrieval failures for all or some of the tested repositories:

* `What is the weather forecast?` returns unrelated model-card/documentation passages rather than an empty/insufficient result.
* `What hardware is required?` returns dependency/configuration evidence when the repository does not state a hardware requirement.

The primary task checks demonstrate that relevant evidence is available and attributed, but they do not validate abstention or top-result precision. The existing compact bi-encoder plus a fixed similarity threshold cannot yet reliably distinguish these unsupported questions from loosely related text.

The result should therefore be treated as an evidence retriever with useful coverage, not a generally reliable question-answering component. Before a future Qwen layer consumes it, add and pass real-network precision/abstention tests for unsupported hardware and unrelated questions, then calibrate or replace the relevance decision independently of answer generation.

## Remaining limitations

* Retrieval returns evidence only; it does not generate a factual answer.
* Framework-documentation discovery follows declared metadata tags. This can collect several framework sources for a multi-framework model, which is correctly attributed but increases ranking noise.
* The integration runner records dependency, loading, inference, hardware, and absent-question probes, but currently asserts only the primary task query. The JSON results must therefore be reviewed rather than interpreted as blanket proof of retrieval quality.
