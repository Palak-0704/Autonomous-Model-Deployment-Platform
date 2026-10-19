# RAG verification report

Date: 2026-10-14. Scope: `Rag/` only. No target-model weights or repository scripts were executed.

## Inspected files

- `Rag/ingestion/loader.py`, `Rag/ingestion/github.py`, `Rag/config.py`
- `Rag/extraction/extractor.py`, `Rag/retrieval/retriever.py`, `Rag/test.py`
- Existing executable checks: `Rag/retrieval/test_semantic_retriever.py` and `Rag/retrieval/test_gpt2_semantic.py`
- Added automated tests: `Rag/tests/test_ingestion_and_extraction.py`, `Rag/tests/test_retrieval.py`; real-network runner: `Rag/tests/integration_real.py`.

## Defects found and fixed

1. Linked GitHub repositories were never collected. The loader now finds explicit GitHub URLs in Hugging Face metadata/model-card text, reads public repository metadata plus selected README, manifest, configuration, and example/source files, and prefixes their paths with `github/`. It does not execute code or fetch weights/binaries. Unavailable GitHub links do not fail Hugging Face ingestion.
2. A matched retrieval result included every extracted item from the same source file. Results now return only the matched provenance snippet.
3. Structured questions such as task, dependency, loading, inference, and hardware questions could rank unrelated semantic evidence first. Provenance-backed structured lookup now takes precedence.
4. An unrelated question (`What is the weather forecast?`) returned arbitrary repository evidence. A no-meaningful-term guard now returns no evidence.
5. CUDA/device lines in examples were incorrectly extracted as hardware requirements. Only explicit requirement language is accepted; a `.to("cuda")` example alone is not a requirement.
6. Conflicting dependency versions were silently overwritten. `ModelKnowledge.conflicts` now retains each distinct version.
7. `Rag/test.py` now supports direct execution as well as module imports, and a dedicated `Rag/tests/` suite makes project-root unittest discovery work. The Gradio interface now displays collected paths, evidence, and retrieval results as JSON in addition to its concise result view.

## Commands executed and results

| Command | Kind | Result |
|---|---|---|
| `./.venv/Scripts/python.exe -m unittest discover -s . -p "test*.py" -v` | Mocked/unit | PASS: 9 tests |
| `./.venv/Scripts/python.exe -m Rag.retrieval.test_semantic_retriever` | Mocked/unit | PASS |
| Gradio construction + invalid-URL callback check | Local UI | PASS |
| Mocked Hugging Face/GitHub network-failure checks | Unit | PASS |
| `./.venv/Scripts/python.exe -m Rag.tests.integration_real` | Real network integration | PASS: 2/2 task retrieval checks |

## Real integration evidence and judgments

| Repository | Expected answer | Retrieved answer/evidence | Judgment |
|---|---|---|---|
| `distilbert/distilroberta-base` | `fill-mask` | `task` evidence: `fill-mask` | Correct |
| `openai/whisper-tiny` | `automatic-speech-recognition` | `task` evidence: `automatic-speech-recognition` | Correct |

For both repositories, the actual run retrieved `transformers` and `torch` for dependency questions and repository-provided `from_pretrained`/pipeline examples for loading and inference questions. Both model cards exposed an explicit GitHub repository; the run collected GitHub README/manifests and example/source paths with `github/` provenance. The full raw evaluation is in `reports/rag_verification/real_integration.json`.

Negative-question metric: 2/2 repositories returned no result for `What is the weather forecast?` after the fix. Hardware-requirement results are deliberately conservative: Whisper returned no verified requirement; DistilRoBERTa returned generic linked-project CPU/GPU documentation, which is not a model-specific requirement and should not be treated as one.

## Answering limitation

There is no answer-generating component in this RAG. `Rag/test.py` presents retrieved, attributed evidence; it does **not** synthesize a natural-language answer. Therefore factual correctness above is judged against the returned evidence, not an LLM answer. No Qwen or other LLM was added.

## Remaining limitations

- GitHub selection is capped at 40 relevant text/source files but broad upstream repositories can still contribute generic documentation; provenance is preserved so callers can distinguish it.
- GitHub API unauthenticated rate limits can prevent optional linked-repository collection; Hugging Face ingestion remains available.
- Semantic retrieval needs `sentence-transformers`; its embedding model is loaded lazily. The integration run used the existing MiniLM retriever model, not a target model.
- The extractor records dependency-version conflicts. It does not yet perform a general contradiction analysis for every arbitrary prose field.
