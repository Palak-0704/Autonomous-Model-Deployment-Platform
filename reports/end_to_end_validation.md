# End-to-end workflow validation

Date: 2026-10-19 (Asia/Kolkata)

## Changes made

* Retrieval now indexes list-valued evidence as individual attributed snippets,
  preventing one match from returning an entire mixed fact collection.
* The Qwen RAG adapter now limits repeated provenance records in the bounded
  prompt payload.
* Deployment plans exclude packages that appear only in official framework
  documentation. The safe Transformers runtime explicitly installs
  `transformers` and `torch`.
* Smoke tests and inference now run through the session virtual environment,
  with `trust_remote_code=False`, instead of loading the model in the parent
  process.

## Executed checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s Rag\tests -p "test*.py" -v
.\.venv\Scripts\python.exe -m Rag.retrieval.test_semantic_retriever
.\.venv\Scripts\python.exe -m deployment.test_deployment
.\.venv\Scripts\python.exe -m qwen.qwen_client_tests
.\.venv\Scripts\python.exe -m qwen.rag_loop_tests
.\.venv\Scripts\python.exe -m qwen.evidence_tests
.\.venv\Scripts\python.exe -m qwen.test_ui
```

Results: 45 RAG tests passed. The semantic-retriever, deployment-plan, Qwen
model-reuse, duplicate-loop, evidence-guard, and Qwen UI callback checks all
passed. These are local/offline checks unless explicitly noted below.

## Colab Sync and Qwen GPU evidence

The existing `.colabsync/config.json` remains configured for
`/content/Project -2`; no separate synchronization mechanism was introduced.
The local sync manifest was last updated on 2026-10-07, so it is not evidence
that this validation's changes have reached Colab.

The existing notebook records a real Tesla T4 run: CUDA was available, Qwen
weights loaded, the RAG collected five documents for
`Zero-Vision/Llama-3-MixSense`, and Qwen asked two retrieval questions.
That run ended in `KeyboardInterrupt` while generating the final deployment
plan. It is therefore evidence of partial GPU execution, **not** a successful
end-to-end Qwen pipeline run.

## Real local deployment attempt

The safe deployment flow was started for `sshleifer/tiny-gpt2`. Inspection
succeeded and produced an isolated-session plan using only `transformers` and
`torch`. The session virtual environment was created and dependency
installation began. At the time this record was written, the installer had
not completed and there is no `SUCCESS` deployment report, model-load result,
or inference output. Do not treat the model deployment as validated yet.

## Required next verification

1. Let the isolated `tiny-gpt2` dependency install finish, then record a
   successful smoke test and inference output.
2. Open the existing synchronized Colab workspace, confirm the new hashes are
   synchronized, and rerun the existing notebook's terminal-pipeline cell on
   a GPU. Save its complete output only if the final validated plan is printed
   without an exception.
3. Run real-network RAG integration checks after the above changes and review
   unsupported-question abstention as well as task-query recall.
