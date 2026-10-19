"""Minimal evidence-only Qwen <-> existing RAG question-answer loop."""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from Rag.extraction.extractor import extract_knowledge
from Rag.ingestion.loader import load_repository
from Rag.retrieval.retriever import KnowledgeRetriever

from .decision import create_deployment_plan_with_response
from .prompt import build_rag_question_prompt
from .qwen_client import generate_response


NOT_SPECIFIED = "Not specified in repository"
MAX_QUESTIONS = 8
MAX_RAG_ANSWER_CHARS = 1_800
MAX_COLLECTED_EVIDENCE_CHARS = 6_000


def _question_key(question: str) -> str:
    """Canonical form used only to prevent repeat RAG requests."""
    return re.sub(r"[^a-z0-9]+", " ", question.casefold()).strip()


@dataclass
class RepositoryRag:
    """Small adapter that exposes the existing RAG retrieval as ask(question)."""

    repository_url: str
    retriever: KnowledgeRetriever
    asked_questions: set[str] = field(default_factory=set)

    @classmethod
    def from_repository(cls, repository_url: str) -> "RepositoryRag":
        print(f"\n========== RAG LOADING ==========")
        print(f"Repository: {repository_url}")
        documents = load_repository(repository_url)
        print(f"Documents collected: {len(documents)}")
        knowledge = extract_knowledge(documents, repository_url=repository_url)
        print(f"\n========== EXTRACTED KNOWLEDGE ==========")
        # The extractor now returns a single ModelKnowledge object.
        # We need to iterate over its 'evidence' dict instead of iterating over the object itself.
        for field_name, evidence_list in knowledge.evidence.items():
             print(f"FIELD: {field_name} | COUNT: {len(evidence_list)} | SAMPLE: {str(evidence_list[0].get('value'))[:100]}...")
        return cls(repository_url=repository_url, retriever=KnowledgeRetriever(knowledge))

    def ask(self, question: str) -> str:
        """Return only retrieved repository evidence for one new question."""
        print(f"\n========== QWEN QUESTION ==========")
        print(f"Question: {question}")
        
        key = _question_key(question)
        if not question.strip() or key in self.asked_questions:
            return NOT_SPECIFIED
        self.asked_questions.add(key)
        
        results = self.retriever.retrieve(question, top_k=1)
        
        print(f"\n========== RAG RESPONSE ==========")
        if not results:
            print("NO RETRIEVED EVIDENCE: No matching evidence found above retrieval threshold.")
            return NOT_SPECIFIED
        
        result = results[0]
        print(f"Question: {question}")
        print(f"Retrieved evidence: {result.get('value')[:200]}...")
        print(f"Source: {result.get('source')}")
        print(f"Similarity score: {result.get('score', 'N/A')}")
        
        payload = {
            "field": result["field"],
            "value": result["value"][:MAX_RAG_ANSWER_CHARS],
            # A result has one directly supporting snippet.  Keeping only a
            # few source records prevents provenance repetition from crowding
            # out evidence in the bounded Qwen prompt.
            "source": [
                {"source": item.get("source", ""), "path": item.get("path", "")}
                for item in result["source"][:3]
            ],
        }
        return json.dumps(payload, indent=2, ensure_ascii=True)


def _bounded_evidence(evidence_parts: list[str]) -> str:
    """Keep the newest evidence window without duplicating entire RAG results."""
    collected = "\n\n".join(evidence_parts)
    if len(collected) <= MAX_COLLECTED_EVIDENCE_CHARS:
        return collected
    return collected[-MAX_COLLECTED_EVIDENCE_CHARS:]


def _next_action(model_url: str, user_request: str, collected_evidence: str) -> tuple[str, str]:
    """Parse Qwen's next-question decision without treating it as evidence."""
    print(f"\n========== EVIDENCE SENT TO QWEN ==========")
    print(f"Evidence sent: {collected_evidence}")
    
    print(f"[4] Qwen request started at {time.time()}")
    request_context = f"USER REQUEST:\n{user_request}\n\n{collected_evidence}"
    raw = generate_response(build_rag_question_prompt(model_url, request_context), max_new_tokens=128)
    print(f"[5] Qwen response received at {time.time()}")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE).strip()
    
    print(f"\n========== QWEN DECISION ==========")
    try:
        value = json.loads(text)
        print(f"Decision: {value.get('action')}")
        print(f"Reason: {value.get('reason')}")
        print(f"Question: {value.get('question')}")
    except:
        print("Failed to parse Qwen JSON response.")
        print(f"Raw response: {raw}")

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            return "final", ""
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            return "final", ""
    if not isinstance(value, dict) or value.get("action") == "final":
        return "final", ""
    question = value.get("question")
    if value.get("action") != "ask" or not isinstance(question, str) or not question.strip():
        return "final", ""
    return "ask", question.strip()


def run_question_answer_loop(
    repository_url: str,
    user_request: str = "Create an evidence-only deployment and testing plan.",
    max_questions: int = MAX_QUESTIONS,
) -> tuple[list[tuple[str, str]], str, str, dict[str, Any]]:
    """Run Qwen questions against RAG, then return raw response and safe plan."""
    rag = RepositoryRag.from_repository(repository_url)
    interaction: list[tuple[str, str]] = []
    evidence_parts: list[str] = []
    asked_keys: set[str] = set()
    duplicate_attempts = 0
    completion_reason = "safety limit"
    for _ in range(max_questions):
        action, question = _next_action(repository_url, user_request, _bounded_evidence(evidence_parts))
        if action == "final":
            completion_reason = "Qwen decided enough information was collected"
            break
        key = _question_key(question)
        if key in asked_keys:
            # Never resend a duplicate to RAG. Feed the unavailable state back
            # to Qwen once, then complete normally if it repeats again.
            duplicate_attempts += 1
            answer = NOT_SPECIFIED
            interaction.append((question, answer))
            evidence_parts.append(
                f"QWEN QUESTION (DUPLICATE, NOT SENT TO RAG):\n{question}\n"
                f"RAG ANSWER:\n{answer}\n"
                "CONTROL: This question is already unavailable. Ask a different topic or return final."
            )
            if duplicate_attempts >= 2:
                completion_reason = "controller completed after repeated duplicate questions"
                break
            continue
        asked_keys.add(key)
        answer = rag.ask(question)
        interaction.append((question, answer))
        evidence_parts.append(f"QWEN QUESTION:\n{question}\nRAG ANSWER:\n{answer}")

    final_evidence = _bounded_evidence(evidence_parts) or NOT_SPECIFIED
    raw_response, plan = create_deployment_plan_with_response(
        user_request, repository_url, final_evidence
    )
    return interaction, completion_reason, raw_response, plan
