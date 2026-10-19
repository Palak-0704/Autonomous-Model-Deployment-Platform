"""Convert RAG context into a structured Qwen deployment plan."""

import json
import re
from typing import Any

from .prompt import NOT_SPECIFIED_CODE_MARKER, build_code_prompt, build_prompt
from .qwen_client import generate_response


PLAN_KEYS = (
	"model",
	"environment",
	"dependencies",
	"installation",
	"loading",
	"inference",
	"special_requirements",
	"missing_information",
)

NOT_SPECIFIED = "Not specified in repository"


def _normalise(text: str) -> str:
	"""Normalise whitespace only, preserving the actual evidence content."""
	return " ".join(text.casefold().split())


def _is_supported(value: str, rag_context: str) -> bool:
	return _normalise(value) in _normalise(rag_context)


def unsupported_plan_values(plan: dict[str, Any], rag_context: Any) -> dict[str, Any]:
	"""Return generated plan values that are not directly present in RAG evidence."""
	context = rag_context if isinstance(rag_context, str) else json.dumps(rag_context, default=str)
	rejected: dict[str, Any] = {}
	for key, value in plan.items():
		if value == NOT_SPECIFIED:
			continue
		if isinstance(value, str):
			if not _is_supported(value, context):
				rejected[key] = value
		elif isinstance(value, list):
			unsupported = [item for item in value if not isinstance(item, str) or not _is_supported(item, context)]
			if unsupported:
				rejected[key] = unsupported
		else:
			rejected[key] = value
	return rejected


def evidence_only_plan(plan: dict[str, Any], rag_context: Any) -> dict[str, Any]:
	"""Replace every unsupported generated field with the required missing marker."""
	rejected = unsupported_plan_values(plan, rag_context)
	return {key: NOT_SPECIFIED if key in rejected else value for key, value in plan.items()}


def _parse_json(response: str) -> dict[str, Any]:
	"""Parse Qwen's JSON response, tolerating Markdown or a short preamble."""
	text = response.strip()
	text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
	try:
		value = json.loads(text)
	except json.JSONDecodeError as error:
		# Instruction models occasionally prefix an otherwise valid JSON object
		# with a brief sentence. Decode the first object rather than failing the
		# RAG-to-plan smoke test for formatting alone.
		start = text.find("{")
		if start < 0:
			raise ValueError("Qwen returned invalid JSON.") from error
		try:
			value, _ = json.JSONDecoder().raw_decode(text[start:])
		except json.JSONDecodeError as nested_error:
			raise ValueError("Qwen returned invalid JSON.") from nested_error
	if not isinstance(value, dict):
		raise ValueError("Qwen returned JSON that is not an object.")
	return {key: value.get(key, NOT_SPECIFIED) for key in PLAN_KEYS}


def create_deployment_plan(user_request: str, model_url: str, rag_context: Any) -> dict[str, Any]:
	"""Ask Qwen for a structured deployment plan grounded in RAG evidence."""
	_, plan = create_deployment_plan_with_response(user_request, model_url, rag_context)
	return plan


def create_deployment_plan_with_response(
	user_request: str, model_url: str, rag_context: Any
) -> tuple[str, dict[str, Any]]:
	"""Return Qwen's raw response and its parsed deployment plan."""
	prompt = build_prompt(user_request, model_url, rag_context)
	response = generate_response(prompt)
	return response, evidence_only_plan(_parse_json(response), rag_context)


def _code_is_supported(code: str, rag_context: Any) -> bool:
	"""Accept only a non-empty code answer copied from retrieved RAG evidence."""
	if not code.strip() or code.strip() == NOT_SPECIFIED_CODE_MARKER:
		return True
	context = rag_context if isinstance(rag_context, str) else json.dumps(rag_context, default=str)
	return _normalise(code) in _normalise(context)


def create_evidence_only_code(user_request: str, model_url: str, rag_context: Any) -> str:
	"""Return Qwen-selected repository code, or the missing marker if unsupported."""
	response = generate_response(build_code_prompt(user_request, model_url, rag_context))
	code = re.sub(r"^```(?:python)?\s*|\s*```$", "", response.strip(), flags=re.IGNORECASE).strip()
	return code if _code_is_supported(code, rag_context) else NOT_SPECIFIED_CODE_MARKER
