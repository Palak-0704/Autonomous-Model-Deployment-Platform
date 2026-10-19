"""Small Gradio UI for manually testing the complete RAG pipeline."""

from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any

import gradio as gr

try:  # supports both ``python -m Rag.test`` and direct local launching
	from .extraction.extractor import extract_knowledge
	from .ingestion.loader import HuggingFaceAuthenticationError, HuggingFaceRepositoryError, load_repository
	from .retrieval.retriever import KnowledgeRetriever
except ImportError:  # pragma: no cover - direct script convenience only
	from Rag.extraction.extractor import extract_knowledge
	from Rag.ingestion.loader import HuggingFaceAuthenticationError, HuggingFaceRepositoryError, load_repository
	from Rag.retrieval.retriever import KnowledgeRetriever


DOCUMENTS_DIR = Path(__file__).parent / "documents"


def _safe_filename(model_name: str) -> str:
	name = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name or "unknown-model")
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
	return f"{name}_{timestamp}.txt"


def _format(value: Any) -> str:
	return json.dumps(value, indent=2, ensure_ascii=True, default=str)


def _format_results(results: list[dict[str, Any]]) -> str:
	"""Format only retrieved values for the user-facing answer."""
	if not results:
		return "Information not found in available sources"

	lines = []
	for index, item in enumerate(results, start=1):
		lines.extend([
			f"{index}. Field: {item['field']}",
			f"   Information: {item['value']}",
		])
	return "\n".join(lines)


FIELD_LABELS = {
	"model_name": "Model name", "model_id": "Model ID", "task": "Task",
	"pipeline_type": "Pipeline", "architecture": "Architecture", "model_type": "Model type",
	"primary_framework": "Primary framework", "framework": "Framework",
	"alternative_frameworks": "Alternative frameworks", "library": "Library",
	"available_formats": "Available formats", "required_format": "Required format",
	"recommended_format": "Recommended format", "model_format": "Model format",
	"quantization": "Quantization", "parameter_count": "Parameter count", "model_size": "Model size",
	"python_requirements": "Python requirements", "pytorch_requirements": "PyTorch requirements",
	"tensorflow_requirements": "TensorFlow requirements", "cuda_requirements": "CUDA requirements",
	"gpu_cpu_requirements": "GPU/CPU requirements", "hardware_requirements": "Hardware requirements",
	"dependencies": "Dependencies", "dependency_versions": "Dependency versions",
	"optional_dependencies": "Optional dependencies", "system_dependencies": "System dependencies",
	"conflicts": "Conflicting declarations",
	"installation_commands": "Installation commands", "environment_setup_commands": "Environment setup",
	"loading_code": "Loading code", "loading_methods": "Loading method",
	"tokenizer_processor_requirements": "Tokenizer/processor", "required_files": "Required files",
	"configuration_requirements": "Configuration", "input_format": "Input",
	"output_format": "Output", "inference_code": "Inference code",
	"inference_methods": "Inference method", "inference_examples": "Inference example",
	"generation_parameters": "Generation parameters", "special_requirements": "Special requirements",
	"warnings": "Warnings", "limitations": "Limitations",
	"repository_instructions": "Repository instructions",
}

FIELD_GROUPS = (
	("MODEL", ("model_name", "model_id", "task", "pipeline_type", "architecture", "model_type", "primary_framework", "framework", "alternative_frameworks", "library", "available_formats", "required_format", "recommended_format", "model_format", "quantization")),
	("ENVIRONMENT", ("python_requirements", "pytorch_requirements", "tensorflow_requirements", "cuda_requirements", "gpu_cpu_requirements", "hardware_requirements", "model_size", "parameter_count")),
	("DEPENDENCIES", ("dependencies", "dependency_versions", "conflicts", "optional_dependencies", "system_dependencies")),
	("INSTALLATION", ("installation_commands", "environment_setup_commands", "repository_instructions")),
	("LOADING", ("loading_code", "loading_methods", "tokenizer_processor_requirements", "required_files", "configuration_requirements")),
	("INFERENCE", ("input_format", "output_format", "inference_code", "inference_methods", "inference_examples", "generation_parameters")),
	("SPECIAL REQUIREMENTS", ("special_requirements", "warnings", "limitations")),
)

ACTIVE_SECTION_GAPS = {
	"MODEL": {
		"available_formats": "Available formats",
		"required_format": "Required format",
		"recommended_format": "Recommended format",
	},
	"ENVIRONMENT": {
		"python_requirements": "Python requirements",
		"pytorch_requirements": "PyTorch requirements",
		"tensorflow_requirements": "TensorFlow requirements",
		"cuda_requirements": "CUDA requirements",
		"gpu_cpu_requirements": "GPU/CPU requirements",
		"hardware_requirements": "Hardware requirements",
	},
	"DEPENDENCIES": {
		"dependency_versions": "Dependency versions",
	},
	"INSTALLATION": {
		"installation_commands": "Installation commands",
		"environment_setup_commands": "Environment setup",
	},
	"LOADING": {
		"required_files": "Required files",
		"configuration_requirements": "Configuration",
		"loading_code": "Loading code",
		"tokenizer_processor_requirements": "Tokenizer/processor",
	},
	"INFERENCE": {
		"input_format": "Input",
		"output_format": "Output",
		"inference_code": "Inference code",
	},
}


def _display_value(value: Any, field: str) -> str:
	if isinstance(value, dict):
		text = "\n".join(f"{key}: {item}" for key, item in value.items())
	elif isinstance(value, list):
		separator = "\n" if field.endswith(("_code", "_commands", "_methods", "_examples")) else "; "
		text = separator.join(str(item) for item in value)
	else:
		text = str(value)
	if field.endswith(("_code", "_commands", "_methods", "_examples")):
		return text
	return text if len(text) <= 900 else text[:897].rstrip() + "..."


def _deployment_context(knowledge: dict[str, Any]) -> str:
	"""Render only deployment facts actually extracted from this repository."""
	evidence = knowledge.get("evidence", {})
	lines = []
	for section, fields in FIELD_GROUPS:
		present = [field for field in fields if knowledge.get(field) and evidence.get(field)]
		if not present:
			continue
		lines.append(section)
		for field in present:
			value = _display_value(knowledge[field], field)
			sources = []
			for item in evidence.get(field, []):
				path = item.get("path") or item.get("source", "unknown source")
				if path not in sources:
					sources.append(path)
			if not sources:
				continue
			source_text = ", ".join(sources)
			lines.append(f"{FIELD_LABELS.get(field, field)}: {value}")
			lines.append(f"  Evidence: {source_text}")
		for field, label in ACTIVE_SECTION_GAPS.get(section, {}).items():
			if field not in present:
				lines.append(f"{label}: Not specified in repository")
		lines.append("")
	return "\n".join(lines).strip() or "Not specified in repository"


def run_test(repository_url: str, query: str, access_token: str = "") -> tuple[str, str, str]:
	"""Run URL -> loader -> extractor -> retriever and save the complete result."""
	result: dict[str, Any] = {
		"url": repository_url,
		"query": query,
		"collected_files": [],
		"extracted_knowledge": {},
		"evidence": {},
		"retrieval_results": [],
		"errors": [],
	}
	try:
		documents = load_repository(repository_url.strip(), token=access_token.strip())
		result["collected_files"] = [document.path or document.document_type for document in documents]
		knowledge = extract_knowledge(documents, repository_url=repository_url.strip())
		knowledge_data = knowledge.to_dict()
		result["extracted_knowledge"] = knowledge_data
		result["evidence"] = knowledge_data.get("evidence", {})
		result["retrieval_results"] = KnowledgeRetriever(knowledge).retrieve(query.strip()) if query.strip() else []
	except HuggingFaceAuthenticationError:
		result["errors"].append("Hugging Face authentication required. Please provide a valid Hugging Face Access Token.")
	except HuggingFaceRepositoryError as error:
		result["errors"].append(str(error))
	except ValueError as error:
		result["errors"].append(str(error))
	except Exception:
		result["errors"].append("An unexpected RAG error occurred while reading the repository.")

	DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
	output_path = DOCUMENTS_DIR / _safe_filename(result["extracted_knowledge"].get("model_name", "unknown-model"))
	output_path.write_text(_format(result), encoding="utf-8")
	if result["errors"]:
		message = result["errors"][0]
		return message, message, _format(result)
	return (
		_deployment_context(result["extracted_knowledge"]),
		_format_results(result["retrieval_results"]),
		_format({"collected_files": result["collected_files"], "evidence": result["evidence"], "retrieval_results": result["retrieval_results"]}),
	)


def create_app() -> gr.Blocks:
	"""Build the manual RAG test interface."""
	with gr.Blocks(title="RAG Test") as app:
		gr.Markdown("# Hugging Face RAG Test")
		with gr.Row():
			repository_url = gr.Textbox(
				label="Hugging Face model/repository URL",
				value="https://huggingface.co/openai-community/gpt2",
			)
			query = gr.Textbox(label="Query", value="How do I load and run this model?")
		access_token = gr.Textbox(
			label="Hugging Face Access Token (optional)",
			placeholder="Required only for private or gated repositories",
			type="password",
		)
		button = gr.Button("Authenticate / Run")
		deployment_context = gr.Textbox(label="Deployment Context", lines=28)
		answer = gr.Textbox(label="Relevant Retrieved Information", lines=10)
		details = gr.Code(label="Collected files, supporting evidence, and retrieval results", language="json")
		button.click(
			fn=run_test,
			inputs=[repository_url, query, access_token],
			outputs=[deployment_context, answer, details],
		)
	return app


if __name__ == "__main__":
	create_app().launch(footer_links=[], run_history=False)
