"""Extract deployment-oriented knowledge from Hugging Face repository documents."""

from dataclasses import asdict, dataclass, field
import json
import re
import sys
import tomllib
from typing import Any, Iterable

from packaging.requirements import InvalidRequirement, Requirement

from ..ingestion.loader import Document

# Prose snippets indexed per document. A model's own README is the primary evidence
# source, so it gets a larger window than linked or framework documents.
MODEL_DOCUMENT_SNIPPET_LIMIT = 40
DEFAULT_DOCUMENT_SNIPPET_LIMIT = 8


@dataclass
class ModelKnowledge:
	model_name: str = ""
	model_id: str = ""
	repository_url: str = ""
	architecture: list[str] = field(default_factory=list)
	model_type: str = ""
	task: str = ""
	pipeline_type: str = ""
	framework: list[str] = field(default_factory=list)
	primary_framework: str = ""
	alternative_frameworks: list[str] = field(default_factory=list)
	library: str = ""
	model_format: list[str] = field(default_factory=list)
	available_formats: list[str] = field(default_factory=list)
	required_format: str = ""
	recommended_format: str = ""
	parameter_count: str = ""
	model_size: str = ""
	quantization: list[str] = field(default_factory=list)
	python_requirements: list[str] = field(default_factory=list)
	pytorch_requirements: list[str] = field(default_factory=list)
	tensorflow_requirements: list[str] = field(default_factory=list)
	cuda_requirements: list[str] = field(default_factory=list)
	gpu_cpu_requirements: list[str] = field(default_factory=list)
	hardware_requirements: list[str] = field(default_factory=list)
	dependencies: list[str] = field(default_factory=list)
	dependency_versions: dict[str, str] = field(default_factory=dict)
	conflicts: dict[str, list[str]] = field(default_factory=dict)
	optional_dependencies: list[str] = field(default_factory=list)
	development_dependencies: list[str] = field(default_factory=list)
	system_dependencies: list[str] = field(default_factory=list)
	installation_commands: list[str] = field(default_factory=list)
	environment_setup_commands: list[str] = field(default_factory=list)
	loading_code: list[str] = field(default_factory=list)
	inference_code: list[str] = field(default_factory=list)
	loading_methods: list[str] = field(default_factory=list)
	tokenizer_processor_requirements: list[str] = field(default_factory=list)
	inference_methods: list[str] = field(default_factory=list)
	inference_examples: list[str] = field(default_factory=list)
	input_format: list[str] = field(default_factory=list)
	output_format: list[str] = field(default_factory=list)
	generation_parameters: dict[str, str] = field(default_factory=dict)
	required_files: list[str] = field(default_factory=list)
	configuration_requirements: list[str] = field(default_factory=list)
	special_requirements: list[str] = field(default_factory=list)
	limitations: list[str] = field(default_factory=list)
	warnings: list[str] = field(default_factory=list)
	repository_instructions: list[str] = field(default_factory=list)
	evidence: dict[str, list[dict[str, str]]] = field(default_factory=dict)
	source_documents: list[str] = field(default_factory=list)
	model_specific_requirements: list[str] = field(default_factory=list)
	framework_compatibility: list[str] = field(default_factory=list)
	inference_hardware: list[str] = field(default_factory=list)
	training_hardware: list[str] = field(default_factory=list)
	document_evidence: list[str] = field(default_factory=list)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


def _clean(value: Any) -> str:
	return " ".join(str(value).strip().split())


def _unique(values: Iterable[str]) -> list[str]:
	return list(dict.fromkeys(value for value in (_clean(item) for item in values) if value))


def _short(value: Any, limit: int = 600) -> str:
	text = _clean(value)
	return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _summary(value: Any, limit: int = 260) -> str:
	text = _short(value, limit)
	return re.split(r"(?<=[.!?])\s+", text)[0]


def _document_name(document: Document) -> str:
	return (document.path or document.document_type).replace("\\", "/").lower()


def _metadata(documents: list[Document]) -> tuple[dict[str, Any], Document | None]:
	for document in documents:
		if document.document_type == "model_metadata":
			try:
				return json.loads(document.content), document
			except json.JSONDecodeError:
				return {}, document
	return {}, None


def _json_documents(documents: list[Document]) -> dict[str, tuple[dict[str, Any], Document]]:
	result = {}
	for document in documents:
		name = _document_name(document)
		if not name.endswith(".json") or document.document_type == "model_metadata":
			continue
		try:
			value = json.loads(document.content)
		except json.JSONDecodeError:
			continue
		if isinstance(value, dict):
			result[name] = (value, document)
	return result


def _evidence(knowledge: ModelKnowledge, field_name: str, value: Any, document: Document) -> None:
	if not value:
		return
	knowledge.evidence.setdefault(field_name, []).append(
		{"source": document.source, "path": document.path or document.document_type, "origin": document.origin, "value": _clean(value)}
	)


def _set(knowledge: ModelKnowledge, field_name: str, value: Any, document: Document) -> None:
	if isinstance(value, list):
		value = _unique(value)
	if not value:
		return
	setattr(knowledge, field_name, value)
	_evidence(knowledge, field_name, value, document)


def _append(knowledge: ModelKnowledge, field_name: str, values: Iterable[str], document: Document) -> None:
	if field_name.endswith(("_code", "_commands", "_methods", "_examples")):
		new_values = list(dict.fromkeys(value.strip() for value in values if value and value.strip()))
	else:
		new_values = _unique(values)
	if not new_values:
		return
	current = getattr(knowledge, field_name)
	for value in new_values:
		if value not in current:
			current.append(value)
			_evidence(knowledge, field_name, value, document)


def _headings(text: str) -> list[tuple[str, str]]:
	pattern = re.compile(r"(?ms)^\s*(#{1,6})\s+(.+?)\s*$")
	matches = list(pattern.finditer(text))
	return [
		(_clean(match.group(2)).lower(), text[match.end() : matches[index + 1].start() if index + 1 < len(matches) else len(text)].strip())
		for index, match in enumerate(matches)
	]


def _sections(text: str, keywords: Iterable[str]) -> list[str]:
	keywords = tuple(keyword.lower() for keyword in keywords)
	return [_summary(body) for title, body in _headings(text) if any(keyword in title for keyword in keywords)]


def _code_blocks(text: str) -> list[str]:
	return [block.strip() for block in re.findall(r"(?ms)```(?:\w+)?\s*\n(.*?)```", text)]


def _document_snippets(text: str, limit: int = 8) -> list[str]:
	"""Small general-purpose evidence chunks for questions outside fixed fields."""
	chunks = []
	for part in re.split(r"\n\s*\n|(?<=[.!?])\s+(?=[A-Z])", text):
		cleaned = _clean(part)
		if 40 <= len(cleaned) <= 700 and cleaned not in chunks:
			chunks.append(cleaned)
			if len(chunks) >= limit:
				break
	return chunks


def _clean_code(block: str, limit: int = 900) -> str:
	lines = []
	for line in block.splitlines():
		line = re.sub(r"^(?:>>>|\.\.\.)\s*", "", line.strip())
		if not line or line.startswith("#"):
			continue
		if "bibtex" in line.lower() or line.startswith(("author=", "title=", "year=")):
			continue
		lines.append(line)
	return "\n".join(lines)[:limit].rstrip()


def _code_lines(block: str, keywords: tuple[str, ...], limit: int = 700) -> str:
	lines = []
	for raw_line in block.splitlines():
		line = re.sub(r"^(?:>>>|\.\.\.)\s*", "", raw_line.strip())
		if line and not line.startswith("#"):
			lines.append(line)
	selected = [line for line in lines if any(keyword in line.lower() for keyword in keywords)]
	return "\n".join(selected)[:limit].rstrip()


def _code_examples(blocks: Iterable[str], keywords: tuple[str, ...]) -> list[str]:
	"""Keep complete relevant code blocks, rather than selected fragments."""
	results = []
	for block in blocks:
		cleaned = _clean_code(block, limit=20_000)
		if cleaned and any(keyword in cleaned.lower() for keyword in keywords):
			if cleaned not in results:
				results.append(cleaned)
	return results[:3]


def _import_dependencies(text: str) -> list[str]:
	packages = []
	for match in re.finditer(r"(?m)^\s*(?:from|import)\s+([A-Za-z_][\w.-]*)", text):
		package = match.group(1).split(".", 1)[0]
		# Standard-library modules are never installable dependencies.
		if package not in sys.stdlib_module_names:
			if package not in packages:
				packages.append(package)
	return packages


def _commands(text: str, prefixes: tuple[str, ...]) -> list[str]:
	return [
		line.strip()
		for line in text.splitlines()
		if line.strip().lower().startswith(prefixes)
	]


def _requirement_lines(document: Document) -> tuple[list[str], dict[str, str]]:
	name = _document_name(document)
	if name.endswith("pyproject.toml"):
		try:
			project = tomllib.loads(document.content).get("project", {})
			values = list(project.get("dependencies", []))
			for optional in project.get("optional-dependencies", {}).values():
				values.extend(optional)
			return _parse_package_specs(values)
		except (tomllib.TOMLDecodeError, AttributeError):
			return [], {}
	if name.endswith("setup.py"):
		match = re.search(r"install_requires\s*=\s*\[(.*?)\]", document.content, re.S)
		return _parse_package_specs(re.findall(r"[\"']([^\"']+)[\"']", match.group(1))) if match else ([], {})
	lines = [line.strip() for line in document.content.splitlines() if line.strip() and not line.lstrip().startswith(('#', ';'))]
	if name.endswith("environment.yml"):
		lines = [re.sub(r"^[- ]+", "", line) for line in lines if re.match(r"^\s*-\s*[A-Za-z]", line)]
	return _parse_package_specs(lines)


def _parse_package_specs(specifications: Iterable[str]) -> tuple[list[str], dict[str, str]]:
	packages = []
	versions = {}
	for line in specifications:
		line = line.strip()
		if line.startswith(("- ", "--")) or line.startswith(("pip", "conda")):
			continue
		try:
			requirement = Requirement(line)
		except InvalidRequirement:
			continue
		package = requirement.name.lower()
		packages.append(package)
		# Keep the source spelling/order of the constraint. ``Requirement`` only
		# validates and identifies the package; it must not rewrite evidence.
		constraint = line[len(requirement.name):].strip()
		if constraint:
			versions[package] = constraint
	return packages, versions


def _extract_manifest(knowledge: ModelKnowledge, document: Document) -> None:
	name = _document_name(document)
	packages, versions = _requirement_lines(document)
	if not packages:
		return
	if "requirements-dev" in name or name.endswith(("dev-requirements.txt", "dev.txt")):
		_append(knowledge, "development_dependencies", packages, document)
	else:
		_append(knowledge, "dependencies", packages, document)
	for package, version in versions.items():
		_record_dependency_version(knowledge, package, version)
		_evidence(knowledge, "dependency_versions", f"{package}{version}", document)
		if package == "python":
			_append(knowledge, "python_requirements", [f"python{version}"], document)


def _record_dependency_version(knowledge: ModelKnowledge, package: str, version: str) -> None:
	"""Keep incompatible manifest declarations visible rather than overwriting one."""
	previous = knowledge.dependency_versions.get(package)
	if previous and previous != version:
		values = knowledge.conflicts.setdefault(f"dependency_versions.{package}", [previous])
		if version not in values:
			values.append(version)
		return
	knowledge.dependency_versions[package] = version


def _extract_code(knowledge: ModelKnowledge, text: str, document: Document) -> None:
	blocks = _code_blocks(text)
	loading = _code_examples(blocks, ("from_pretrained", "load_model", "pipeline(", "llamacpp", "vllm"))
	inference = _code_examples(blocks, ("pipeline(", "generate(", "predict(", "encode(", "model(", "output", "logits", "embedding", "transcription"))
	_append(knowledge, "loading_code", loading, document)
	_append(knowledge, "inference_code", inference, document)
	imported_dependencies = _import_dependencies("\n".join(blocks))
	known_dependencies = {
		str(package).lower().replace("_", "-")
		for package in knowledge.dependencies
	}
	new_dependencies = [
		package for package in imported_dependencies
		if package.lower().replace("_", "-") not in known_dependencies
	]
	_append(knowledge, "dependencies", new_dependencies, document)
	unique_dependencies = []
	seen_dependencies = set()
	for package in knowledge.dependencies:
		key = package.lower().replace("_", "-")
		if key not in seen_dependencies:
			seen_dependencies.add(key)
			unique_dependencies.append(package)
	knowledge.dependencies = unique_dependencies
	processor_lines = [
		line.strip()
		for block in blocks
		for line in block.splitlines()
		if re.search(r"(?:tokenizer|processor|feature_extractor|image_processor|sampling_rate)", line, re.I)
	]
	_append(knowledge, "tokenizer_processor_requirements", processor_lines, document)
	input_lines = []
	output_lines = []
	for block in blocks:
		input_lines.extend(re.sub(r"^(?:>>>|\.\.\.)\s*", "", line.strip()) for line in block.splitlines() if any(token in line.lower() for token in ("tokenizer(", "processor(", "input", "sentences", "audio", "image", "prompt")))
		output_lines.extend(re.sub(r"^(?:>>>|\.\.\.)\s*", "", line.strip()) for line in block.splitlines() if any(token in line.lower() for token in ("output", "logits", "generated", "embedding", "transcription", "prediction")) and "generated_text" not in line.lower())
	_append(knowledge, "input_format", [_clean_code("\n".join(input_lines), 400)], document)
	_append(knowledge, "output_format", [_clean_code("\n".join(output_lines), 400)], document)
	for block in blocks:
		install_commands = _commands(block, ("pip install", "python -m pip install", "conda install", "mamba install"))
		_append(knowledge, "installation_commands", install_commands, document)
		_extract_install_packages(knowledge, install_commands, document)
		_append(knowledge, "environment_setup_commands", _commands(block, ("conda create", "python -m venv", "uv venv", "source ")), document)
	if loading:
		_append(knowledge, "loading_methods", ["Repository provides a model loading example."], document)
	if inference:
		_append(knowledge, "inference_methods", ["Repository provides an inference example."], document)


def _extract_install_packages(knowledge: ModelKnowledge, commands: Iterable[str], document: Document) -> None:
	for command in commands:
		parts = command.split()
		try:
			install_index = next(index for index, part in enumerate(parts) if part.lower() == "install")
		except StopIteration:
			continue
		for part in parts[install_index + 1 :]:
			if part.startswith("-") or "/" in part or part.startswith("git"):
				continue
			match = re.match(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)([<>=!~]+.*)?$", part)
			if match:
				package = match.group(1).lower()
				_append(knowledge, "dependencies", [package], document)
				if match.group(2):
					_record_dependency_version(knowledge, package, match.group(2))
					_evidence(knowledge, "dependency_versions", f"{package}{match.group(2)}", document)


def _add_runtime_dependencies(knowledge: ModelKnowledge, document: Document) -> None:
	"""Add only runtime packages supported by explicit metadata or imports."""
	known = {"transformers", "torch", "tensorflow", "jax", "flax", "keras", "sentence-transformers", "diffusers", "onnxruntime", "vllm"}
	text = document.content.lower()
	packages = []
	if knowledge.library.lower() in known:
		packages.append(knowledge.library.lower())
	if knowledge.primary_framework.lower() in known:
		packages.append(knowledge.primary_framework.lower())
	for package in {"torch", "tensorflow", "sentence-transformers", "diffusers", "onnxruntime", "vllm"}:
		if re.search(rf"(?:from|import)\s+{re.escape(package.replace('-', '_'))}\b", text):
			packages.append(package)
	existing = {str(package).lower().replace("_", "-") for package in knowledge.dependencies}
	_append(
		knowledge,
		"dependencies",
		[package for package in packages if package.lower().replace("_", "-") not in existing],
		document,
	)


def _extract_metadata(knowledge: ModelKnowledge, metadata: dict[str, Any], document: Document) -> None:
	model_id = metadata.get("modelId") or metadata.get("id")
	_set(knowledge, "model_name", model_id, document)
	_set(knowledge, "model_id", model_id, document)
	_set(knowledge, "task", metadata.get("pipeline_tag"), document)
	_set(knowledge, "pipeline_type", metadata.get("pipeline_tag"), document)
	_set(knowledge, "library", metadata.get("library_name"), document)
	tags = [tag for tag in metadata.get("tags", []) if isinstance(tag, str)]
	frameworks = [tag for tag in tags if tag.lower() in {"pytorch", "tensorflow", "jax", "flax", "keras"}]
	_append(knowledge, "framework", frameworks, document)
	if frameworks:
		_set(knowledge, "primary_framework", frameworks[0], document)
		_append(knowledge, "alternative_frameworks", frameworks[1:], document)
	for tag in tags:
		if tag.lower() in {"safetensors", "onnx", "tflite", "gguf", "ggml"}:
			_append(knowledge, "model_format", [tag], document)
			_append(knowledge, "available_formats", [tag], document)
	config = metadata.get("config", {})
	if isinstance(config, dict):
		_set(knowledge, "architecture", config.get("architectures", []), document)
		_set(knowledge, "model_type", config.get("model_type"), document)
		# ``transformers_version`` records the library version that saved the checkpoint.
		# It is not a requirement, so it is kept only as a configuration fact below and is
		# not written to dependency_versions (where it created false conflicts).
	for key in ("transformers_version", "torch_dtype", "vocab_size", "n_layer", "n_embd"):
		if key in config:
			_append(knowledge, "configuration_requirements", [f"{key}={config[key]}"], document)
	safetensors = metadata.get("safetensors", {})
	if isinstance(safetensors, dict):
		_set(knowledge, "parameter_count", safetensors.get("total") or safetensors.get("parameters"), document)
		_set(knowledge, "model_size", safetensors.get("total_size"), document)


def extract_knowledge(documents: Iterable[Document], repository_url: str = "") -> ModelKnowledge:
	"""Extract deployment facts without treating unrelated repository text as evidence."""
	document_list = list(documents)
	knowledge = ModelKnowledge(repository_url=repository_url, source_documents=[document.source for document in document_list])
	metadata, metadata_document = _metadata(document_list)
	if metadata_document:
		_extract_metadata(knowledge, metadata, metadata_document)
		if not knowledge.repository_url:
			knowledge.repository_url = metadata_document.source.replace("/api/models/", "/")
	json_files = _json_documents(document_list)
	for name, (config, document) in json_files.items():
		base = name.rsplit("/", 1)[-1]
		if base == "config.json":
			_set(knowledge, "architecture", config.get("architectures", []), document)
			_set(knowledge, "model_type", config.get("model_type"), document)
		if "quant" in base:
			_append(knowledge, "quantization", [f"{key}={value}" for key, value in config.items()], document)
		if base in {"generation_config.json", "generation_config.json"}:
			knowledge.generation_parameters.update({key: _clean(value) for key, value in config.items() if value is not None})
			_evidence(knowledge, "generation_parameters", config, document)
		if base in {"tokenizer_config.json", "processor_config.json", "preprocessor_config.json"}:
			_append(knowledge, "tokenizer_processor_requirements", [f"{key}={value}" for key, value in config.items() if key in {"tokenizer_class", "processor_class", "feature_extractor_type", "image_processor_type", "do_normalize", "sampling_rate"}], document)
	for document in document_list:
		name = _document_name(document)
		base_name = name.rsplit("/", 1)[-1]
		if base_name in {"config.json", "generation_config.json", "tokenizer_config.json", "processor_config.json", "preprocessor_config.json", "quantization_config.json", "environment.yml", "pyproject.toml"}:
			_append(knowledge, "required_files", [document.path or document.document_type], document)
		if name.endswith(("requirements.txt", "requirements-dev.txt", "pyproject.toml", "setup.py", "setup.cfg", "environment.yml")):
			_extract_manifest(knowledge, document)
		if name.endswith((".md", ".rst")) or name.endswith(".py") or document.document_type == "framework_documentation":
			# Source files are code, not prose evidence; their comments and docstrings otherwise
			# crowd out the model card in retrieval.
			if not name.endswith(".py"):
				snippet_limit = MODEL_DOCUMENT_SNIPPET_LIMIT if document.origin == "model_repository" else DEFAULT_DOCUMENT_SNIPPET_LIMIT
				_append(knowledge, "document_evidence", _document_snippets(document.content, limit=snippet_limit), document)
			_extract_code(knowledge, document.content, document)
			sections = _sections(document.content, ("install", "setup", "getting started"))
			_append(knowledge, "repository_instructions", sections, document)
			_append(knowledge, "limitations", _sections(document.content, ("limitation", "bias", "out of scope")), document)
			_append(knowledge, "warnings", _sections(document.content, ("warning", "caution", "disclaimer")), document)
			_append(knowledge, "special_requirements", _sections(document.content, ("requirement", "hardware", "cuda", "gpu", "trust_remote_code")), document)
			_append(knowledge, "input_format", _sections(document.content, ("input", "prompt")), document)
			_append(knowledge, "output_format", _sections(document.content, ("output", "prediction")), document)
			_append(knowledge, "inference_methods", _sections(document.content, ("inference", "usage", "how to use")), document)
			install_commands = _commands(document.content, ("pip install", "python -m pip install", "conda install", "mamba install"))
			_append(knowledge, "installation_commands", install_commands, document)
			_extract_install_packages(knowledge, install_commands, document)
			_append(knowledge, "environment_setup_commands", _commands(document.content, ("conda create", "python -m venv", "uv venv")), document)
			if document.origin == "model_repository" and "trust_remote_code" in document.content:
				# A mention is not proof that custom code is required, so the wording stays
				# factual; the deployment planner still blocks on this conservatively.
				_append(knowledge, "special_requirements", ["Repository documentation mentions trust_remote_code; custom repository code may be required (not verified)"], document)
			_add_runtime_dependencies(knowledge, document)
			for line in document.content.splitlines():
				clean_line = _clean(line)
				lower_line = clean_line.lower()
				if re.search(r"\bpython\s+(?:requires|version)\s*(?:[>=<~!]+\s*)?[\d.]+", lower_line):
					_append(knowledge, "python_requirements", [clean_line], document)
				if re.search(r"\bcuda\s+(?:toolkit|runtime|version)\s*(?:[>=<~!]+\s*)?[\d.]+", lower_line):
					_append(knowledge, "cuda_requirements", [clean_line], document)
				# Framework documentation describes the framework, not this model. Capability
				# statements ("supports") are not requirements, and "device" is too ambiguous.
				if (
					document.origin != "official_framework_documentation"
					and re.search(r"\b(?:gpus?|cpu|cuda|vram|ram|memory)\b", lower_line)
					and re.search(r"\b(?:requires?|required|needs?|needed|recommended?|minimum|at least|must have)\b", lower_line)
				):
					_append(knowledge, "gpu_cpu_requirements", [clean_line], document)
		for line in document.content.splitlines():
			if document.document_type == "framework_documentation":
				match = re.search(r"python_requires\s*=\s*[\"']([^\"']+)", line, re.I)
				if match:
					_append(knowledge, "python_requirements", [f"python{match.group(1)}"], document)
				match = re.search(r"supported_python_versions\s*=\s*\((\d+)\s*,\s*(\d+)\)", line, re.I)
				if match:
					_append(knowledge, "python_requirements", [f"SUPPORTED_PYTHON_VERSIONS = ({match.group(1)}, {match.group(2)}) (CPython 3.{match.group(1)} through 3.{match.group(2)})"], document)
			match = re.search(r"(?:python|torch|pytorch|tensorflow|cuda)\s*(?:>=|==|<=|>|<)\s*([0-9][^\s,;]*)", line, re.I)
			if match:
				value = _clean(line)
				field_name = "python_requirements" if "python" in line.lower() else "pytorch_requirements" if "torch" in line.lower() else "tensorflow_requirements" if "tensorflow" in line.lower() else "cuda_requirements"
				_append(knowledge, field_name, [value], document)
	if knowledge.available_formats and not knowledge.required_format:
		# Availability is evidence; the repository has not declared a required format.
		pass
	if knowledge.framework:
		knowledge.primary_framework = knowledge.primary_framework or knowledge.framework[0]
		knowledge.alternative_frameworks = [item for item in knowledge.framework if item != knowledge.primary_framework]
	unique_dependencies = []
	seen_dependencies = set()
	for package in knowledge.dependencies:
		key = package.lower().replace("_", "-")
		if key not in seen_dependencies:
			seen_dependencies.add(key)
			unique_dependencies.append(package)
	knowledge.dependencies = unique_dependencies
	# ``conflicts`` is only visible to retrieval and the UI if it carries evidence records.
	for conflict_key, values in knowledge.conflicts.items():
		package = conflict_key.split(".", 1)[-1]
		declared = re.compile(rf"{re.escape(package)}(?![A-Za-z0-9_.-])", re.I)
		conflict_summary = f"{package} has conflicting declarations ({', '.join(values)})"
		for item in knowledge.evidence.get("dependency_versions", []):
			if declared.match(item["value"]):
				# One record per declaring source so each source stays attributable.
				message = f"{conflict_summary}; this source declares {item['value']}"
				knowledge.evidence.setdefault("conflicts", []).append({**item, "value": message})
	for field in ("python_requirements", "pytorch_requirements", "tensorflow_requirements", "cuda_requirements", "dependencies", "dependency_versions"):
		for item in knowledge.evidence.get(field, []):
			value = item["value"]
			if item.get("origin") == "official_framework_documentation":
				if value not in knowledge.framework_compatibility:
					knowledge.framework_compatibility.append(value)
					knowledge.evidence.setdefault("framework_compatibility", []).append(dict(item))
			elif item.get("origin") == "model_repository":
				if value not in knowledge.model_specific_requirements:
					knowledge.model_specific_requirements.append(value)
					knowledge.evidence.setdefault("model_specific_requirements", []).append(dict(item))
	return knowledge
