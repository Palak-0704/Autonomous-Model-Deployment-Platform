"""Validated, evidence-backed deployment planning; no commands are executed here."""

from dataclasses import asdict, dataclass, field
from typing import Any

from Rag.extraction.extractor import ModelKnowledge


@dataclass
class DeploymentPlan:
    repository_url: str
    model_id: str
    task: str
    framework: str
    dependencies: list[str]
    loading_method: str
    input_hint: str
    uncertainties: list[str] = field(default_factory=list)
    evidence: dict[str, list[dict[str, str]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SUPPORTED_TASKS = {
    "text-generation", "text2text-generation", "fill-mask", "text-classification",
    "token-classification", "question-answering", "feature-extraction", "image-classification",
    "automatic-speech-recognition",
}


def build_plan(knowledge: ModelKnowledge) -> DeploymentPlan:
    task = knowledge.pipeline_type or knowledge.task
    framework = knowledge.primary_framework or knowledge.library
    # Official compatibility pages are retrieval evidence, not packages the
    # target model requires.  Only model/linked-repository declarations are
    # eligible for installation.
    dependency_evidence = knowledge.evidence.get("dependencies", [])
    model_declared = {
        str(item.get("value", "")).casefold()
        for item in dependency_evidence
        if item.get("origin") != "official_framework_documentation"
    }
    dependencies = [item for item in knowledge.dependencies if item.casefold() in model_declared]
    uses_transformers = knowledge.library.casefold() == "transformers" or framework.casefold() in {"transformers", "pytorch", "torch"}
    if uses_transformers and "transformers" not in dependencies:
        dependencies.append("transformers")
    # The built-in safe adapter uses the PyTorch-backed Transformers pipeline.
    # Declare that runtime requirement explicitly rather than relying on a
    # package installed in the parent process.
    if uses_transformers and "torch" not in dependencies:
        dependencies.append("torch")
    uncertainties = []
    if not task:
        uncertainties.append("Repository does not declare a pipeline task.")
    elif task not in SUPPORTED_TASKS:
        uncertainties.append(f"Task '{task}' has no built-in safe adapter.")
    if not framework:
        uncertainties.append("Repository does not declare a primary framework.")
    if any("trust_remote_code" in item.lower() for item in knowledge.special_requirements):
        uncertainties.append("Repository documents custom code; it is blocked until explicitly approved.")
    return DeploymentPlan(
        repository_url=knowledge.repository_url,
        model_id=knowledge.model_id,
        task=task,
        framework=framework,
        dependencies=dependencies,
        loading_method="transformers.pipeline" if task in SUPPORTED_TASKS else "not selected",
        input_hint="Repository documented input: " + "; ".join(knowledge.input_format[:2]) if knowledge.input_format else "Not specified in repository",
        uncertainties=uncertainties,
        evidence=knowledge.evidence,
    )


def validate_plan(plan: DeploymentPlan) -> list[str]:
    """Return blockers; accepting a plan never authorizes repository code execution."""
    blockers = []
    if not plan.model_id:
        blockers.append("Model ID is unavailable.")
    if not plan.task:
        blockers.append("Pipeline task is unavailable; no input adapter can be selected safely.")
    if plan.task and plan.task not in SUPPORTED_TASKS:
        blockers.append(f"Unsupported built-in task: {plan.task}.")
    if plan.framework and plan.framework.lower() not in {"transformers", "pytorch", "torch"}:
        blockers.append(f"Unsupported framework for automatic loading: {plan.framework}.")
    if any("custom code" in item.lower() for item in plan.uncertainties):
        blockers.append("Custom repository code is not executed automatically.")
    return blockers
