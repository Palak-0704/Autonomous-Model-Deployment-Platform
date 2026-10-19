"""Offline regression checks for deployment planning and safe runtime behavior."""

from unittest.mock import patch

from Rag.extraction.extractor import ModelKnowledge
from .manager import DeploymentManager, SessionStatus
from .plan import build_plan, validate_plan


def main() -> None:
    knowledge = ModelKnowledge(model_id="org/demo", repository_url="https://huggingface.co/org/demo", task="text-generation", pipeline_type="text-generation", primary_framework="transformers", dependencies=["transformers"])
    plan = build_plan(knowledge)
    assert not validate_plan(plan)
    assert {"transformers", "torch"} <= set(plan.dependencies)
    documented_only = ModelKnowledge(
        model_id="org/demo", task="text-generation", pipeline_type="text-generation",
        primary_framework="pytorch", library="transformers", dependencies=["jax", "transformers"],
        evidence={"dependencies": [
            {"value": "jax", "origin": "official_framework_documentation"},
            {"value": "transformers", "origin": "model_repository"},
        ]},
    )
    assert build_plan(documented_only).dependencies == ["transformers", "torch"]
    custom = ModelKnowledge(model_id="org/demo", task="text-generation", pipeline_type="text-generation", primary_framework="transformers", special_requirements=["trust_remote_code=True is required by repository documentation"])
    assert validate_plan(build_plan(custom)) == ["Custom repository code is not executed automatically."]
    manager = DeploymentManager()
    with patch("deployment.manager.load_repository", return_value=[]), patch("deployment.manager.extract_knowledge", return_value=knowledge):
        result = manager.inspect(knowledge.repository_url)
    assert result.status is SessionStatus.READY
    assert "downloads are not started" in result.message
    print("Deployment planning and no-download inspection tests passed.")


if __name__ == "__main__":
    main()
