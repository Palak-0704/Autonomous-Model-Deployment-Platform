"""Offline tests for the evidence-only Qwen decision guard."""

from .decision import evidence_only_plan, unsupported_plan_values


def main() -> None:
    missing_runtime_context = """MODEL
Model name: demo/model
Task: text-generation

DEPENDENCIES
Dependencies: transformers; torch
"""
    invented_runtime_plan = {
        "model": "demo/model",
        "environment": "python_version: 3.8; cuda_version: 11.7",
        "dependencies": ["transformers==4.25.1", "torch==1.10.2+cu117"],
        "installation": "pip install transformers torch",
        "loading": "Not specified in repository",
        "inference": "Not specified in repository",
        "special_requirements": "GPU required",
        "missing_information": "Not specified in repository",
    }
    rejected = unsupported_plan_values(invented_runtime_plan, missing_runtime_context)
    assert {"environment", "dependencies", "installation", "special_requirements"} <= set(rejected)
    safe_plan = evidence_only_plan(invented_runtime_plan, missing_runtime_context)
    for field in rejected:
        assert safe_plan[field] == "Not specified in repository"

    version_range_context = "DEPENDENCIES\nDependency versions: torch>=2.0\n"
    preserved = {"dependencies": "torch>=2.0"}
    changed = {"dependencies": "torch==2.0.0"}
    assert not unsupported_plan_values(preserved, version_range_context)
    assert "dependencies" in unsupported_plan_values(changed, version_range_context)

    no_hardware_context = "MODEL\nModel name: demo/model\n"
    hardware_plan = {"environment": "CUDA 12.1", "special_requirements": "NVIDIA GPU"}
    safe_hardware_plan = evidence_only_plan(hardware_plan, no_hardware_context)
    assert set(safe_hardware_plan.values()) == {"Not specified in repository"}
    print("Evidence-only missing-information tests passed.")


if __name__ == "__main__":
    main()
