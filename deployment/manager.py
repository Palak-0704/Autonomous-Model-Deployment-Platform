"""Orchestrates inspection, validated planning, optional runtime setup, and smoke tests."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
from uuid import uuid4
from typing import Any

from Rag.extraction.extractor import extract_knowledge
from Rag.ingestion.loader import load_repository

from .plan import DeploymentPlan, build_plan, validate_plan
from .runtime import RuntimeManager


class SessionStatus(str, Enum):
    INSPECTED = "INSPECTED"; BLOCKED = "BLOCKED"; READY = "READY"; SUCCESS = "SUCCESS"; FAILED = "FAILED"


@dataclass
class DeploymentResult:
    session_id: str
    status: SessionStatus
    message: str
    plan: dict[str, Any] = field(default_factory=dict)
    report_path: str = ""
    logs: list[str] = field(default_factory=list)


class DeploymentManager:
    def __init__(self, report_root: Path = Path("deployment_reports")) -> None:
        self.report_root = report_root
        self.runtime = RuntimeManager()
        self.sessions: dict[str, DeploymentResult] = {}

    def inspect(self, repository_url: str, token: str = "") -> DeploymentResult:
        session_id = uuid4().hex[:12]
        documents = load_repository(repository_url, token=token)
        knowledge = extract_knowledge(documents, repository_url=repository_url)
        plan = build_plan(knowledge)
        blockers = validate_plan(plan)
        result = DeploymentResult(session_id, SessionStatus.BLOCKED if blockers else SessionStatus.READY,
            "; ".join(blockers) if blockers else "Plan validated. Local model downloads are not started during inspection.", plan.to_dict(), logs=[f"Collected {len(documents)} lightweight repository documents."])
        result.report_path = self._write_report(result)
        self.sessions[session_id] = result
        return result

    def prepare_runtime(self, result: DeploymentResult, install_dependencies: bool = False) -> DeploymentResult:
        if result.status != SessionStatus.READY:
            return result
        runtime = self.runtime.create(result.session_id)
        result.logs.append(runtime.message)
        if install_dependencies:
            try:
                result.logs.append(self.runtime.install(runtime.path, result.plan.get("dependencies", [])))
            except Exception as error:
                result.status, result.message = SessionStatus.FAILED, str(error)
        else:
            result.message = "Validated plan; runtime created without downloads. Enable installation only when ready to run the target model."
        result.report_path = self._write_report(result)
        self.sessions[result.session_id] = result
        return result

    def smoke_test(self, result: DeploymentResult) -> DeploymentResult:
        if result.status != SessionStatus.READY:
            return result
        try:
            runtime = self.runtime.root / result.session_id / "runtime"
            if not self.runtime.python_path(runtime).exists():
                raise RuntimeError("Create and install the isolated runtime before running a smoke test.")
            smoke_input = {
                "text-generation": "Hello, my name is",
                "text2text-generation": "summarize: A short test sentence.",
                "fill-mask": "The capital of France is <mask>.",
                "text-classification": "This is a helpful product.",
                "token-classification": "Alice lives in London.",
                "question-answering": {"question": "What is the answer?", "context": "The answer is a smoke test."},
                "feature-extraction": "A small inference smoke test.",
            }.get(result.plan["task"])
            if smoke_input is None:
                raise ValueError(f"{result.plan['task']} needs a user-provided task-specific sample; automatic smoke testing is unsafe.")
            output = self.runtime.run_transformers_pipeline(
                runtime, result.plan["model_id"], result.plan["task"], smoke_input
            )
            result.status, result.message = SessionStatus.SUCCESS, "Target model loaded and completed a smoke test."
            result.logs.append("Smoke output: " + repr(output)[:1200])
        except Exception as error:
            result.status, result.message = SessionStatus.FAILED, f"Smoke test failed: {error}"
        result.report_path = self._write_report(result)
        self.sessions[result.session_id] = result
        return result

    def predict(self, session_id: str, value: Any) -> Any:
        """Run a user request only after the session's smoke test succeeded."""
        result = self.sessions.get(session_id)
        if result is None or result.status is not SessionStatus.SUCCESS:
            raise RuntimeError("Run a successful smoke test before submitting model input.")
        runtime = self.runtime.root / session_id / "runtime"
        return self.runtime.run_transformers_pipeline(
            runtime, result.plan["model_id"], result.plan["task"], value
        )

    def _write_report(self, result: DeploymentResult) -> str:
        self.report_root.mkdir(parents=True, exist_ok=True)
        path = self.report_root / f"{result.session_id}.json"
        path.write_text(json.dumps({"timestamp": datetime.now().isoformat(), **asdict(result), "status": result.status.value}, indent=2, default=str), encoding="utf-8")
        return str(path)
