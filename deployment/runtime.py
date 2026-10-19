"""Controlled per-session virtual-environment lifecycle."""

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import venv
import json
from typing import Any


@dataclass
class RuntimeResult:
    path: Path
    created: bool
    message: str


class RuntimeManager:
    def __init__(self, root: Path = Path("deployment_sessions")) -> None:
        self.root = root

    def create(self, session_id: str) -> RuntimeResult:
        """Create an empty isolated environment. Package installation is deliberately separate."""
        path = self.root / session_id / "runtime"
        if path.exists():
            return RuntimeResult(path, False, "Existing isolated runtime reused; no packages changed.")
        path.parent.mkdir(parents=True, exist_ok=True)
        venv.EnvBuilder(with_pip=True, clear=False).create(path)
        return RuntimeResult(path, True, "Created an empty isolated runtime; dependency installation requires explicit approval.")

    @staticmethod
    def python_path(runtime: Path) -> Path:
        return runtime / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

    def install(self, runtime: Path, packages: list[str], timeout: int = 300) -> str:
        """Install only normalized package specifications selected from the validated plan."""
        if not packages:
            return "No repository-supported packages to install."
        safe = [item for item in packages if item and not any(x in item for x in ("/", "\\", ";", "&&", "|", "git+"))]
        if len(safe) != len(packages):
            raise ValueError("Unsafe dependency specification was rejected.")
        command = [str(self.python_path(runtime)), "-m", "pip", "install", "--disable-pip-version-check", *safe]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        output = (completed.stdout + "\n" + completed.stderr)[-8000:]
        if completed.returncode:
            raise RuntimeError(f"Dependency installation failed (exit {completed.returncode}):\n{output}")
        return output

    def run_transformers_pipeline(
        self, runtime: Path, model_id: str, task: str, value: Any, timeout: int = 600
    ) -> Any:
        """Load and invoke an allowlisted Transformers pipeline inside a session venv."""
        script = """
import json
import sys
from transformers import pipeline

request = json.load(sys.stdin)
runner = pipeline(request['task'], model=request['model_id'], trust_remote_code=False)
print(json.dumps(runner(request['value']), default=str))
"""
        completed = subprocess.run(
            [str(self.python_path(runtime)), "-c", script],
            input=json.dumps({"model_id": model_id, "task": task, "value": value}),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if completed.returncode:
            output = (completed.stdout + "\n" + completed.stderr)[-8000:]
            raise RuntimeError(f"Isolated model execution failed (exit {completed.returncode}):\n{output}")
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError("Isolated model execution did not return JSON output.") from error
