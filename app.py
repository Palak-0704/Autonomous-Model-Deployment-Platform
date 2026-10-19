"""Local control panel. Inspection is lightweight; running a model is an explicit second action."""

import json
import gradio as gr
from deployment import DeploymentManager

manager = DeploymentManager()
sessions = {}

def inspect(url: str, token: str) -> tuple[str, str]:
    try:
        result = manager.inspect(url.strip(), token.strip()); sessions[result.session_id] = result
        return result.session_id, json.dumps({"status": result.status.value, "message": result.message, "plan": result.plan, "report": result.report_path}, indent=2)
    except Exception as error:
        return "", json.dumps({"status": "FAILED", "message": str(error)}, indent=2)

def prepare(session_id: str, install: bool) -> str:
    result = sessions.get(session_id)
    if not result: return "Unknown session. Inspect a repository first."
    result = manager.prepare_runtime(result, install); sessions[session_id] = result
    return json.dumps({"status": result.status.value, "message": result.message, "report": result.report_path}, indent=2)

def smoke(session_id: str) -> str:
    result = sessions.get(session_id)
    if not result: return "Unknown session. Inspect a repository first."
    result = manager.smoke_test(result); sessions[session_id] = result
    return json.dumps({"status": result.status.value, "message": result.message, "logs": result.logs[-2:], "report": result.report_path}, indent=2)

def infer(session_id: str, user_input: str) -> str:
    try:
        return json.dumps(manager.predict(session_id, user_input), indent=2, default=str)
    except Exception as error:
        return json.dumps({"error": str(error)})

with gr.Blocks(title="Autonomous ML Deployment") as app:
    gr.Markdown("# Autonomous ML Deployment and Testing\nInspection never downloads model weights. Qwen reasoning runs in your synchronized Colab GPU runtime.")
    url = gr.Textbox(label="Public Hugging Face model URL", value="https://huggingface.co/distilbert/distilroberta-base")
    token = gr.Textbox(label="Optional Hugging Face token", type="password")
    inspect_button = gr.Button("Inspect and validate plan")
    session = gr.Textbox(label="Session ID")
    result = gr.Code(label="Validated deployment plan / status", language="json")
    inspect_button.click(inspect, [url, token], [session, result])
    install = gr.Checkbox(label="I approve installation into this session's isolated runtime", value=False)
    prepare_button = gr.Button("Create isolated runtime")
    prepare_button.click(prepare, [session, install], result)
    smoke_button = gr.Button("Download, load, and smoke-test target model")
    smoke_button.click(smoke, session, result)
    gr.Markdown("### Test a loaded text model\nFor image or audio tasks, the safe loader requires a task-specific sample and is intentionally blocked from automatic smoke testing.")
    user_input = gr.Textbox(label="Text input (text tasks only)")
    inference_output = gr.Code(label="Model output", language="json")
    gr.Button("Run inference").click(infer, [session, user_input], inference_output)

if __name__ == "__main__": app.launch()
