"""Serve Gemma (open weights) on a Modal GPU for the simulator's debriefs.

    pip install modal && modal setup                    # once
    modal secret create surgisense-gemma \\
        HF_TOKEN=<Hugging Face token with Gemma access> GEMMA_MODAL_TOKEN=<any long random string>
    modal deploy modal_gemma.py                         # prints the URL

Then in the simulator's .env:

    GEMMA_MODAL_URL=<the printed URL>
    GEMMA_MODAL_TOKEN=<the same random string>

The server is vLLM's OpenAI-compatible API (POST /v1/chat/completions, images allowed),
scales to zero when idle and starts on the first debrief (a minute or two the first time).
"""

import os
import subprocess

import modal

MODEL = os.environ.get("GEMMA_HF_MODEL", "google/gemma-3-12b-it")
GPU = os.environ.get("GEMMA_GPU", "L40S")
PORT = 8000

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.10.1.1", "huggingface_hub[hf_transfer]==0.34.4")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
weights = modal.Volume.from_name("surgisense-gemma-weights", create_if_missing=True)
app = modal.App("surgisense-gemma")


@app.function(image=image, gpu=GPU, timeout=60*60, scaledown_window=10*60,
              secrets=[modal.Secret.from_name("surgisense-gemma")],
              volumes={"/root/.cache/huggingface": weights})
@modal.concurrent(max_inputs=8)
@modal.web_server(port=PORT, startup_timeout=20*60)
def serve():
    subprocess.Popen([
        "vllm", "serve", MODEL, "--served-model-name", "gemma", "--host", "0.0.0.0",
        "--port", str(PORT), "--max-model-len", "8192", "--limit-mm-per-prompt", '{"image": 1}',
        "--api-key", os.environ["GEMMA_MODAL_TOKEN"],
    ])
