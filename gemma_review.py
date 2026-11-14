"""Gemma debrief: a written review of the scene and of the attempt's scores.

Gemma (Google's open model) looks at a rendered frame of the operating field together with
the attempt's task scores and the best results recorded on Aiven, and writes a short,
specific debrief. Two backends:

- Modal (preferred, open weights on our own GPU): set GEMMA_MODAL_URL to the URL printed
  by `modal deploy modal_gemma.py`, and GEMMA_MODAL_TOKEN to the token stored in the Modal
  secret (see modal_gemma.py). The server speaks the OpenAI chat-completions API (vLLM).
- Google AI Studio (hosted Gemma): SURGERY_GEMMA_API_KEY or GEMINI_API_KEY.
"""

import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from aiven_store import _env

HOSTED_MODEL = os.environ.get("GEMMA_MODEL", "gemma-4-26b-a4b-it")
HOSTED_URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"
PROMPT = (
    "You are the debriefing coach of a two-handed MuJoCo surgical-skills simulator (skin laceration, "
    "heart laceration, or a guided coronary bypass on a heart model). The right hand cuts and drives the "
    "needle; the left hand steadies tissue, lifts each edge with forceps and pulls the suture through. "
    "Use the image for what is visible (instrument positions, wound, stitches) and the data for timing, "
    "travel, refused actions and needle accuracy, compared with the best recorded results. "
    "Give at most four short plain-text sentences: what went well, the one task that cost the most, and "
    "one concrete thing to change next time using the simulator's controls. "
    "Stay within the simulator; do not give real clinical advice or discuss patients."
)


def load_api_key():
    """Google AI Studio key: SURGERY_GEMMA_API_KEY or GEMINI_API_KEY (environment or .env)."""
    return _env("SURGERY_GEMMA_API_KEY") or _env("GEMINI_API_KEY")


def backend():
    if _env("GEMMA_MODAL_URL"):
        return "Gemma on Modal"
    if load_api_key():
        return "Gemma (Google AI Studio)"
    return None


def _jpeg(frame):
    from PIL import Image
    out = BytesIO()
    Image.fromarray(frame).save(out, format="JPEG", quality=85)
    return base64.b64encode(out.getvalue()).decode("ascii")


def _post(url, body, headers, timeout):
    request = Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def review(frame, facts):
    """Debrief text for a rendered frame (H x W x 3 uint8, or None) and `facts` (text)."""
    image = _jpeg(frame) if frame is not None else None
    modal_url = _env("GEMMA_MODAL_URL")
    if modal_url:
        content = [{"type": "text", "text": f"{PROMPT}\n\n{facts}"}]
        if image:
            content.insert(0, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}})
        result = _post(modal_url.rstrip("/") + "/v1/chat/completions",
                       {"model": _env("GEMMA_MODAL_MODEL") or "gemma",
                        "messages": [{"role": "user", "content": content}],
                        "max_tokens": 300, "temperature": .2},
                       {"Authorization": f"Bearer {_env('GEMMA_MODAL_TOKEN') or ''}"}, timeout=120)
        text = result["choices"][0]["message"]["content"]
    else:
        parts = [{"text": f"{PROMPT}\n\n{facts}"}]
        if image:
            parts.insert(0, {"inlineData": {"mimeType": "image/jpeg", "data": image}})
        result = _post(HOSTED_URL.format(HOSTED_MODEL),
                       {"contents": [{"parts": parts}],
                        "generationConfig": {"maxOutputTokens": 400, "temperature": .2}},
                       {"x-goog-api-key": load_api_key() or ""}, timeout=45)
        candidate = next(iter(result.get("candidates", [])), {})
        text = " ".join(p.get("text", "") for p in candidate.get("content", {}).get("parts", [])
                        if not p.get("thought"))
    text = " ".join(text.split())
    if not text:
        raise ValueError("empty Gemma reply")
    return text


class GemmaReviewer:
    """Runs reviews in the background; poll() returns the finished text once."""

    def __init__(self):
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.pending = None

    @property
    def busy(self):
        return self.pending is not None

    def request(self, frame, facts):
        if self.pending is None:
            self.pending = self.worker.submit(review, frame, facts)

    def poll(self):
        if self.pending is None or not self.pending.done():
            return None
        future, self.pending = self.pending, None
        try:
            return future.result()
        except HTTPError as error:
            return {401: "Gemma: the API key was rejected (401).", 403: "Gemma: no access to the model (403).",
                    404: "Gemma: model not found (404).", 429: "Gemma: quota reached, try later (429)."
                    }.get(error.code, f"Gemma request failed ({error.code}).")
        except (URLError, TimeoutError, OSError):
            return "Gemma: could not reach the server."
        except (ValueError, KeyError, IndexError):
            return "Gemma returned no text."
        except Exception as error:      # e.g. Pillow missing: never stop the simulator
            return f"Gemma review failed ({type(error).__name__})."

    def close(self):
        self.worker.shutdown(wait=False, cancel_futures=True)
