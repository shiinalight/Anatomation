"""Hosted Gemma feedback for rendered practice-pad exercises."""

import base64
import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import mujoco
from PIL import Image


MODEL = "gemma-4-26b-a4b-it"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
PROMPT = (
    "You are a visual coach in a fictional MuJoCo suturing exercise on a silicone pad. "
    "Start with the supplied Current action, preserving every key and repetition count. "
    "Never replace it with a later step. "
    "Use the image for the visible relationship between the hand, tool, and practice pad. "
    "Use the simulator state for progress, tool identity, and controls. Say if the view is obscured. "
    "Output at most two short plain-text sentences with no headings. Stay within this simulator "
    "and its controls. Do not discuss patient care, anatomy, real procedures, or diagnosis."
)


def analyze_frame(api_key, frame, cue, observations):
    jpeg = BytesIO()
    Image.fromarray(frame).save(jpeg, format="JPEG", quality=87)
    body = {"contents": [{"parts": [
        {"inlineData": {
            "mimeType": "image/jpeg",
            "data": base64.b64encode(jpeg.getvalue()).decode("ascii"),
        }},
        {"text": f"{PROMPT}\nCurrent action: {cue}\nSimulator state: {observations}"},
    ]}], "generationConfig": {
        "maxOutputTokens": 512, "temperature": 0.2,
        "thinkingConfig": {"thinkingLevel": "minimal"},
    }}
    request = Request(URL, data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    })
    with urlopen(request, timeout=35) as response:
        result = json.load(response)
    candidate = next(iter(result.get("candidates", [])), {})
    parts = candidate.get("content", {}).get("parts", [])
    text = " ".join(part.get("text", "") for part in parts if not part.get("thought"))
    text = " ".join(text.split())
    if not text:
        raise ValueError("Empty Gemma response")
    return text


class GemmaCoach:
    def __init__(self, model, api_key, camera="bench"):
        self.renderer = mujoco.Renderer(model, height=540, width=768)
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.api_key = api_key
        self.camera = camera
        self.pending = None
        self.requested_state = None
        self.previous_state = None

    def close(self):
        self.renderer.close()
        self.worker.shutdown(wait=False, cancel_futures=True)

    def update(self, data, requested, cue, draw, observations, identity):
        if self.previous_state != identity:
            self.requested_state = None
        self.previous_state = identity
        message = None
        if self.pending and self.pending.done():
            completed, self.pending = self.pending, None
            try:
                message = completed.result()
                if self.requested_state != identity:
                    message = "The exercise changed during the review. Press B to review the current step."
            except HTTPError as error:
                message = {
                    400: "Google rejected the image request (400).",
                    401: "Google rejected this API key (401).",
                    403: "This project cannot access Gemma (403).",
                    404: "Gemma model unavailable for this project (404).",
                    429: "Gemma quota reached. Try again later (429).",
                }.get(error.code, f"Gemma request failed ({error.code}).")
            except TimeoutError:
                message = "Gemma timed out. Press B to retry."
            except URLError:
                message = "Could not reach Google. Press B to retry."
            except ValueError:
                message = "Gemma returned no text. Press B to retry."
        if requested and not self.pending:
            self.renderer.update_scene(data, camera=self.camera)
            draw(self.renderer.scene)
            frame = self.renderer.render()
            self.requested_state = identity
            self.pending = self.worker.submit(analyze_frame, self.api_key, frame, cue, observations)
        return message
