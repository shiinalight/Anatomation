"""Gemini voice coach: spoken (or typed) questions about the current step or a past attempt.

The microphone records one question at a time (it stops after a pause), Gemini answers from
the simulator's state, and the answer is read aloud (macOS `say`, Windows speech, or espeak).
Ported from the ai-assistant-aiven branch (hadi-ce04) and adapted to the two-handed station.
"""

import base64
import json
import os
import subprocess
import sys
import wave
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import sounddevice as sd


MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
MAX_SECONDS = 18
BLOCK_SECONDS = .03
SILENCE_SECONDS = .66
SPEAK = (
    "Add-Type -AssemblyName System.Speech; "
    "[Console]::InputEncoding = [System.Text.Encoding]::UTF8; "
    "$speech = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "try { $speech.Speak([Console]::In.ReadToEnd()) } finally { $speech.Dispose() }"
)


def answer_question(api_key, pcm, rate, context, question=None, *, review=False):
    if question is None:
        audio = BytesIO()
        with wave.open(audio, "wb") as wav:
            wav.setparams((1, 2, rate, 0, "NONE", "not compressed"))
            wav.writeframes(pcm)
        question_part = {"inlineData": {"mimeType": "audio/wav",
                                        "data": base64.b64encode(audio.getvalue()).decode("ascii")}}
    else:
        question_part = {"text": question}

    if review:
        instructions = (
            "You review recorded virtual practice sessions in a two-handed MuJoCo suturing simulator "
            "(skin laceration, heart laceration, coronary bypass). "
            "Answer using only the supplied session records. Cite elapsed event timestamps when useful. "
            "Distinguish elapsed session time from the time between actions. "
            "If a detail was not recorded, say so. Do not infer clinical skill or give real surgical advice. "
            "Keep answers under 150 words."
        )
    else:
        instructions = (
            "You are a voice coach in a two-handed MuJoCo surgical-skills simulator (skin laceration, "
            "heart laceration, or a guided coronary bypass on a heart model). The right hand cuts and drives "
            "the needle; the left hand steadies tissue, lifts each edge with forceps, and pulls the suture "
            "through; knots are three-throw instrument ties. The keyboard follows the hand the next step needs. "
            "Hands-only procedures use no keys at all, only wrist movements: when the supplied state says "
            "hands-only, describe wrist movements and never mention keys. "
            "Answer the trainee's question using the supplied simulator state and controls. "
            "If asked what to do now, start with the supplied Current action, preserving every key "
            "and repetition count. Never replace it with a later step. Only describe actions implemented "
            "by the simulator controls. Explain why the action applies when asked for details. "
            "For a walkthrough, cover each remaining stage and its keys in order. "
            "Use natural spoken language and stay under 100 words. "
            "Stay within the simulator. Briefly redirect real surgery or patient-care questions "
            "to simulator controls. Give simulator answers directly without a disclaimer. "
            "If there is no intelligible question, ask the trainee to repeat it."
        )
    body = {
        "systemInstruction": {"parts": [{"text": instructions}]},
        "contents": [{"parts": [question_part, {"text": context}]}],
        "generationConfig": {"maxOutputTokens": 360, "thinkingConfig": {"thinkingLevel": "minimal"}},
    }
    request = Request(URL, data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    })
    with urlopen(request, timeout=25) as response:
        result = json.load(response)
    candidate = next(iter(result.get("candidates", [])), {})
    parts = candidate.get("content", {}).get("parts", [])
    text = " ".join(part.get("text", "") for part in parts if not part.get("thought"))
    text = " ".join(text.split())
    if not text:
        raise ValueError("Empty Gemini response")
    return text


class SpeechTurn:
    """Keep a short lead-in and end the question after a pause in speech."""

    def __init__(self, rate):
        self.rate = rate
        self.lead = deque(maxlen=9)
        self.blocks = []
        self.samples = 0
        self.silence = 0
        self.voiced = 0

    def add(self, block):
        samples = np.frombuffer(block, dtype="<i2").astype(np.float32)
        voiced = np.sqrt(np.mean(samples * samples)) >= 125
        if not self.blocks:
            self.lead.append(block)
            self.voiced = self.voiced + len(samples) if voiced else 0
            if self.voiced >= self.rate * .06:
                self.blocks.extend(self.lead)
                self.samples = sum(map(len, self.blocks)) // 2
            return None
        self.blocks.append(block)
        self.samples += len(samples)
        if voiced:
            self.voiced += len(samples)
            self.silence = 0
        else:
            self.silence += len(samples)
        if self.silence < self.rate * SILENCE_SECONDS and self.samples < self.rate * MAX_SECONDS:
            return None
        pcm = b"".join(self.blocks) if self.voiced >= self.rate * .20 else None
        self.__init__(self.rate)
        return pcm


class GeminiVoice:
    def __init__(self, api_key):
        self.api_key = api_key
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.stream = None
        self.pending = None
        self.speech = None
        self.context = None
        self.previous_identity = None
        self.read_aloud = True
        self.active = False
        self.last_answer = ""
        self.message = "Start voice with V, then ask a question."

    def close(self):
        if self.stream:
            self.stream.close()
        if self.speech and self.speech.poll() is None:
            self.speech.terminate()
        self.worker.shutdown(wait=False, cancel_futures=True)

    def _record(self):
        self.samples = deque(maxlen=90)
        self.rate = int(sd.query_devices(kind="input")["default_samplerate"])
        self.turn = SpeechTurn(self.rate)
        self.stream = sd.RawInputStream(samplerate=self.rate, channels=1, dtype="int16",
                                        blocksize=round(self.rate * BLOCK_SECONDS), callback=self._capture)
        try:
            self.stream.start()
        except BaseException:
            self.stream.close()
            self.stream = None
            raise

    def _capture(self, data, frames, time_info, status):
        self.samples.append(bytes(data))

    def _stop_recording(self):
        if not self.stream:
            return
        stream, self.stream = self.stream, None
        stream.close()
        self.samples.clear()

    def _send(self, pcm, context, identity, review=False):
        self._stop_recording()
        self.context = identity
        self.read_aloud = True
        self.pending = self.worker.submit(answer_question, self.api_key, pcm, self.rate,
                                          self._prompt(context), review=review)
        self.message = "Gemini is answering..."

    def _prompt(self, context):
        if self.last_answer:
            return context + "\nPrevious coach reply, for follow-up questions: " + self.last_answer
        return context

    def ask(self, question, context, identity, *, review=False):
        if self.pending or self.speech:
            return False
        self._stop_recording()
        self.context = identity
        self.previous_identity = identity    # (else update() takes it for a new exercise and drops the answer)
        self.read_aloud = False
        self.pending = self.worker.submit(answer_question, self.api_key, None, None,
                                          self._prompt(context), question, review=review)
        self.message = "Gemini is answering..."
        return True

    def _finish(self, context):
        future, self.pending = self.pending, None
        try:
            answer = future.result()
        except HTTPError as error:
            self.active = False
            self.message = f"Gemini request failed ({error.code}). Press V to retry."
            try:
                with error:
                    details = json.load(error).get("error", {}).get("details", [])
                if any(detail.get("reason") == "API_KEY_INVALID" for detail in details):
                    self.message = "Google rejected the API key. Update .env and reopen the viewer."
            except (OSError, ValueError):
                pass
            return
        except (TimeoutError, URLError, ValueError):
            self.active = False
            self.message = "Gemini response unavailable. Press V to retry."
            return
        if context != self.context:
            self.message = "The exercise changed while I answered. Ask for the current step."
            return
        self.last_answer = answer
        self.message = answer
        if not self.read_aloud or not self.active:
            return
        try:
            if sys.platform == "win32":
                command = ["powershell.exe", "-NoProfile", "-Command", SPEAK]
            elif sys.platform == "darwin":
                command = ["say", "-f", "-"]
            else:
                command = ["espeak", "--stdin"]
            self.speech = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            with self.speech.stdin as pipe:
                pipe.write(answer.encode("utf-8"))
        except OSError:
            self.message += "\nCould not start speech playback."

    def update(self, context, toggle, identity, *, review=False):
        if identity != self.previous_identity:
            self.context = None
            self.last_answer = ""
        self.previous_identity = identity
        if toggle:
            self.active = not self.active
            if not self.active:
                self._stop_recording()
                self.read_aloud = False
                if self.speech and self.speech.poll() is None:
                    self.speech.terminate()
                    self.speech = None
        if self.pending and self.pending.done():
            self._finish(identity)
        if self.speech and self.speech.poll() is not None:
            if self.speech.returncode:
                self.message += "\nSpeech playback failed."
            self.speech = None
        if self.active and not (self.stream or self.pending or self.speech):
            try:
                self._record()
            except (sd.PortAudioError, ValueError):
                self.active = False
                self.message = "Microphone unavailable. Check your default input."
        if self.stream:
            while self.samples:
                pcm = self.turn.add(self.samples.popleft())
                if pcm:
                    self._send(pcm, context, identity, review)
                    break
        return self.message
