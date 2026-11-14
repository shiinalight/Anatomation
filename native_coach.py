"""Connect the native MuJoCo Equality controls to the coaching services."""

import json
import os
from pathlib import Path

from gemini_voice import GeminiVoice
from gemma_coach import GemmaCoach


VOICE = "AI Voice"
REVIEW = "Review scene"
SAVE = "Save attempt"
HISTORY = "AI history"
NEXT = "Next attempt"
ASSISTANT = "AI Assistant"


def load_api_key():
    configured = os.environ.get("SURGERY_GEMMA_API_KEY", "").strip()
    if configured:
        return configured
    project = Path(__file__).resolve().parent
    for path in (project / ".env", project.parent / ".env"):
        if not path.is_file():
            continue
        names = {"SURGERY_GEMMA_API_KEY", "GEMINI_API_KEY"} if path.parent == project else {"SURGERY_GEMMA_API_KEY"}
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            name, _, value = line.partition("=")
            if name.strip() in names and value.strip():
                value = value.strip().strip('"\'')
                if value and value != "your_key_here":
                    return value
    return os.environ.get("GEMINI_API_KEY", "").strip() or None


def controls(history):
    return (VOICE, REVIEW, ASSISTANT, SAVE, HISTORY, NEXT) if history else (VOICE, REVIEW, ASSISTANT)


class NativeCoach:
    def __init__(self, model, key, history, mode, resources):
        self.gemini = GeminiVoice(key)
        resources.callback(self.gemini.close)
        self.gemma = GemmaCoach(model, key)
        resources.callback(self.gemma.close)
        self.ids = {name: model.equality(name).id for name in controls(history)}
        self.history, self.mode = history, mode
        self.history_open = False
        self.attempts = []
        self.index = 0
        self.pending = None
        self.history_status = ""
        self.scene_reply = ""
        self.voice_reply = ""
        self.previous_context = None
        self.popup = None
        self.resources = resources

    def clicks(self, data):
        clicked = {name for name, index in self.ids.items() if data.eq_active[index]}
        for name in clicked - {VOICE, HISTORY}:
            data.eq_active[self.ids[name]] = False
        return clicked

    def set_voice_indicator(self, data, clicked):
        index = self.ids[VOICE]
        if bool(data.eq_active[index]) == (VOICE in clicked):
            data.eq_active[index] = self.gemini.active

    def saved(self, pending):
        if pending:
            self.pending = pending
            self.history_status = "Saved locally. Syncing to Aiven…"
        else:
            self.history_status = "Start a practice action before saving."

    def update(self, data, clicked, pressed, cue, observations, identity, draw):
        was_open = self.history_open
        self.history_open = HISTORY in clicked
        if self.history_open and not was_open and not self.pending:
            self.pending = self.history.refresh(self.mode)
            self.history_status = "Loading saved attempts…"
        if self.pending and self.pending.done():
            try:
                self.attempts, source = self.pending.result()
            except (OSError, ValueError):
                self.history_status = "Session history unavailable. Reopen history to retry."
            else:
                self.index = 0
                self.history_status = f"{source} · {len(self.attempts)} recent attempts"
            self.pending = None
        if NEXT in clicked and self.history_open and self.attempts:
            self.index = (self.index + 1) % len(self.attempts)

        selected = self.attempts[self.index] if self.history_open and self.attempts else None
        if selected:
            evidence = {name: value for name, value in selected.items() if name not in ("samples", "id")}
            evidence.setdefault("sample_count", len(selected.get("samples", ())))
            context = ("Recorded simulator attempt. Event timestamps are seconds from its start.\n"
                       + json.dumps(evidence, ensure_ascii=False))
            voice_identity = ("saved", selected["id"])
        else:
            context = f"{observations}\nCurrent action: {cue}"
            voice_identity = ("live", identity)
        reply_context = voice_identity if selected else ("live",)
        if self.previous_context != reply_context:
            self.voice_reply = ""
            self.previous_context = reply_context
        toggle = (VOICE in clicked) != self.gemini.active or ord("V") in pressed
        # Keep the microphone off until there is an attempt to ask about.
        if self.history_open and not selected:
            toggle = self.gemini.active
        old_reply = self.gemini.message
        old_voice = self.gemini.pending
        finished = old_voice is not None and old_voice.done()
        reply = self.gemini.update(context, toggle, voice_identity, review=bool(selected))
        if finished or (reply != old_reply and not reply.startswith("Gemini is answering")):
            self.voice_reply = reply

        old_review = self.gemma.pending
        review_requested = REVIEW in clicked or ord("B") in pressed
        review = self.gemma.update(data, review_requested, cue, draw, observations, identity)
        if review_requested and self.gemma.pending is not old_review:
            self.scene_reply = ""
        if review:
            self.scene_reply = review
        if ASSISTANT in clicked:
            if self.popup is None:
                from assistant_window import AssistantWindow

                self.popup = AssistantWindow()
                self.resources.callback(self.popup.root.destroy)
            self.popup.show()
        if self.popup and self.popup.visible:
            question = self.popup.question
            waiting_for_history = self.history_open and not selected
            if question and not waiting_for_history:
                if self.gemini.ask(question, context, voice_identity, review=bool(selected)):
                    self.popup.question = None
                    self.voice_reply = ""
            self.popup.update(self.status(), self.contents(cue, selected),
                              bool(self.gemini.pending or self.gemini.speech or waiting_for_history))
        if self.gemini.pending and self.gemini.pending is not old_voice and not selected:
            return "Gemini"
        if review_requested and self.gemma.pending is not old_review:
            return "Gemma"
        return None

    def status(self):
        if self.gemini.speech:
            return "Gemini is speaking"
        if self.gemini.pending:
            return "Gemini is answering"
        if self.gemini.stream:
            return "Listening. Pause when finished."
        if self.gemma.pending:
            return "Gemma is reviewing the scene"
        return "Microphone off"

    def contents(self, cue, selected):
        lines = []
        if not self.history_open:
            lines.append(f"Current action\n{cue}")
        if self.history_open or self.history_status.startswith(("Saved", "Could", "Start")):
            lines.append(self.history_status)
        if selected:
            from training_history import describe_attempt

            stamp = selected["started_at"][:19].replace("T", " ")
            lines.append(f'Attempt {self.index + 1}/{len(self.attempts)} · {selected.get("site", "Skin").title()}'
                         f"\n{stamp} UTC\n{describe_attempt(selected)}")
        if self.voice_reply:
            lines.append(f"Gemini\n{self.voice_reply}")
        if self.scene_reply:
            lines.append(f"Gemma\n{self.scene_reply}")
        if selected and selected["events"]:
            events = (f'{event["seconds"]:6.2f}s   {event["event"].capitalize()} · {event["detail"]}'
                      for event in selected["events"])
            lines.append("Attempt timeline\n" + "\n".join(events))
        return "\n\n".join(lines)
