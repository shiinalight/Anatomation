"""The AI coach: Gemini answers questions (voice or typed), Gemma writes debriefs, and the
assistant window shows the next step, the scores and the replies.

Everything the coach knows comes from the simulator: the station's next step and hand, the
recent step events, the last attempt's task scores, and the best results on Aiven.
"""

import json

from gemma_review import GemmaReviewer, backend, load_api_key

TOOL_KEYS = "1 scalpel, 2 needle holder, 3 forceps, 4 scissors, G take or set down"
CONTROLS = (
    "Controls: the keyboard moves the hand the next step needs (Z swaps hands). " + TOOL_KEYS + ". "
    "Arrows move, Q raises, E lowers. N aligns the tool on its target. F grips with forceps or an empty "
    "hand, or passes the needle. T throws a knot (three throws). X trims with scissors. "
    "R starts over, H re-aims the gloves. Panel: Autopilot replays the best recorded attempt."
)


def describe_tasks(summary, store=None):
    lines = []
    for task in summary.get("tasks", []):
        best = store.benchmark(summary["site"], task["task"]) if store else None
        compare = (f" (best {best['seconds']:.0f} s, {best['score']:.0f}/100)" if best else "")
        lines.append(f"- {task['task']}: {task['score']:.0f}/100, {task['seconds']:.0f} s{compare}, "
                     f"travel {task['travel_cm']:.0f} cm, {task['rejected']} refused, "
                     f"needle {task['needle_error_mm']:.1f} mm off")
    return "\n".join(lines)


def attempt_facts(summary, store=None):
    head = (f"Attempt at the {summary['site']} site: {'complete' if summary['complete'] else 'incomplete'}, "
            f"score {summary['score']:.0f}/100, {summary['duration_s']:.0f} s, "
            f"{summary.get('stitches', 0)} stitches.")
    return head + "\nTasks (best recorded results in brackets):\n" + (describe_tasks(summary, store) or "- none")


class Coach:
    def __init__(self, store):
        self.store = store
        self.key = load_api_key()
        self.voice = None
        self.voice_error = None
        if self.key:
            try:
                from gemini_voice import GeminiVoice
                self.voice = GeminiVoice(self.key)
            except (ImportError, OSError) as error:     # sounddevice / PortAudio missing
                self.voice_error = f"Voice unavailable ({error})"
        self.gemma = GemmaReviewer() if backend() else None
        self.window = None
        self.gemma_reply = ""
        self.last_attempt = None
        self.recent = []            # recent step events, for context

    @property
    def available(self):
        return self.key is not None or self.gemma is not None

    # --- context -------------------------------------------------------------------------
    def note_events(self, events):
        for kind, detail in events:
            if kind in ("setdown", "pickup"):
                continue
            text = detail.get("reason") if kind == "rejected" else json.dumps(detail)
            self.recent.append(f"{kind}: {text}")
        del self.recent[:-10]

    def context(self, station, autopilot):
        side, step = station.plan()
        hands = ", ".join(f"{h.side} hand holds {h.tool}" for h in station.hands.values())
        lines = [
            f"Site: {station.site.name}. " + ("The autopilot is replaying the best attempt. " if autopilot else ""),
            f"Current action: {(side.capitalize() + ' hand: ') if side else ''}{step}",
            f"Hands: {hands}. Stitches tied: {len(station.stitches)}.",
            CONTROLS,
        ]
        if self.recent:
            lines.append("Recent events: " + "; ".join(self.recent[-6:]))
        if self.last_attempt:
            lines.append("Last finished attempt:\n" + attempt_facts(self.last_attempt, self.store))
        return "\n".join(lines)

    # --- per frame -------------------------------------------------------------------------
    def update(self, station, attempt_id, autopilot, voice_toggled, review_clicked,
               assistant_clicked, render):
        context = self.context(station, autopilot)
        if assistant_clicked and (self.window is None or not self.window.alive):
            from assistant_window import AssistantClient
            self.window = AssistantClient()
        elif assistant_clicked:
            self.window.show()          # already open (maybe behind the viewer): bring it forward
        if self.voice is not None:
            question = self.window.question() if self.window is not None else None
            if question:
                self.voice.ask(question, context, attempt_id)
            self.voice.update(context, voice_toggled, attempt_id)
        if review_clicked and self.gemma is not None:
            facts = context
            self.gemma_reply = "Gemma is reviewing the scene..."
            self.gemma.request(render(), facts)
        if self.gemma is not None:
            reply = self.gemma.poll()
            if reply:
                self.gemma_reply = reply
        if self.window is not None:
            self.window.update(self.status(), self.contents(station), self.busy)

    def attempt_finished(self, summary, render):
        """A finished attempt: keep it for context and ask Gemma for a debrief."""
        self.last_attempt = summary
        if self.gemma is not None and summary["complete"]:
            self.gemma_reply = "Gemma is writing the debrief..."
            self.gemma.request(render(), attempt_facts(summary, self.store))

    @property
    def voice_on(self):
        return self.voice is not None and self.voice.active

    @property
    def busy(self):
        return bool(self.voice and (self.voice.pending or self.voice.speech))

    def status(self):
        if not self.available:
            return "AI off: set SURGERY_GEMMA_API_KEY in .env"
        if self.voice and self.voice.speech:
            return "Gemini is speaking"
        if self.voice and self.voice.pending:
            return "Gemini is answering"
        if self.voice and self.voice.stream:
            return "Listening: ask, then pause"
        if self.gemma and self.gemma.busy:
            return "Gemma is reviewing"
        return self.voice_error or ("Voice off (AI Voice button or V)" if self.voice else "Gemma ready")

    def contents(self, station):
        side, step = station.plan()
        parts = [f"Next step\n{(side.capitalize() + ' hand: ') if side else ''}{step}"]
        if self.voice and self.voice.last_answer:
            parts.append(f"Gemini\n{self.voice.last_answer}")
        elif self.voice and self.voice.message:
            parts.append(f"Gemini\n{self.voice.message}")
        if self.gemma_reply:
            parts.append(f"Gemma\n{self.gemma_reply}")
        if self.last_attempt:
            parts.append("Last attempt\n" + attempt_facts(self.last_attempt, self.store))
        parts.append(f"Storage\n{self.store.status}")
        return "\n\n".join(parts)

    def close(self):
        if self.voice is not None:
            self.voice.close()
        if self.gemma is not None:
            self.gemma.close()
        if self.window is not None:
            self.window.close()
