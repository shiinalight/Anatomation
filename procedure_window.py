"""The Procedures window: a dropdown of every procedure, run as its own process (Tk must own
its process's main thread, which mjpython gives to the MuJoCo viewer).

The simulator sends {"procedures": [[key, label], ...], "current": key, "checklist": text};
the window sends {"procedure": key} when the trainee picks one.
"""

import json
import queue
import sys
import threading
import tkinter as tk
from tkinter import ttk

from assistant_window import FloatOverFullScreen, WindowProcess


class ProcedureClient(WindowProcess):
    def __init__(self, procedures, current):
        from pathlib import Path
        super().__init__(Path(__file__).resolve())
        self.procedures = [[p.key, p.label] for p in procedures]
        self.update(current, "")

    def update(self, current, checklist):
        self.send({"procedures": self.procedures, "current": current, "checklist": checklist})

    def chosen(self):
        picks = [m["procedure"] for m in self.received() if m.get("procedure")]
        return picks[-1] if picks else None


class ProcedureWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Anatomation | Procedures")
        self.root.geometry("380x330+40+80")
        self.inbox = queue.SimpleQueue()
        self.keys, self.labels, self.current = [], [], None

        panel = ttk.Frame(self.root, padding=12)
        panel.pack(fill="both", expand=True)
        ttk.Label(panel, text="Procedure").pack(anchor="w")
        self.choice = ttk.Combobox(panel, state="readonly", height=10)
        self.choice.pack(fill="x", pady=(4, 10))
        self.choice.bind("<<ComboboxSelected>>", self.pick)
        ttk.Label(panel, text="Steps").pack(anchor="w")
        self.steps = tk.Text(panel, wrap="word", height=12, state="disabled", padx=6, pady=6)
        self.steps.pack(fill="both", expand=True, pady=(4, 0))
        ttk.Label(panel, foreground="#666",
                  text="Choosing a procedure saves the current attempt and starts fresh. P cycles.").pack(
                      anchor="w", pady=(8, 0))
        threading.Thread(target=self._read, daemon=True).start()
        self.root.after(100, self._apply)
        self.floating = FloatOverFullScreen(self.root)     # over a full-screen viewer too

    def _read(self):
        for line in sys.stdin:
            try:
                self.inbox.put(json.loads(line))
            except ValueError:
                continue
        self.inbox.put(None)

    def _apply(self):
        while not self.inbox.empty():
            message = self.inbox.get_nowait()
            if message is None:
                self.root.destroy()
                return
            if "raise" in message:
                self.floating.bring_forward()
                continue
            pairs = message.get("procedures", [])
            if [k for k, _ in pairs] != self.keys:
                self.keys = [k for k, _ in pairs]
                self.labels = [label for _, label in pairs]
                self.choice.configure(values=self.labels)
            current = message.get("current")
            if current in self.keys and current != self.current:
                self.current = current
                self.choice.current(self.keys.index(current))
            self.steps.configure(state="normal")
            self.steps.delete("1.0", "end")
            self.steps.insert("1.0", message.get("checklist", ""))
            self.steps.configure(state="disabled")
        self.root.after(100, self._apply)

    def pick(self, _event=None):
        index = self.choice.current()
        if 0 <= index < len(self.keys) and self.keys[index] != self.current:
            print(json.dumps({"procedure": self.keys[index]}), flush=True)


if __name__ == "__main__":
    ProcedureWindow().root.mainloop()
