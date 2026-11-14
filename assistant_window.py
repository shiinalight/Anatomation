"""The AI assistant window, run as its own process.

On macOS, mjpython gives the main thread to the MuJoCo viewer and Tk must own the main
thread of its process, so the window cannot live inside the simulator. The simulator starts
this script and talks to it in JSON lines: it sends {"status", "content", "busy"} on stdin,
and the window sends {"question": ...} on stdout when the trainee types one.
(Window layout from the ai-assistant-aiven branch, hadi-ce04.)
"""

import json
import queue
import sys
import threading
import tkinter as tk
from tkinter import ttk


class WindowProcess:
    """Simulator side of a Tk window running as its own process: send() a JSON message
    (repeats are skipped), received() the messages the window sent."""

    def __init__(self, script):
        import subprocess
        from pathlib import Path
        python = Path(sys.executable)
        plain = python.with_name("python3")       # mjpython's launcher is not needed for Tk
        self.process = subprocess.Popen(
            [str(plain if plain.exists() else python), str(script)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.inbox = queue.SimpleQueue()
        self.last = None
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.process.stdout:
            try:
                self.inbox.put(json.loads(line))
            except ValueError:
                continue

    @property
    def alive(self):
        return self.process.poll() is None

    def send(self, message):
        if message == self.last or not self.alive:
            return
        self.last = message
        try:
            self.process.stdin.write(json.dumps(message) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def received(self):
        while not self.inbox.empty():
            yield self.inbox.get_nowait()

    def close(self):
        if self.alive:
            self.process.terminate()


class AssistantClient(WindowProcess):
    """The assistant window: status, replies and typed questions."""

    def __init__(self):
        from pathlib import Path
        super().__init__(Path(__file__).resolve())
        self.questions = []

    def update(self, status, content, busy):
        self.send({"status": status, "content": content, "busy": busy})

    def question(self):
        self.questions += [m["question"] for m in self.received() if m.get("question")]
        return self.questions.pop(0) if self.questions else None


class AssistantWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SurgiSense | AI Assistant")
        self.root.geometry("480x460")
        self.root.minsize(340, 260)
        self.inbox = queue.SimpleQueue()
        self.busy = False

        panel = ttk.Frame(self.root, padding=12)
        panel.pack(fill="both", expand=True)
        self.status = ttk.Label(panel)
        self.status.pack(fill="x", pady=(0, 8))
        body = ttk.Frame(panel)
        body.pack(fill="both", expand=True)
        self.text = tk.Text(body, wrap="word", state="disabled", padx=8, pady=8, height=10)
        scroll = ttk.Scrollbar(body, command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.text.pack(fill="both", expand=True)
        composer = ttk.Frame(panel)
        composer.pack(fill="x", pady=(8, 0))
        self.entry = ttk.Entry(composer)
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", self.send)
        self.button = ttk.Button(composer, text="Ask Gemini", command=self.send)
        self.button.pack(side="right", padx=(6, 0))
        threading.Thread(target=self._read, daemon=True).start()
        self.root.after(100, self._apply)

    def _read(self):
        for line in sys.stdin:
            try:
                self.inbox.put(json.loads(line))
            except ValueError:
                continue
        self.inbox.put(None)        # the simulator closed

    def _apply(self):
        while not self.inbox.empty():
            message = self.inbox.get_nowait()
            if message is None:
                self.root.destroy()
                return
            self.status.configure(text=message.get("status", ""))
            self.text.configure(state="normal")
            self.text.delete("1.0", "end")
            self.text.insert("1.0", message.get("content", ""))
            self.text.configure(state="disabled")
            self.busy = bool(message.get("busy"))
            self.button.configure(state="disabled" if self.busy else "normal")
        self.root.after(100, self._apply)

    def send(self, _event=None):
        question = self.entry.get().strip()
        if question and not self.busy:
            print(json.dumps({"question": question[:500]}), flush=True)
            self.entry.delete(0, "end")
        return "break"


if __name__ == "__main__":
    AssistantWindow().root.mainloop()
