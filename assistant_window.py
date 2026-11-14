"""Small, non-modal assistant for the native MuJoCo viewer."""

import tkinter as tk
from tkinter import ttk


class AssistantWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.title("SurgiSense | AI Assistant")
        self.root.geometry("440x370")
        self.root.minsize(340, 250)
        self.root.protocol("WM_DELETE_WINDOW", self.hide)
        self.visible = False
        self.last_content = None
        self.last_status = None
        self.last_busy = None
        self.question = None

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

    def show(self):
        self.visible = True
        self.root.deiconify()
        self.root.lift()

    def hide(self):
        self.visible = False
        self.root.withdraw()

    def send(self, _event=None):
        question = self.entry.get().strip()
        if question and not self.last_busy:
            self.question = question[:500]
            self.entry.delete(0, "end")
        return "break"

    def update(self, status, content, busy):
        if not self.visible:
            return
        if status != self.last_status:
            self.status.configure(text=status)
            self.last_status = status
        if content != self.last_content:
            self.text.configure(state="normal")
            self.text.delete("1.0", "end")
            self.text.insert("1.0", content)
            self.text.configure(state="disabled")
            self.last_content = content
        if busy != self.last_busy:
            self.button.configure(state="disabled" if busy else "normal")
            self.last_busy = busy
        self.root.update()
