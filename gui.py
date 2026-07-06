import customtkinter as ctk
import threading
import time
from speak import speak
from listen import listen_for_command
from main import compiled_app as graph
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

class TESrACTUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("TESrACT — Agentic Interface")
        self.geometry("600x800")
        ctk.set_appearance_mode("dark")
        self.attributes("-topmost", True) 

        self.status_label = ctk.CTkLabel(self, text="TESrACT • ONLINE", font=("Orbitron", 22, "bold"), text_color="#00FFCC")
        self.status_label.pack(pady=40)

        self.log_box = ctk.CTkTextbox(self, width=550, height=550, corner_radius=15, font=("Consolas", 13))
        self.log_box.pack(pady=10)

        self.history: list[BaseMessage] = []
        self._config: RunnableConfig = {"configurable": {"thread_id": "gui_session"}}
        threading.Thread(target=self.run_assistant, daemon=True).start()

    def set_status(self, status, color="#00FFCC"):
        self.after(0, lambda: self.status_label.configure(text=status, text_color=color))

    def update_log(self, text, tag="TESrACT"):
        prefix = ">>> " if tag == "USER" else "TESrACT: "
        self.after(0, lambda: self._perform_log_update(f"{prefix}{text}\n\n"))

    def _perform_log_update(self, text):
        self.log_box.insert("end", text)
        self.log_box.see("end")

    def run_assistant(self):
        speak("TESrACT visual interface active. At your service.", wait_for_speech=True)
        while True:
            self.set_status("LISTENING...", "#3399FF")
            command = listen_for_command()
            if command:
                self.update_log(command, tag="USER")
                self.set_status("ANALYZING...", "#FFFF00")
                try:
                    # Simple control sync for the desktop GUI (full privileges handled in main web flow)
                    current_state = graph.get_state(self._config)
                    state_values = current_state.values if current_state else None
                    ctrl = bool((state_values or {}).get("control_allowed", False))
                    result = graph.invoke(
                        {
                            "messages": self.history + [HumanMessage(content=command)],
                            "control_allowed": ctrl,
                        },
                        config=self._config,
                    )
                    self.history = list(result.get("messages", []))
                    final_res = next((msg.content for msg in reversed(self.history) if isinstance(msg.content, str)), "")
                    if final_res:
                        self.set_status("EXECUTING...", "#00FF66")
                        self.update_log(final_res, tag="TESrACT")
                        speak(final_res, wait_for_speech=True)
                except Exception as e:
                    self.update_log(f"ERROR: {e}", tag="ERROR")
            time.sleep(0.1)

if __name__ == "__main__":
    TESrACTUI().mainloop()