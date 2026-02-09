import pyttsx3
import threading
from typing import Any

class Speaker:
    def __init__(self):
        self.lock = threading.Lock()

    def _speak_worker(self, text):
        """Initialize and run engine entirely within the thread."""
        with self.lock:
            try:
                # Re-initializing inside the thread solves the Windows SAPI5 hang
                engine = pyttsx3.init()
                engine.setProperty('rate', 180)
                voices: Any = engine.getProperty('voices')
                if voices:
                    engine.setProperty('voice', voices[0].id)
                
                print(f"JARVIS: {text}")
                engine.say(text)
                engine.runAndWait()
                # Clean up to release the driver
                engine.stop() 
                del engine 
            except Exception as e:
                print(f"TTS Error: {e}")

    def speak(self, text, wait_for_speech=False):
        if not text:
            return
            
        thread = threading.Thread(target=self._speak_worker, args=(text,), daemon=not wait_for_speech)
        thread.start()

        if wait_for_speech:
            thread.join()

_speaker_instance = Speaker()

def speak(text, wait_for_speech=False):
    _speaker_instance.speak(text, wait_for_speech)