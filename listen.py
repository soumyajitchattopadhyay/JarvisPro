import speech_recognition as sr

# Initialize the recognizer once, not in the function.
r = sr.Recognizer()
r.pause_threshold = 0.8 # Default is 0.8, 1.0 is a good starting point

def listen_for_command():
    with sr.Microphone() as source:
        print("Listening...")
        r.adjust_for_ambient_noise(source, duration=1) 
        
        try:
            # timeout: how long to wait for speech to START
            # phrase_time_limit: how long a single sentence can be
            audio = r.listen(source, timeout=10, phrase_time_limit=20)
        except sr.WaitTimeoutError:
            return None

    try:
        print("Recognizing...")
        # Add 'type: ignore' to suppress the false IDE error
        command = r.recognize_google(audio, language='en-gb') # type: ignore
        print(f"You said: {command}\n")
        return command.lower()
    except (sr.UnknownValueError, sr.RequestError):
        return None

# --- Test it! ---
if __name__ == "__main__":
    print("Say something, I'm listening...")
    command = listen_for_command()
    if command:
        print(f"Python captured: {command}")