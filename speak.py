import requests
import pygame
import time
import os

COLAB_URL = "https://fluidic-unevil-camryn.ngrok-free.dev/speak"

def speak(text, wait_for_speech=True):
    if not text:
        return
    
    try:
        # 1. Send text to the Cloud
        response = requests.post(COLAB_URL, json={"text": text}, timeout=60)
        
        if response.status_code == 200:
            # --- FIX: Ensure pygame releases the file before writing ---
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
                pygame.mixer.music.unload() # Releases the file handle
            
            # Save the new voice file
            with open("temp_voice.wav", "wb") as f:
                f.write(response.content)
            
            # 2. Play the audio locally
            if not pygame.mixer.get_init():
                pygame.mixer.init()
                
            pygame.mixer.music.load("temp_voice.wav")
            pygame.mixer.music.play()
            
            # 3. Mute the listener
            if wait_for_speech:
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                # Unload again after finished to be safe for the next run
                pygame.mixer.music.unload()
                    
    except Exception as e:
        print(f"Cloud Voice Error: {e}")