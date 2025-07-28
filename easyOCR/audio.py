import json
import os
from gtts import gTTS
import pygame
import time
from langdetect import detect

# Define paths
json_path = "input_word.json"  # Input JSON file containing the word
audio_output_dir = "audio_output"

# Create output directory if it doesn’t exist
os.makedirs(audio_output_dir, exist_ok=True)

# Load the word from the JSON file
if not os.path.exists(json_path):
    raise FileNotFoundError(f"JSON file not found: {json_path}")

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract the word (assuming a simple structure like {"word": "value"} or just the word as a string)
word = data.get("word", data) if isinstance(data, dict) else data
if not isinstance(word, str):
    raise ValueError("Input must be a string or a dictionary with a 'word' key containing a string")

# Detect the language of the word
try:
    detected_lang = detect(word)
except Exception as e:
    print(f"Language detection failed: {e}, using English as fallback")
    detected_lang = 'en'

# Language code mapping for gTTS (covering languages from your previous context)
lang_codes = {
    'hi': 'hi',  # Hindi
    'ko': 'ko',  # Korean
    'ru': 'ru',  # Russian
    'es': 'es',  # Spanish
    'fr': 'fr',  # French
    'de': 'de',  # German
    'it': 'it',  # Italian
    'tr': 'tr',  # Turkish
    'en': 'en'   # English
}

# Map detected language to gTTS-supported code, fallback to English
tts_lang = lang_codes.get(detected_lang, 'en')

# Generate speech in the detected language
audio_file = os.path.join(audio_output_dir, f"output_audio.mp3")
try:
    tts = gTTS(text=word, lang=tts_lang)
    tts.save(audio_file)
    print(f"Audio saved to {audio_file} in {tts_lang} accent")

    # Play the audio
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)
    pygame.mixer.music.unload()
except Exception as e:
    print(f"Error generating or playing audio: {e}")