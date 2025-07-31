import json
import os
from googletrans import Translator, LANGUAGES
from langdetect import detect

# Define paths
json_path = "text.json"  # Input JSON file containing the word
output_path = "translated_output.json"  # Output JSON file for translated text

# Create output directory if it doesn’t exist
os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

# Load the word from the JSON file
if not os.path.exists(json_path):
    raise FileNotFoundError(f"JSON file not found: {json_path}")

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract the word and target language
if isinstance(data, str):
    word = data  # If JSON is a simple string, use it as the word
    target_lang = "en"  # Default to English for string input
elif isinstance(data, dict):
    word = data.get("word")
    if not word or not isinstance(word, str):
        raise ValueError("Dictionary input must contain a 'word' key with a string value")
    target_lang = data.get("target_lang", "en")  # Default to English if not specified
else:
    raise ValueError("Input must be a string or a dictionary with a 'word' key")

# Validate word
if not isinstance(word, str):
    raise ValueError("Word must be a string")

# Initialize translator
translator = Translator()

# Detect the source language of the word
try:
    detected_source = detect(word)
    print(f"Detected source language: {detected_source} ({LANGUAGES.get(detected_source, 'Unknown')})")
except Exception as e:
    print(f"Language detection failed: {e}, using 'auto' detection by googletrans")
    detected_source = None

# Validate target language
if target_lang not in LANGUAGES:
    print(f"Unsupported language code: {target_lang}. Defaulting to English.")
    target_lang = "en"

# Check if source language is English; if so, skip translation
if detected_source == "en":
    print(f"Source language is English; no translation needed for '{word}'.")
    translated_text = word
    translated_src = detected_source
else:
    # Translate the word with detected source language
    try:
        translated = translator.translate(word, src=detected_source, dest=target_lang)
        translated_text = translated.text
        translated_src = translated.src
        print(f"Translating '{word}' from {LANGUAGES.get(translated_src, 'auto')} to '{translated_text}' in {LANGUAGES[target_lang]}")
        # Warn if the detected source differs from googletrans's detection
        if detected_source and translated_src != detected_source:
            print(f"Warning: Language detection mismatch - detected {detected_source}, googletrans used {translated_src}")
    except Exception as e:
        print(f"Translation error: {e}")
        translated_text = f"Translation failed: {e}"
        translated_src = "unknown" if not detected_source else detected_source

# Save the translated text to a JSON file
output_data = {
    "original_word": word,
    "translated_word": translated_text,
    "target_language": target_lang,
    "language_name": LANGUAGES.get(target_lang, "Unknown"),
    "detected_source_language": translated_src,
    "note": "No translation performed (source is English)" if detected_source == "en" else ""
}
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print(f"Translation saved to {output_path}")