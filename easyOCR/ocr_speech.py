import easyocr
from PIL import Image
from googletrans import Translator
import cv2
import os
from gtts import gTTS
import pygame
import time

# Initialize separate readers due to language compatibility restrictions
reader_ko_en = easyocr.Reader(['ko', 'en'], gpu=True)  # Korean with English only
reader_hi_en = easyocr.Reader(['hi', 'en'], gpu=True)  # Hindi with English only
reader_ru_en = easyocr.Reader(['ru', 'en'], gpu=True)  # Russian with English only
reader_multi = easyocr.Reader(['es', 'fr', 'de', 'it', 'tr'], gpu=True)  # Other compatible languages

# Initialize translator
translator = Translator()

# Language code mapping for gTTS
lang_codes = {
    'ko': 'ko', 'hi': 'hi', 'ru': 'ru', 'es': 'es', 'fr': 'fr',
    'de': 'de', 'it': 'it', 'tr': 'tr', 'en': 'en'
}

# Function to run OCR with all readers and get the best result
def run_ocr_all(image_path):
    results = []
    print(f"Processing image: {image_path}")
    
    # Pre-process image
    image_cv = cv2.imread(image_path)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite('temp.png', thresh)
    
    # Process with Korean + English reader
    ko_en_results = reader_ko_en.readtext('temp.png')
    results.extend(ko_en_results)
    
    # Process with Hindi + English reader
    hi_en_results = reader_hi_en.readtext('temp.png')
    results.extend(hi_en_results)
    
    # Process with Russian + English reader
    ru_en_results = reader_ru_en.readtext('temp.png')
    results.extend(ru_en_results)
    
    # Process with multi-language reader
    multi_results = reader_multi.readtext('temp.png')
    results.extend(multi_results)
    
    os.remove('temp.png')
    
    # Find the best result (highest confidence)
    best_result = max(results, key=lambda x: x[2] if len(x) == 3 else -1, default=None)
    return best_result

# Function to detect language (simplified approximation based on reader)
def detect_language(text):
    if any(ord(c) >= 0xAC00 and ord(c) <= 0xD7AF for c in text):  # Korean Hangul range
        return 'ko'
    elif any(0x0900 <= ord(c) <= 0x097F for c in text):  # Hindi Devanagari range
        return 'hi'
    elif any(0x0400 <= ord(c) <= 0x04FF for c in text):  # Russian Cyrillic range
        return 'ru'
    elif any(c.isalpha() and c.lower() in 'abcdefghijklmnopqrstuvwxyz' for c in text):
        return 'en'  # Default to English for Latin scripts
    return 'en'  # Fallback

# Example usage
image_path = 'test_images/image_e2.png'  # Replace with your test image
image = Image.open(image_path)
image.show()  # View the image for verification

# Run OCR and get the best result
best_result = run_ocr_all(image_path)

# Display detected and translated text, save results, and add TTS
if best_result and len(best_result) == 3:
    bbox, text, prob = best_result
    print(f"Detected Text: {text} (Confidence: {prob:.2f})")
    try:
        # Detect language and generate TTS
        detected_lang = detect_language(text)
        tts_lang = lang_codes.get(detected_lang, 'en')
        tts = gTTS(text=text, lang=tts_lang)
        tts.save("detected_audio.mp3")
        time.sleep(0.5)  # Wait for file to be written
        pygame.mixer.init()
        pygame.mixer.music.load("detected_audio.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
        pygame.mixer.music.unload()  # Close the audio stream
        os.remove("detected_audio.mp3")
        
        # Translate to English
        translated = translator.translate(text, dest='en')
        print(f"Translated to English: {translated.text}")
        with open('ocr_results.txt', 'a', encoding='utf-8') as f:
            f.write(f"Image: {image_path}, Detected: {text}, Confidence: {prob:.2f}, Translated: {translated.text}\n")
    except Exception as e:
        print(f"Translation or TTS failed: {e}, using original text: {text}")
else:
    print("No text detected or incomplete data")