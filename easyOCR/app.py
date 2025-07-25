import os
import json
import requests
from flask import Flask, request, render_template,  send_from_directory
from PIL import Image
import numpy as np
import cv2
import pygame
from googletrans import Translator
from gtts import gTTS
from langdetect import detect
import easyocr
import pytesseract

app = Flask(__name__)

# --- Paths ---
BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
TMP_DIR = os.path.join(BASE_DIR, 'tmp')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
for d in (UPLOAD_DIR, TMP_DIR, STATIC_DIR): os.makedirs(d, exist_ok=True)

# CRAFT endpoint
CRAFT_URL = 'http://localhost:6000/detect'

# Initialize translators and readers
translator = Translator()
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
reader_ko_en = easyocr.Reader(['ko', 'en'], gpu=True)
reader_hi_en = easyocr.Reader(['hi', 'en'], gpu=True)
reader_ru_en = easyocr.Reader(['ru', 'en'], gpu=True)
reader_multi = easyocr.Reader(['es', 'fr', 'de', 'it', 'tr'], gpu=True)

# Language codes for TTS
lang_codes = {
    'ko': 'ko', 'hi': 'hi', 'ru': 'ru', 'es': 'es', 'fr': 'fr',
    'de': 'de', 'it': 'it', 'tr': 'tr', 'en': 'en'
}

# OCR & translation utility functions

def detect_language(text):
    try:
        return detect(text)
    except:
        return 'en'


def run_ocr_all(image_np, box_id):
    # Direct OCR without extra preprocessing (handled by craftservice)
    # EasyOCR on the raw crop
    results = []
    results += reader_ko_en.readtext(image_np)
    results += reader_hi_en.readtext(image_np)
    results += reader_ru_en.readtext(image_np)
    results += reader_multi.readtext(image_np)

    # Tesseract fallback if no result
    if not results:
        text = pytesseract.image_to_string(image_np,
            lang='hin+eng+kor+rus+spa+fra+deu+ita+tur', config='--psm 6')
        if text.strip():
            results.append((None, text.strip(), 0.5))

    # Return best match
    if results:
        best = max(results, key=lambda x: x[2] if len(x) == 3 else -1)
        return best  # (bbox, text, confidence)
    return None

# Cropping helpers

def get_coordinates(polygon):
    # polygon as list of points
    if isinstance(polygon, list) and isinstance(polygon[0], list):
        return polygon
    raise KeyError("Unsupported polygon format")


def get_bounding_rect(polygon):
    coords = get_coordinates(polygon)
    xs = [pt[0] for pt in coords]
    ys = [pt[1] for pt in coords]
    return [min(xs), min(ys), max(xs), max(ys)]


@app.route('/result', methods=['POST'])
def result():
    # 1. Save user image
    file = request.files.get('file')
    if not file:
        return "No image provided", 400
    fname = file.filename
    img_path = os.path.join(UPLOAD_DIR, fname)
    file.save(img_path)

    # 2. Send to CRAFT for boxes
    with open(img_path, 'rb') as f:
        craft_resp = requests.post(CRAFT_URL, files={'image': f})
    if craft_resp.status_code != 200:
        return "CRAFT service failed", 500
    boxes = craft_resp.json()

    # 3. Load image array and init pygame
    pil = Image.open(img_path).convert('RGB')
    img_np = np.array(pil)
    pygame.mixer.init()

    # 4. Process each box
    output = []
    all_translated = []
    for idx, box in enumerate(boxes):
        try:
            # cropping
            if isinstance(box[0], list) and len(box) > 2:
                xs = [pt[0] for pt in box]; ys = [pt[1] for pt in box]
                x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
            elif isinstance(box[0], list):
                (x0, y0), (x1, y1) = box
                x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
            else:
                x0, y0, x1, y1 = map(int, box)
            if x1 <= x0 or y1 <= y0:
                continue

            crop_np = img_np[y0:y1, x0:x1]
            ocr = run_ocr_all(crop_np, f"{fname}_{idx}")
            if ocr:
                _, text, conf = ocr
            else:
                text, conf = "", 0.0

            # translate
            lang = detect_language(text)
            translated = translator.translate(text, src=lang, dest='en').text if text else ''
            all_translated.append(translated)

            # audio
            audio_name = f"audio_{idx}.mp3"
            audio_path = os.path.join(STATIC_DIR, audio_name)
            if text:
                gTTS(text=text, lang=lang_codes.get(lang, 'en')).save(audio_path)
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy(): pygame.time.wait(100)
            else:
                audio_name = ''

            output.append({
                'box': box,
                'detected_text': text,
                'translated_text': translated,
                'confidence': conf,
                'audio_file': audio_name
            })
        except Exception as e:
            print(f"Error box {idx}: {e}")
            continue

    # 5. Save JSON
    json_path = os.path.join(STATIC_DIR, 'data.json')
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(output, jf, ensure_ascii=False, indent=4)

    # 6. Save combined audio
    if all_translated:
        combo = ' '.join(all_translated)
        combo_path = os.path.join(STATIC_DIR, 'data.mp3')
        gTTS(text=combo, lang='en').save(combo_path)

    return '', 204

if __name__ == '__main__':
    app.run(debug=True, port=5000)
