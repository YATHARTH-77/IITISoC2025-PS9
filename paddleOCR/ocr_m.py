import cv2
from paddleocr import PaddleOCR
from PIL import Image
from googletrans import Translator
import os
from gtts import gTTS
import pygame
import time
from langdetect import detect
import numpy as np
import json

# Define paths
json_path = "result/coords_pre_a-street-scene-from-amalfi-town-italy-street-signs-and-street-lamp-CC63JF.json"  # CRAFT coordinates JSON
image_path = "result/res_pre_a-street-scene-from-amalfi-town-italy-street-signs-and-street-lamp-CC63JF.jpg"    # Input image
output_folder = "output_folder"
audio_output_dir = "audio_output"

# Create output directories if they don’t exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(audio_output_dir, exist_ok=True)

# Initialize PaddleOCR readers with angle classification (CPU mode)
reader_ko = PaddleOCR(lang='korean', use_angle_cls=True, use_gpu=False)  # Korean
reader_hi = PaddleOCR(lang='devanagari', use_angle_cls=True, use_gpu=False)  # Hindi
reader_ru = PaddleOCR(lang='cyrillic', use_angle_cls=True, use_gpu=False)  # Russian
reader_multi = PaddleOCR(lang='latin', use_angle_cls=True, use_gpu=False)  # Spanish, French, German, Italian, Turkish

# Initialize translator
translator = Translator()

# Language code mapping for gTTS
lang_codes = {
    'ko': 'ko', 'hi': 'hi', 'ru': 'ru', 'es': 'es', 'fr': 'fr',
    'de': 'de', 'it': 'it', 'tr': 'tr', 'en': 'en'
}

# Load the image
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")
image = Image.open(image_path)

# Load coordinates from the JSON file and debug structure
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
print("JSON structure:", json.dumps(data, indent=2))
if isinstance(data, list):
    polygons = [item["boxes"] for item in data if "boxes" in item]  # Extract all boxes lists
    polygons = [poly for sublist in polygons for poly in sublist]  # Flatten the list
elif isinstance(data, dict) and "boxes" in data:
    polygons = data["boxes"]
else:
    raise ValueError("JSON format invalid. Expected a list of dicts with 'boxes' key or a dict with 'boxes' key.")

# Function to extract coordinates with fallback
def get_coordinates(polygon):
    if isinstance(polygon, dict) and "coordinates" in polygon:
        return polygon["coordinates"]
    raise KeyError(f"No 'coordinates' key found in {polygon}")

# Function to compute bounding rectangle from coordinates
def get_bounding_rect(polygon):
    coords = get_coordinates(polygon)
    xs = [point[0] for point in coords]
    ys = [point[1] for point in coords]
    return [min(xs), min(ys), max(xs), max(ys)]  # [x_min, y_min, x_max, y_max]

# Function to run OCR with all readers and get the best result
def run_ocr_all(image_np):
    results = []
    print(f"Processing image region")
    
    # Process with each reader
    readers = [reader_ko, reader_hi, reader_ru, reader_multi]
    for reader in readers:
        result = reader.ocr(image_np, cls=True)
        if result and len(result) > 0 and result[0]:
            results.extend([item[1] for item in result[0] if len(item[1]) == 2])
        else:
            print("No results from one of the readers")
    
    # Debug: Print all results for inspection
    print("All results:", results)
    
    # Find the best result (highest confidence)
    if results:
        best_result = max(results, key=lambda x: x[1] if len(x) == 2 else -1)
        return best_result
    return None

# Function to detect language
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'en'  # Fallback to English

# Process each text region
output_results = []
image_base = os.path.basename(image_path).split('.')[0]
for idx, polygon in enumerate(polygons):
    try:
        # Extract coordinates and crop image
        x_min, y_min, x_max, y_max = get_bounding_rect(polygon)
        cropped_image = image.crop((x_min, y_min, x_max, y_max)).convert('RGB')
        cropped_image_np = np.array(cropped_image)  # Convert to NumPy array
        
        # Run OCR
        best_result = run_ocr_all(cropped_image_np)
        if best_result and len(best_result) == 2:
            text, prob = best_result
            print(f"Region {idx} - Detected Text: {text} (Confidence: {prob:.2f})")
            
            # Detect language and generate TTS
            detected_lang = detect_language(text)
            tts_lang = lang_codes.get(detected_lang, 'en')
            audio_file = os.path.join(audio_output_dir, f"audio_{image_base}_{idx}.mp3")
            try:
                tts = gTTS(text=text, lang=tts_lang)
                tts.save(audio_file)
                time.sleep(0.5)
                pygame.mixer.init()
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                pygame.mixer.music.unload()
            except Exception as e:
                audio_file = f"TTS failed: {e}"
            
            # Translate to English
            try:
                translated_text = translator.translate(text, dest='en').text
            except Exception as e:
                translated_text = f"Translation failed: {e}"
            
            # Store result
            output_results.append({
                "coordinates": get_coordinates(polygon),
                "detected_text": text,
                "translated_text": translated_text,
                "confidence": prob,
                "audio_file": audio_file
            })
        else:
            output_results.append({
                "coordinates": get_coordinates(polygon),
                "detected_text": "No text detected",
                "translated_text": "N/A",
                "confidence": 0.0,
                "audio_file": "N/A"
            })
    except KeyError as e:
        print(f"Error processing region {idx}: Missing coordinates - {e}")
        output_results.append({
            "coordinates": [],
            "detected_text": f"Error: {e}",
            "translated_text": "N/A",
            "confidence": 0.0,
            "audio_file": "N/A"
        })
    except Exception as e:
        print(f"Error processing region {idx}: {e}")
        output_results.append({
            "coordinates": get_coordinates(polygon) if "coordinates" in polygon else [],
            "detected_text": f"Error: {e}",
            "translated_text": "N/A",
            "confidence": 0.0,
            "audio_file": "N/A"
        })

# Save results to JSON
output_json_path = os.path.join(output_folder, "results.json")
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump({"image": image_path, "texts": output_results}, f, ensure_ascii=False, indent=4)

print(f"Results saved to {output_json_path}")