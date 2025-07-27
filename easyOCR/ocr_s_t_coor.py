import easyocr
from PIL import Image
from googletrans import Translator
import cv2
import os
from gtts import gTTS
import pygame
import time
from langdetect import detect
import numpy as np
import json
import pytesseract

# Define paths
json_path = "result/coords_9e5e8ee8439d4b2fa9fe704b1e3dd4d7.json"  # CRAFT coordinates JSON
image_path = "result/res_9e5e8ee8439d4b2fa9fe704b1e3dd4d7.jpg"     # Input image
output_folder = "output_folder"


# Create output directories if they don’t exist
os.makedirs(output_folder, exist_ok=True)


# Initialize OCR readers
reader_ko_en = easyocr.Reader(['ko', 'en'], gpu=True)
reader_hi_en = easyocr.Reader(['hi', 'en'], gpu=True)
reader_ru_en = easyocr.Reader(['ru', 'en'], gpu=True)
reader_multi = easyocr.Reader(['es', 'fr', 'de', 'it', 'tr'], gpu=True)

# Set Tesseract path (adjust based on your system)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update this path

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
def run_ocr_all(image_np, box_id):
    # Enhance image
    scale_factor = 2
    resized = cv2.resize(image_np, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    enhanced = cv2.equalizeHist(sharpened)
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Try EasyOCR with multiple versions
    images_to_process = [resized, thresh, enhanced]
    results = []
    for img in images_to_process:
        results.extend(reader_ko_en.readtext(img))
        results.extend(reader_hi_en.readtext(img))
        results.extend(reader_ru_en.readtext(img))
        results.extend(reader_multi.readtext(img))

    # Fallback to Tesseract if no results
    if not results:
        text = pytesseract.image_to_string(resized, lang='hin+eng+kor+rus+spa+fra+deu+ita+tur', config='--psm 6')
        if text.strip():
            results.append((None, text.strip(), 0.5))  # Placeholder confidence

    # Save image if no text detected
    if not results:
        cv2.imwrite(os.path.join(output_folder, f"no_text_detected_{box_id}.png"), image_np)

    # Return the best result
    if results:
        best_result = max(results, key=lambda x: x[2] if len(x) == 3 else -1)
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
        cropped_image_np = np.array(cropped_image)

        # Run OCR
        best_result = run_ocr_all(cropped_image_np, f"{image_base}_{idx}")
        if best_result and len(best_result) == 3:
            bbox, text, prob = best_result
            print(f"Region {idx} - Detected Text: {text} (Confidence: {prob:.2f})")

            # Detect language and generate TTS
            

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
                
            })
        else:
            output_results.append({
                "coordinates": get_coordinates(polygon),
                "detected_text": "No text detected",
                "translated_text": "N/A",
                "confidence": 0.0,
                
            })
    except KeyError as e:
        print(f"Error processing region {idx}: Missing coordinates - {e}")
        output_results.append({
            "coordinates": [],
            "detected_text": f"Error: {e}",
            "translated_text": "N/A",
            "confidence": 0.0,
            
        })
    except Exception as e:
        print(f"Error processing region {idx}: {e}")
        output_results.append({
            "coordinates": get_coordinates(polygon) if "coordinates" in polygon else [],
            "detected_text": f"Error: {e}",
            "translated_text": "N/A",
            "confidence": 0.0,
            
        })

# Save results to JSON
output_json_path = os.path.join(output_folder, "results.json")
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump({"image": image_path, "texts": output_results}, f, ensure_ascii=False, indent=4)

print(f"Results saved to {output_json_path}")