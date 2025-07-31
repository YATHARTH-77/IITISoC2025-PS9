import os
import json
import requests
from flask import Flask, request, render_template, send_from_directory, jsonify
from PIL import Image, ImageFile
import numpy as np
import cv2
import pygame
from googletrans import Translator
from gtts import gTTS
from langdetect import detect
import easyocr
import pytesseract
import time
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor
from preprocessing import preprocess_image

# TrOCR imports
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Groq import for summarization
from groq import Groq

# Enable loading of truncated images for better stability
ImageFile.LOAD_TRUNCATED_IMAGES = True

# TrOCR Class - integrated directly from trocr.py
class TrOCRWithCoordinates:
    def __init__(self, model_name="microsoft/trocr-large-handwritten"):
        """Initialize the TrOCR model and text detector"""
        print(f"Loading TrOCR model: {model_name}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load TrOCR processor and model
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        
        # Initialize EasyOCR for text detection (coordinates)
        print("Loading text detector...")
        self.detector = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        print("Models loaded successfully!")

    def detect_text_regions(self, image_path):
        """Detect text regions and get bounding boxes using EasyOCR"""
        try:
            results = self.detector.readtext(image_path, paragraph=False, width_ths=0.7)
            text_regions = []
            for i, (bbox, text, confidence) in enumerate(results):
                bbox_array = np.array(bbox).astype(int)
                text_regions.append({
                    'id': i,
                    'bbox': bbox_array.tolist(),
                    'confidence': float(confidence),
                    'detected_text': text
                })
            return text_regions
        except Exception as e:
            print(f"Error detecting text regions: {e}")
            return []

    def crop_text_region(self, image, bbox, padding=5):
        """Crop text region from image with padding"""
        bbox_array = np.array(bbox)
        x_min = int(min(bbox_array[:, 0]))
        y_min = int(min(bbox_array[:, 1]))
        x_max = int(max(bbox_array[:, 0]))
        y_max = int(max(bbox_array[:, 1]))
        
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(image.width, x_max + padding)
        y_max = min(image.height, y_max + padding)
        
        return image.crop((x_min, y_min, x_max, y_max))

    def recognize_text_region(self, image_region, max_length=256):
        """Recognize text from a cropped image region using TrOCR"""
        try:
            pixel_values = self.processor(image_region, return_tensors="pt").pixel_values.to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True
                )
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return generated_text.strip()
        except Exception as e:
            print(f"Error recognizing text region: {e}")
            return ""

    def split_into_words(self, text, bbox):
        """Split recognized text into individual words with estimated coordinates"""
        words = text.split()
        if not words:
            return []

        bbox_array = np.array(bbox)
        x_min = min(bbox_array[:, 0])
        y_min = min(bbox_array[:, 1])
        x_max = max(bbox_array[:, 0])
        y_max = max(bbox_array[:, 1])
        region_width = x_max - x_min

        word_list = []
        total_chars = sum(len(word) for word in words) + len(words) - 1
        current_x = x_min

        for i, word in enumerate(words):
            word_width = (len(word) / total_chars) * region_width
            word_bbox = [
                [int(current_x), int(y_min)],
                [int(current_x + word_width), int(y_min)],
                [int(current_x + word_width), int(y_max)],
                [int(current_x), int(y_max)]
            ]

            word_dict = {
                "box": word_bbox,
                "detected_text": word,
                "confidence": 1.0,
                "language": "en",
                "is_handwritten": True,
                "spell_checked": False,
                "original_text": word
            }
            word_list.append(word_dict)
            
            space_width = (1 / total_chars) * region_width if i < len(words) - 1 else 0
            current_x += word_width + space_width

        return word_list

    def process_image(self, image_path, max_length=256):
        """Process entire image and return JSON in EasyOCR format"""
        start_time = time.time()
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            print("Detecting text regions...")
            text_regions = self.detect_text_regions(image_path)
            
            if not text_regions:
                print("No text regions detected")
                return self.create_empty_result(image_path, start_time)

            print(f"Found {len(text_regions)} text regions")
            all_results = []
            
            for i, region in enumerate(text_regions):
                print(f"Processing region {i+1}/{len(text_regions)}...")
                cropped_image = self.crop_text_region(image, region['bbox'])
                trocr_text = self.recognize_text_region(cropped_image, max_length)
                
                if trocr_text:
                    words = self.split_into_words(trocr_text, region['bbox'])
                    all_results.extend(words)
                else:
                    word_dict = {
                        "box": region['bbox'],
                        "detected_text": region['detected_text'],
                        "confidence": region['confidence'],
                        "language": "en",
                        "is_handwritten": True,
                        "spell_checked": False,
                        "original_text": region['detected_text']
                    }
                    all_results.append(word_dict)

            processing_time = time.time() - start_time
            result = {
                "language": "en",
                "is_handwritten": True,
                "spell_check_enabled": False,
                "processing_time": round(processing_time, 1),
                "total_detections": len(all_results),
                "fast_processing": True,
                "results": all_results
            }
            return result

        except Exception as e:
            print(f"Error processing image: {e}")
            return self.create_empty_result(image_path, start_time, str(e))

    def create_empty_result(self, image_path, start_time, error=None):
        """Create empty result structure"""
        processing_time = time.time() - start_time
        result = {
            "language": "en",
            "is_handwritten": True,
            "spell_check_enabled": False,
            "processing_time": round(processing_time, 1),
            "total_detections": 0,
            "fast_processing": True,
            "results": []
        }
        if error:
            result["error"] = error
        return result

# Groq API Configuration for Spell Check and Summarization
GROQ_API_KEY = "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"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# Summarization Functions (integrated from Summarizer.py)
def summarize_text(text, model="llama3-70b-8192"):
    """Summarize the given text using Groq API"""
    if not text or len(text.strip()) < 50:  # Skip very short texts
        return "Text too short to summarize."
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        prompt = f"""
You are a professional summarizer. Read the full input text carefully and thoroughly. Do not start summarizing until you have completely processed the entire content. Your goal is to understand the core message, main arguments, and key insights. Once you have understood everything, generate a well-structured summary in the same language as the input text. The summary should be concise, informative, and accurate, capturing all important points without omitting critical details. Do not translate. Do not include your own opinion. Do not add any preamble, labels, or comments—output the summary only. Your output must be *only the summary* — do not include any title, label, comment, or metadata

Input Text:
{text}
"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in text summarization: {e}")
        return "Error occurred during summarization."

def save_summary_to_file(summary_text, file_path):
    """Save summary text to a file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        return True
    except Exception as e:
        print(f"Error saving summary to file: {e}")
        return False

app = Flask(__name__)
CORS(app)

# --- Paths ---
BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
TMP_DIR = os.path.join(BASE_DIR, 'tmp')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
for d in (UPLOAD_DIR, TMP_DIR, STATIC_DIR):
    os.makedirs(d, exist_ok=True)

# CRAFT endpoint (only for printed text)
CRAFT_URL = 'http://localhost:6000/detect'

# Initialize translators and readers
translator = Translator()
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Language-specific EasyOCR readers for printed text
print("Initializing EasyOCR readers...")
reader_en = easyocr.Reader(['en'], gpu=True)
reader_hi = easyocr.Reader(['hi', 'en'], gpu=True)
reader_ru = easyocr.Reader(['ru', 'en'], gpu=True)
reader_fr = easyocr.Reader(['fr', 'en'], gpu=True)
reader_es = easyocr.Reader(['es', 'en'], gpu=True)
reader_ko = easyocr.Reader(['ko', 'en'], gpu=True)
reader_de = easyocr.Reader(['de', 'en'], gpu=True)
reader_it = easyocr.Reader(['it', 'en'], gpu=True)
reader_tr = easyocr.Reader(['tr', 'en'], gpu=True)
reader_multi = easyocr.Reader(['es', 'fr', 'de', 'it', 'tr', 'en'], gpu=True)

# TrOCR model configuration
TROCR_MODEL_CONFIG = {
    "base": "microsoft/trocr-base-handwritten",
    "large": "microsoft/trocr-large-handwritten"
}

# Initialize TrOCR for handwritten text recognition
print("Initializing TrOCR for handwritten text...")
try:
    selected_model = "base" #'large'
    model_name = TROCR_MODEL_CONFIG[selected_model]
    print(f"Loading TrOCR model: {model_name}")
    
    trocr_processor = TrOCRWithCoordinates(model_name)
    print(f"TrOCR {selected_model} model initialized successfully")
except Exception as e:
    print(f"Warning: TrOCR initialization failed: {e}")
    trocr_processor = None

# Language codes and mappings
lang_codes = {
    'ko': 'ko', 'hi': 'hi', 'ru': 'ru', 'es': 'es', 'fr': 'fr',
    'de': 'de', 'it': 'it', 'tr': 'tr', 'en': 'en'
}

tesseract_lang_codes = {
    'en': 'eng', 'hi': 'hin', 'ru': 'rus', 'fr': 'fra', 'es': 'spa',
    'ko': 'kor', 'de': 'deu', 'it': 'ita', 'tr': 'tur'
}

lang_names = {
    'ko': 'Korean', 'hi': 'Hindi', 'ru': 'Russian', 'es': 'Spanish', 'fr': 'French',
    'de': 'German', 'it': 'Italian', 'tr': 'Turkish', 'en': 'English'
}

# **UPDATED: Whole Text Spell Check Functions**
def correct_whole_text_with_groq(whole_text, lang_code="en"):
    """Corrects spelling and grammar on entire text using Groq LLaMA3 model."""
    if not whole_text or len(whole_text.strip()) < 3:
        return whole_text

    lang_name = lang_names.get(lang_code, "English")
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    system_prompt = (
        f"You are a professional spell checker for {lang_name}. "
        f"Correct only spelling errors in the complete text. "
        f"Maintain the original word order and structure. "
        f"Return only the corrected text without any explanations, comments, or extra words. "
        f"Do not add words like 'error', 'corrected', 'fixed' or any metadata."
    )

    user_prompt = (
        f"Correct only the spelling errors in this complete {lang_name} text. "
        f"Keep all original words and their positions. Only fix spelling mistakes. "
        
        f"Do not add explanations or metadata words. "
        f"❗ Use the context of the previous and following words to determine the correct spelling, choosing the closest valid word in {lang_name} without drastically changing or adding new words.\n"
        f"Return only the corrected complete text:\n\n{whole_text}"
    )

    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1000  # Increased for whole text
    }

    try:
        response = requests.post(GROQ_ENDPOINT, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            corrected_text = response.json()["choices"][0]["message"]["content"].strip()
            # Clean up any extra metadata that might be added
            corrected_text = corrected_text.replace("Corrected:", "").replace("Fixed:", "").strip()
            return corrected_text
        else:
            print(f"Groq API error {response.status_code}: {response.text}")
            return whole_text
    except Exception as e:
        print(f"Whole text spell check error: {e}")
        return whole_text

def apply_whole_text_spell_check(results, enable_spell_check=True):
    """Apply spell check on the entire combined text, then distribute back to individual results"""
    if not enable_spell_check or not results:
        for result in results:
            result['spell_checked'] = False
            result['original_text'] = result.get('detected_text', '')
        return results

    print(f"Applying whole text spell check to {len(results)} text regions...")
    
    # Combine all detected text into one string
    all_text = ' '.join([result.get('detected_text', '') for result in results if result.get('detected_text', '').strip()])
    
    if not all_text.strip():
        print("No text found to spell check")
        for result in results:
            result['spell_checked'] = False
            result['original_text'] = result.get('detected_text', '')
        return results
    
    # Get the dominant language
    language = results[0].get('language', 'en') if results else 'en'
    
    print(f"Combined text ({len(all_text)} chars): {all_text[:100]}...")
    
    # Apply spell check to the whole text
    corrected_whole_text = correct_whole_text_with_groq(all_text, language)
    
    print(f"Corrected whole text: {corrected_whole_text[:100]}...")
    
    # Split the corrected text back into words
    original_words = all_text.split()
    corrected_words = corrected_whole_text.split()
    
    print(f"Original words: {len(original_words)}, Corrected words: {len(corrected_words)}")
    
    # Distribute corrected words back to results
    corrected_results = []
    word_index = 0
    
    for result in results:
        original_text = result.get('detected_text', '').strip()
        if original_text:
            # Get the corrected word(s) for this result
            if word_index < len(corrected_words):
                corrected_result = result.copy()
                corrected_result['original_text'] = original_text
                corrected_result['detected_text'] = corrected_words[word_index]
                corrected_result['spell_checked'] = True
                corrected_results.append(corrected_result)
                word_index += 1
            else:
                # If we run out of corrected words, keep original
                result['spell_checked'] = False
                result['original_text'] = original_text
                corrected_results.append(result)
        else:
            # Empty text
            result['spell_checked'] = False
            result['original_text'] = ''
            corrected_results.append(result)
    
    print("Whole text spell check completed")
    return corrected_results

# Utility functions (unchanged)
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'en'

def get_reader_for_language(language):
    """Get the appropriate EasyOCR reader for the specified language"""
    readers = {
        'en': reader_en, 'hi': reader_hi, 'ru': reader_ru, 'fr': reader_fr,
        'es': reader_es, 'ko': reader_ko, 'de': reader_de, 'it': reader_it,
        'tr': reader_tr, 'AutoDetect': reader_multi
    }
    return readers.get(language, reader_multi)

def enhance_image_for_ocr(image_np):
    """Enhanced image preprocessing optimized for speed and accuracy"""
    scale_factor = 2
    resized = cv2.resize(image_np, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    enhanced = cv2.equalizeHist(sharpened)
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return resized, thresh, enhanced

def run_ocr_language_specific_fast(image_np, box_id, language):
    """Fast OCR for specific language using optimized multi-reader approach"""
    resized, thresh, enhanced = enhance_image_for_ocr(image_np)
    images_to_process = [resized, thresh, enhanced]
    results = []

    primary_reader = get_reader_for_language(language)
    for img in images_to_process:
        try:
            ocr_results = primary_reader.readtext(img)
            results.extend(ocr_results)
        except Exception as e:
            print(f"Primary EasyOCR error for {language}: {e}")
            continue

    if language == 'AutoDetect' or not results:
        additional_readers = [reader_ko, reader_hi, reader_ru, reader_multi]
        for reader in additional_readers:
            if reader == primary_reader:
                continue
            for img in images_to_process:
                try:
                    ocr_results = reader.readtext(img)
                    results.extend(ocr_results)
                except Exception as e:
                    continue

    if not results:
        try:
            if language == 'AutoDetect':
                tesseract_text = pytesseract.image_to_string(
                    resized, lang='hin+eng+kor+rus+spa+fra+deu+ita+tur', config='--psm 6'
                )
            else:
                tesseract_lang = tesseract_lang_codes.get(language, 'eng')
                tesseract_text = pytesseract.image_to_string(resized, lang=tesseract_lang, config='--psm 6')
            
            if tesseract_text.strip():
                results.append((None, tesseract_text.strip(), 0.5))
        except Exception as e:
            print(f"Tesseract error for {language}: {e}")

    if results:
        best = max(results, key=lambda x: x[2] if len(x) == 3 else -1)
        return best
    return None

# Text reordering functions (unchanged)
def get_text_position(box: List[List[int]]) -> tuple:
    """Extract the top-left position from a bounding box."""
    top_y = min(point[1] for point in box)
    left_x = min(point[0] for point in box)
    return (top_y, left_x)

def group_by_lines(text_items: List[Dict], line_threshold: int = 35) -> List[List[Dict]]:
    """Group text items that are on the same line based on their y-coordinates."""
    if not text_items:
        return []

    sorted_items = sorted(text_items, key=lambda item: get_text_position(item['box'])[0])
    lines = []
    current_line = [sorted_items[0]]
    current_y = get_text_position(sorted_items[0]['box'])[0]

    for item in sorted_items[1:]:
        item_y = get_text_position(item['box'])[0]
        if abs(item_y - current_y) <= line_threshold:
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]
            current_y = item_y

    if current_line:
        lines.append(current_line)
    return lines

def sort_line_items(line_items: List[Dict]) -> List[Dict]:
    """Sort items within a line from left to right based on x-coordinates."""
    return sorted(line_items, key=lambda item: get_text_position(item['box'])[1])

def reorder_ocr_results(results: List[Dict], line_threshold: int = 35) -> List[Dict]:
    """Reorder OCR results in natural reading order (top to bottom, left to right)."""
    if not results:
        return []

    print(f"Reordering {len(results)} text items...")
    lines = group_by_lines(results, line_threshold)
    print(f"Found {len(lines)} lines of text")

    reordered_results = []
    for i, line in enumerate(lines):
        sorted_line = sort_line_items(line)
        reordered_results.extend(sorted_line)
        line_text = ' '.join(item['detected_text'] for item in sorted_line if item['detected_text'])
        print(f"Line {i+1}: {line_text}")

    full_text = ' '.join(item['detected_text'] for item in reordered_results if item['detected_text'])
    print(f"Reconstructed text: {full_text}")
    return reordered_results

def detect_dominant_language(all_text_results):
    """Detect the dominant language from all OCR results combined."""
    if not all_text_results:
        return 'en'

    combined_text = ' '.join([result.get('detected_text', '') for result in all_text_results if result.get('detected_text', '').strip()])
    
    if not combined_text.strip() or len(combined_text.strip()) < 20:
        return 'en'

    try:
        detected_lang = detect_language(combined_text)
        print(f"Dominant language detected from combined text: {detected_lang}")
        return detected_lang
    except Exception as e:
        print(f"Language detection failed on combined text: {e}")
        return 'en'

def process_region_fast(args):
    """Process a single OCR region with optimized speed for printed text only"""
    box, idx, img_np, language, is_handwritten = args

    try:
        # Cropping logic
        if isinstance(box[0], list) and len(box) > 2:
            xs = [pt[0] for pt in box]; ys = [pt[1] for pt in box]
            x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        elif isinstance(box[0], list):
            (x0, y0), (x1, y1) = box
            x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
        else:
            x0, y0, x1, y1 = map(int, box)

        if x1 <= x0 or y1 <= y0:
            return None

        crop_np = img_np[y0:y1, x0:x1]

        # Only use this for printed text (not handwritten)
        if not is_handwritten:
            ocr = run_ocr_language_specific_fast(crop_np, f"region_{idx}", language)
            if ocr and len(ocr) >= 3:
                _, text, conf = ocr
            else:
                text, conf = "", 0.0

            print(f"Region {idx} - Detected Text: {text} (Confidence: {conf:.2f})")

            return {
                'box': box,
                'detected_text': text,
                'confidence': conf,
                'language': language,
                'is_handwritten': is_handwritten
            }
        else:
            return None

    except Exception as e:
        print(f"Error processing box {idx}: {e}")
        return None

# **NEW: Summarization Route**
@app.route('/summarize', methods=['POST'])
def summarize_extracted_text():
    """Generate summary from extracted OCR text and save as summary.txt"""
    try:
        print("=== Text summarization requested ===")
        
        # Get text from form data
        text_to_summarize = request.form.get('text', '').strip()
        if not text_to_summarize:
            print("Error: No text provided for summarization")
            return jsonify({'error': 'No text provided for summarization'}), 400

        print(f"Text to summarize ({len(text_to_summarize)} chars): {text_to_summarize[:100]}...")
        
        # Check if text is long enough to summarize
        if len(text_to_summarize) < 50:
            print("Text too short to summarize")
            return jsonify({'error': 'Text is too short to generate a meaningful summary'}), 400
        
        # Generate summary using Groq
        print("Generating summary with Groq API...")
        summary = summarize_text(text_to_summarize)
        
        if not summary or summary.startswith("Error"):
            print("Summarization failed")
            return jsonify({'error': 'Failed to generate summary'}), 500
        
        # Save summary to summary.txt in static directory
        summary_path = os.path.join(STATIC_DIR, 'summary.txt')
        if save_summary_to_file(summary, summary_path):
            print(f"Summary saved to: {summary_path}")
            print(f"Summary preview: {summary[:100]}...")
            
            # Also save summary metadata as JSON
            summary_data = {
                'original_text': text_to_summarize,
                'summary': summary,
                'original_length': len(text_to_summarize),
                'summary_length': len(summary),
                'compression_ratio': round(len(summary) / len(text_to_summarize), 2),
                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'model_used': 'llama3-70b-8192'
            }
            
            summary_json_path = os.path.join(STATIC_DIR, 'summary.json')
            with open(summary_json_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=4)
            
            return jsonify({
                'success': True,
                'message': 'Summary generated successfully',
                'summary': summary,
                'original_length': len(text_to_summarize),
                'summary_length': len(summary),
                'compression_ratio': round(len(summary) / len(text_to_summarize), 2),
                'summary_file': 'summary.txt',
                'summary_url': f'/static/summary.txt?t={int(time.time())}'  # Cache-busting
            }), 200
        else:
            print("Failed to save summary to file")
            return jsonify({'error': 'Failed to save summary to file'}), 500
            
    except Exception as e:
        print(f"❌ Summarization error: {e}")
        return jsonify({'error': f'Summarization failed: {str(e)}'}), 500

@app.route('/get_summary')
def get_summary():
    """Get the saved summary data"""
    try:
        summary_json_path = os.path.join(STATIC_DIR, 'summary.json')
        if os.path.exists(summary_json_path):
            with open(summary_json_path, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
            return jsonify(summary_data)
        else:
            return jsonify({'error': 'No summary available'}), 404
    except Exception as e:
        print(f"Error reading summary: {e}")
        return jsonify({'error': 'Error reading summary'}), 500

# **UNCHANGED: Keep handwritten result route exactly as it is (NO spell check)**
@app.route('/handwritten_result', methods=['POST'])
def handwritten_result():
    """Dedicated route for handwritten text processing using TrOCR with preprocessing"""
    start_time = time.time()
    
    # Get form data
    language = request.form.get('language', 'AutoDetect')
    enable_spell_check = request.form.get('enable_spell_check', 'false').lower() == 'true'
    max_length = int(request.form.get('max_length', '256'))
    
    print(f"Handwritten processing - Language: {language}, Spell check: {enable_spell_check}")
    print(f"Max text length: {max_length}")
    
    # Save user image
    file = request.files.get('file')
    if not file:
        print("Error: No image provided")
        return jsonify({'error': 'No image provided'}), 400
    
    fname = file.filename
    img_path = os.path.join(UPLOAD_DIR, fname)
    file.save(img_path)
    print(f"Processing image: {img_path}")
    
    # Apply preprocessing and save as preprocess.png
    try:
        print("Applying preprocessing to improve handwritten text detection...")
        preprocessed_image = preprocess_image(img_path)
        
        # Save preprocessed image as preprocess.png in static directory
        preprocess_path = os.path.join(STATIC_DIR, 'preprocess.png')
        cv2.imwrite(preprocess_path, preprocessed_image)
        print(f"Preprocessed image saved as: {preprocess_path}")
        
        # Use original image for TrOCR processing
        processing_image_path = img_path
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        print("Continuing with original image...")
        processing_image_path = img_path
    
    # Check if TrOCR is available
    if trocr_processor is None:
        print("Error: TrOCR not available")
        return jsonify({'error': 'TrOCR not available. Please check initialization.'}), 500
    
    # Continue with exact same TrOCR processing as original
    try:
        print("Processing handwritten text with TrOCR...")
        print("Detecting text regions...")
        
        result = trocr_processor.process_image(processing_image_path, max_length)
        
        if not result:
            print("Error: Processing failed")
            return jsonify({'error': 'TrOCR processing failed'}), 500
            
        # Log results
        print(f"Words detected: {result['total_detections']}")
        print(f"Processing time: {result['processing_time']}s")
        
        if result['results']:
            print("Sample detections:")
            for i, detection in enumerate(result['results'][:3]):
                confidence = detection.get('confidence', 1.0)
                text = detection.get('detected_text', '')
                print(f" {i+1}. '{text}' (confidence: {confidence:.3f})")
        
        output = result['results']
        
        # Initialize pygame for audio functionality
        pygame.mixer.init()
        
        # **NO SPELL CHECK FOR HANDWRITTEN - Mark all as not spell-checked**
        for result_item in output:
            result_item['spell_checked'] = False
            result_item['original_text'] = result_item.get('detected_text', '')
        
        # Calculate total processing time
        total_processing_time = round(time.time() - start_time, 1)
        print(f"Total handwritten processing completed in {total_processing_time} seconds")
        
        # Save original JSON (data.json)
        json_path = os.path.join(STATIC_DIR, 'data.json')
        data_json = {
            'batch_info': {
                'total_images': 1,
                'processed_successfully': 1,
                'processed_at': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'language': language,
            'is_handwritten': True,
            'spell_check_enabled': False,  # Never enabled for handwritten
            'processing_time': total_processing_time,
            'trocr_processing_time': result['processing_time'],
            'total_detections': len(output),
            'fast_processing': False,
            'processing_method': 'EasyOCR (detection) + TrOCR (recognition)',
            'model_used': selected_model,
            'max_length': max_length,
            'results': output
        }
        
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(data_json, jf, ensure_ascii=False, indent=4)
        print("Handwritten data.json saved successfully")
        
        # Reorder results and save as final_data.json
        try:
            print("Starting text reordering for handwritten results...")
            reordered_output = reorder_ocr_results(output, line_threshold=25)
            
            final_data_json = {
                'batch_info': {
                    'total_images': 1,
                    'processed_successfully': 1,
                    'processed_at': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'language': language,
                'is_handwritten': True,
                'spell_check_enabled': False,  # Never enabled for handwritten
                'processing_time': total_processing_time,
                'trocr_processing_time': result['processing_time'],
                'total_detections': len(reordered_output),
                'fast_processing': False,
                'processing_method': 'EasyOCR (detection) + TrOCR (recognition)',
                'model_used': selected_model,
                'max_length': max_length,
                'reordered': True,
                'results': reordered_output
            }
            
            # **NEW: Generate automatic summary for handwritten text if substantial**
            try:
                combined_text = ' '.join([item['detected_text'] for item in reordered_output if item.get('detected_text', '').strip()])
                
                if len(combined_text) >= 100:  # Only summarize if substantial text
                    print("Generating automatic summary for handwritten text...")
                    summary = summarize_text(combined_text)
                    
                    if summary and not summary.startswith("Error"):
                        # Save summary
                        summary_path = os.path.join(STATIC_DIR, 'summary.txt')
                        save_summary_to_file(summary, summary_path)
                        
                        # Add summary info to final JSON
                        final_data_json['summary_generated'] = True
                        final_data_json['summary'] = summary
                        final_data_json['combined_text_length'] = len(combined_text)
                        final_data_json['summary_length'] = len(summary)
                        print(f"Automatic summary generated and saved for handwritten text")
                    else:
                        final_data_json['summary_generated'] = False
                        print("Automatic summary generation failed for handwritten text")
                else:
                    final_data_json['summary_generated'] = False
                    print("Handwritten text too short for automatic summarization")
                    
            except Exception as e:
                print(f"Error in automatic summarization for handwritten text: {e}")
                final_data_json['summary_generated'] = False
            
            final_json_path = os.path.join(STATIC_DIR, 'final_data.json')
            with open(final_json_path, 'w', encoding='utf-8') as jf:
                json.dump(final_data_json, jf, ensure_ascii=False, indent=4)
            print("Reordered handwritten final_data.json saved successfully")
            
        except Exception as e:
            print(f"Error during handwritten text reordering: {e}")
            with open(os.path.join(STATIC_DIR, 'final_data.json'), 'w', encoding='utf-8') as jf:
                json.dump(data_json, jf, ensure_ascii=False, indent=4)
        
        return '', 204
        
    except KeyboardInterrupt:
        print("Operation cancelled by user")
        return jsonify({'error': 'Processing cancelled'}), 500
    except Exception as e:
        print(f"Error in handwritten processing: {e}")
        return jsonify({'error': f'Handwritten processing failed: {str(e)}'}), 500

# **UPDATED: Modified main result route with whole text spell check and automatic summarization**
@app.route('/result', methods=['POST'])
def result():
    start_time = time.time()
    
    # Get form data
    language = request.form.get('language', 'AutoDetect')
    is_handwritten = request.form.get('is_handwritten', 'false').lower() == 'true'
    enable_spell_check = request.form.get('enable_spell_check', 'false').lower() == 'true'
    
    print(f"Language selected: {language}")
    print(f"Handwritten mode: {is_handwritten}")
    print(f"Spell check enabled: {enable_spell_check}")
    
    # Redirect handwritten requests to dedicated route
    if is_handwritten:
        print("Redirecting to handwritten processing route...")
        return handwritten_result()
    
    # Continue with fast printed text processing
    print("Processing printed text with fast parallel processing...")
    
    # Save user image
    file = request.files.get('file')
    if not file:
        return "No image provided", 400

    fname = file.filename
    img_path = os.path.join(UPLOAD_DIR, fname)
    file.save(img_path)

    # Send to CRAFT for boxes (only for printed text)
    with open(img_path, 'rb') as f:
        craft_resp = requests.post(CRAFT_URL, files={'image': f})

    if craft_resp.status_code != 200:
        return "CRAFT service failed", 500

    boxes = craft_resp.json()

    # Load image array and init pygame
    try:
        pil = Image.open(img_path).convert('RGB')
        img_np = np.array(pil)
    except Exception as e:
        print(f"Error loading image: {e}")
        return "Failed to load image", 500

    pygame.mixer.init()

    # Process printed text with fast parallel processing
    print(f"Processing {len(boxes)} text regions with fast parallel processing...")
    
    # Prepare arguments for parallel processing (False = not handwritten)
    args_list = [(box, idx, img_np, language, False) for idx, box in enumerate(boxes)]
    
    # Use ThreadPoolExecutor for parallel processing
    output = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_region_fast, args_list))
        
        # Filter out None results and collect valid outputs
        for result in results:
            if result is not None:
                output.append(result)

    # Detect dominant language AFTER processing all regions
    if language == 'AutoDetect' and output:
        print("AutoDetect mode: Analyzing dominant language from all text...")
        dominant_language = detect_dominant_language(output)
        print(f"Dominant language detected: {dominant_language}")
        
        # Update all results with the dominant language
        for result in output:
            result['language'] = dominant_language

    # **UPDATED: Apply whole text spell check if enabled**
    if enable_spell_check:
        print("\n--- Starting whole text spell check post-processing ---")
        spell_check_start = time.time()
        output = apply_whole_text_spell_check(output, enable_spell_check)
        spell_check_time = round(time.time() - spell_check_start, 1)
        print(f"Whole text spell check completed in {spell_check_time} seconds")
    else:
        # Mark all results as not spell-checked
        for result in output:
            result['spell_checked'] = False
            result['original_text'] = result.get('detected_text', '')

    # Calculate processing time
    processing_time = round(time.time() - start_time, 1)
    print(f"Total printed text processing completed in {processing_time} seconds")

    # Save original JSON (data.json)
    json_path = os.path.join(STATIC_DIR, 'data.json')
    data_json = {
        'language': language,
        'is_handwritten': False,
        'spell_check_enabled': enable_spell_check,
        'processing_time': processing_time,
        'total_detections': len(output),
        'fast_processing': True,
        'processing_method': 'CRAFT + EasyOCR',
        'results': output
    }

    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(data_json, jf, ensure_ascii=False, indent=4)
    print("Printed text data.json saved successfully")

    # Reorder the results and save as final_data.json
    try:
        print("\n--- Starting text reordering ---")
        reordered_output = reorder_ocr_results(output, line_threshold=25)
        
        # Create final data structure
        final_data_json = {
            'language': language,
            'is_handwritten': False,
            'spell_check_enabled': enable_spell_check,
            'processing_time': processing_time,
            'total_detections': len(reordered_output),
            'fast_processing': True,
            'processing_method': 'CRAFT + EasyOCR',
            'reordered': True,
            'results': reordered_output
        }
        
        # **NEW: Generate automatic summary for printed text if substantial**
        try:
            combined_text = ' '.join([item['detected_text'] for item in reordered_output if item.get('detected_text', '').strip()])
            
            if len(combined_text) >= 100:  # Only summarize if substantial text
                print("Generating automatic summary for printed text...")
                summary = summarize_text(combined_text)
                
                if summary and not summary.startswith("Error"):
                    # Save summary
                    summary_path = os.path.join(STATIC_DIR, 'summary.txt')
                    save_summary_to_file(summary, summary_path)
                    
                    # Add summary info to final JSON
                    final_data_json['summary_generated'] = True
                    final_data_json['summary'] = summary
                    final_data_json['combined_text_length'] = len(combined_text)
                    final_data_json['summary_length'] = len(summary)
                    print(f"Automatic summary generated and saved for printed text")
                else:
                    final_data_json['summary_generated'] = False
                    print("Automatic summary generation failed for printed text")
            else:
                final_data_json['summary_generated'] = False
                print("Printed text too short for automatic summarization")
                
        except Exception as e:
            print(f"Error in automatic summarization for printed text: {e}")
            final_data_json['summary_generated'] = False
        
        # Save reordered results
        final_json_path = os.path.join(STATIC_DIR, 'final_data.json')
        with open(final_json_path, 'w', encoding='utf-8') as jf:
            json.dump(final_data_json, jf, ensure_ascii=False, indent=4)
        print("Reordered printed text final_data.json saved successfully")
        
    except Exception as e:
        print(f"Error during text reordering: {e}")
        # If reordering fails, copy original data to final_data.json
        with open(os.path.join(STATIC_DIR, 'final_data.json'), 'w', encoding='utf-8') as jf:
            json.dump(data_json, jf, ensure_ascii=False, indent=4)

    return '', 204

# Keep all your existing routes unchanged
@app.route('/data')
def get_data():
    """Endpoint to get OCR data for the frontend"""
    try:
        json_path = os.path.join(STATIC_DIR, 'data.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return jsonify(data.get('results', []))
        return jsonify([])
    except Exception as e:
        print(f"Error reading data.json: {e}")
        return jsonify([])

@app.route('/final_data')
def get_final_data():
    """Endpoint to get reordered OCR data for the frontend"""
    try:
        json_path = os.path.join(STATIC_DIR, 'final_data.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return jsonify(data.get('results', []))
        return jsonify([])
    except Exception as e:
        print(f"Error reading final_data.json: {e}")
        return jsonify([])

@app.route('/audio', methods=['POST'])
def audio():
    """Generate audio from text sent in the request."""
    try:
        print("=== Audio generation requested by user ===")
        
        # Get form data
        text_to_convert = request.form.get('text', '').strip()
        is_all_text = request.form.get('isAllText', 'false').lower() == 'true'
        is_translated_audio = request.form.get('isTranslatedAudioRequest', 'false').lower() == 'true'
        
        print(f"Form data received:")
        print(f" - text: {text_to_convert[:50]}..." if text_to_convert else " - text: (empty)")
        print(f" - isAllText: {is_all_text}")
        print(f" - isTranslatedAudioRequest: {is_translated_audio}")

        if not text_to_convert:
            print("Error: No text provided for audio generation")
            return jsonify({'error': 'No text provided for audio generation'}), 400

        # Detect language from the text
        try:
            detected_lang = detect_language(text_to_convert)
            print(f"Detected language for TTS: {detected_lang}")
            
            # For translated text, force English since it's translated to English
            if is_translated_audio:
                tts_lang = 'en'
                print("Using English for translated text audio")
            else:
                tts_lang = lang_codes.get(detected_lang, 'en')
                print(f"Using TTS language code: {tts_lang}")
                
        except Exception as lang_error:
            print(f"Language detection failed: {lang_error}, defaulting to English")
            tts_lang = 'en'

        # Determine audio file name based on the type of request
        if is_translated_audio:
            audio_filename = 'translated_data.mp3'
        elif is_all_text:
            audio_filename = 'alldata.mp3'
        else:
            audio_filename = 'data.mp3'

        audio_path = os.path.join(STATIC_DIR, audio_filename)

        try:
            # Remove existing audio file if it exists
            if os.path.exists(audio_path):
                os.remove(audio_path)
                print(f"Removed existing {audio_filename}")

            # Generate TTS audio
            print(f"Generating TTS audio using gTTS with language: {tts_lang}")
            tts = gTTS(text=text_to_convert, lang=tts_lang, slow=False)
            tts.save(audio_path)

            # Wait for file to be written completely
            time.sleep(0.5)

            # Verify the audio file was created successfully
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                file_size = os.path.getsize(audio_path)
                print(f"✅ Audio file successfully created: {audio_filename} ({file_size} bytes)")

                return jsonify({
                    'success': True,
                    'message': 'Audio generated successfully',
                    'audio_file': audio_filename,
                    'audio_url': f'/static/{audio_filename}?t={int(time.time())}',
                    'text_length': len(text_to_convert),
                    'language': detected_lang,
                    'tts_language': tts_lang,
                    'file_size_bytes': file_size,
                    'audio_type': 'translated' if is_translated_audio else ('all_text' if is_all_text else 'selected')
                }), 200
            else:
                print(f"❌ Error: Audio file {audio_filename} was not created or is empty")
                return jsonify({'error': f'Failed to create audio file {audio_filename}'}), 500

        except Exception as tts_error:
            print(f"❌ TTS generation error: {tts_error}")
            return jsonify({'error': f'TTS generation failed: {str(tts_error)}'}), 500

    except Exception as e:
        print(f"❌ General audio generation error: {e}")
        return jsonify({'error': f'Audio generation failed: {str(e)}'}), 500

@app.route('/translate', methods=['POST'])
def translate_text():
    """Translate text sent in the request to English."""
    try:
        print("=== Translation requested by user ===")
        
        # Get text from form data
        text_to_translate = request.form.get('text', '').strip()
        if not text_to_translate:
            print("Error: No text provided for translation")
            return jsonify({'error': 'No text provided for translation'}), 400

        print(f"Text to translate ({len(text_to_translate)} chars): {text_to_translate[:100]}...")

        # Detect source language
        try:
            source_lang = detect_language(text_to_translate)
            print(f"Detected source language: {source_lang}")
            
            # Skip translation if already in English
            if source_lang == 'en':
                translated_text = text_to_translate
                print("Text is already in English, no translation needed")
            else:
                # Translate to English
                print(f"Translating from {source_lang} to English...")
                translation = translator.translate(text_to_translate, src=source_lang, dest='en')
                translated_text = translation.text
                print(f"Translation completed: {translated_text[:100]}...")
                
        except Exception as lang_error:
            print(f"Language detection or translation failed: {lang_error}")
            # Try translation without specifying source language
            try:
                translation = translator.translate(text_to_translate, dest='en')
                translated_text = translation.text
                source_lang = 'auto'
                print(f"Auto-translation completed: {translated_text[:100]}...")
            except Exception as trans_error:
                print(f"Translation failed completely: {trans_error}")
                return jsonify({'error': f'Translation failed: {str(trans_error)}'}), 500

        # Save translation result
        translation_data = {
            'original_text': text_to_translate,
            'translated_text': translated_text,
            'source_language': source_lang,
            'target_language': 'en',
            'timestamp': time.time()
        }

        translate_path = os.path.join(STATIC_DIR, 'translate.json')
        with open(translate_path, 'w', encoding='utf-8') as f:
            json.dump(translation_data, f, ensure_ascii=False, indent=4)

        print("Translation result saved to translate.json")

        return jsonify({
            'success': True,
            'message': 'Translation completed successfully',
            'original_text': text_to_translate,
            'translated_text': translated_text,
            'source_language': source_lang,
            'target_language': 'en'
        }), 200

    except Exception as e:
        print(f"❌ Translation error: {e}")
        return jsonify({'error': f'Translation failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
