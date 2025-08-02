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
import logging
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor
from preprocessing import preprocess_image

# TrOCR imports
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Fine-tuned model imports
import torch.nn as nn
from torchvision import transforms

# Groq import for summarization
from groq import Groq

# Enable loading of truncated images for better stability
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# ========================
# FINE-TUNED MODEL CLASSES
# ========================

class SimpleCRNN(nn.Module):
    """Fine-tuned CRNN model for language-specific OCR"""
    def __init__(self, num_classes, img_h=32, hidden_size=256):
        super(SimpleCRNN, self).__init__()
        self.img_h = img_h
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, kernel_size=2, padding=0),
        )

        # RNN layers
        self.rnn = nn.LSTM(512, hidden_size, bidirectional=True, batch_first=True)
        
        # Output layer
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        conv_features = self.cnn(x)
        b, c, h, w = conv_features.size()
        
        if h == 1 and w == 1:
            conv_features = conv_features.view(b, c, 1)
        elif h == 1:
            conv_features = conv_features.squeeze(2)
            conv_features = conv_features.permute(0, 2, 1)
        elif w == 1:
            conv_features = conv_features.squeeze(3)
            conv_features = conv_features.permute(0, 2, 1)
        else:
            conv_features = conv_features.view(b, c, h * w).permute(0, 2, 1)
        
        if conv_features.dim() == 2:
            conv_features = conv_features.unsqueeze(1)
        
        rnn_output, _ = self.rnn(conv_features)
        output = self.classifier(rnn_output)
        
        if output.dim() == 2:
            output = output.unsqueeze(0)
        
        output = output.permute(1, 0, 2)
        output = nn.functional.log_softmax(output, dim=2)
        return output

class LanguageSpecificOCR:
    """Language-specific OCR processor using fine-tuned models"""
    def __init__(self, base_weights_dir="../Weights"):
        self.base_weights_dir = base_weights_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.charsets = {}
        self.preprocessors = {}

        # Language mapping
        self.language_mapping = {
            'en': 'English',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'es': 'Spanish',
            'ko': 'Korean',
            'hi': 'Hindi',
            'tr': 'Turkish',
            'ru': 'Russian'
        }

        print(f"Initializing Language-Specific OCR with weights directory: {base_weights_dir}")
        self._load_available_models()

    def _load_available_models(self):
        """Load all available language models"""
        for lang_code, lang_name in self.language_mapping.items():
            try:
                self._load_language_model(lang_code, lang_name)
            except Exception as e:
                print(f"Warning: Could not load {lang_name} model: {e}")
                continue

    def _load_language_model(self, lang_code, lang_name):
        """Load a specific language model"""
        # Try different model naming conventions
        possible_names = [
            f"best_{lang_code}_ocr_model.pth",
            f"best_{lang_name.lower()}_ocr_model.pth"
        ]
        
        weights_path = None
        for name in possible_names:
            test_path = os.path.join(self.base_weights_dir, lang_name, name)
            if os.path.exists(test_path):
                weights_path = test_path
                break
        
        if not weights_path:
            print(f"Model weights not found for {lang_name} in: {os.path.join(self.base_weights_dir, lang_name)}")
            return

        print(f"Loading {lang_name} model from: {weights_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(weights_path, map_location=self.device)
            
            # Extract character list and model info
            character_list = checkpoint['character_list']
            num_classes = checkpoint['model_state_dict']['classifier.bias'].shape[0]
            
            # Create charset (add blank token at index 0 for CTC)
            charset = [''] + character_list
            self.charsets[lang_code] = charset
            
            # Initialize model
            model = SimpleCRNN(num_classes=num_classes).to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            self.models[lang_code] = model
            
            # Create preprocessor
            preprocess = transforms.Compose([
                transforms.Resize((32, 100)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            self.preprocessors[lang_code] = preprocess
            
            print(f"✅ {lang_name} model loaded successfully (num_classes: {num_classes})")
            
        except Exception as e:
            print(f"❌ Failed to load {lang_name} model: {e}")
            raise

    def decode_output(self, output, charset):
        """Decode model output to text"""
        output = output.cpu().detach().numpy()
        pred = np.argmax(output, axis=2)
        text = ""
        for p in pred[0]:
            if p != 0 and (not text or text[-1] != charset[p]): # Remove blanks and duplicates
                if p < len(charset): # Safety check
                    text += charset[p]
        confidence = np.mean(np.max(output, axis=2)) # Simple confidence metric
        return text, confidence

    def recognize_text(self, image_np, language_code):
        """Recognize text using language-specific fine-tuned model"""
        if language_code not in self.models:
            return None, 0.0

        try:
            # Convert to PIL Image
            img_pil = Image.fromarray(image_np)
            img_tensor = self.preprocessors[language_code](img_pil).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self.models[language_code](img_tensor)
            
            text, confidence = self.decode_output(output, self.charsets[language_code])
            return text, confidence
            
        except Exception as e:
            print(f"Error in {language_code} fine-tuned model: {e}")
            return None, 0.0

    def get_available_languages(self):
        """Get list of available language models"""
        return list(self.models.keys())

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

# Summarization Functions (integrated from test.py)
def summarize_text(text, model="llama3-70b-8192"):
    """Summarize the given text using Groq API"""
    if not text or len(text.strip()) < 50:
        return "Text too short to summarize."

    try:
        client = Groq(api_key=GROQ_API_KEY)
        prompt = f"""
You are a professional summarizer. Read the full input text carefully and thoroughly. Do not start summarizing until you have completely processed the entire content. Your goal is to understand the core message, main arguments, and key insights. Once you have understood everything, generate a well-structured summary in the same language as the input text. The summary should be concise, informative, and accurate, capturing all important points without omitting critical details. Do not translate. Do not include your own opinion. Do not add any preamble, labels, or comments—output the summary only. Your output must be *only the summary* — do not include any title, label, comment,or metadata

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

# Initialize Flask app
app = Flask(__name__)

# Enhanced CORS configuration for tunneling services
CORS(app,
    origins=['*'],
    methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allow_headers=['Content-Type', 'Authorization', 'Accept', 'Origin', 'X-Requested-With', "ngrok-skip-browser-warning"],
    expose_headers=['Content-Range', 'X-Content-Range'],
    supports_credentials=True
)

# Add headers to all responses for better tunnel compatibility
@app.after_request
def after_request(response):
    #response.headers.add('Access-Control-Allow-Origin', '*')
    #response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept,Origin,X-Requested-With,ngrok-skip-browser-warning')
    #response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.headers.add('Cache-Control', 'no-cache, no-store, must-revalidate')
    response.headers.add('Pragma', 'no-cache')
    response.headers.add('Expires', '0')
    return response

# Request/Response logging for debugging tunnel issues
@app.before_request
def log_request_info():
    logger.info(f"Request: {request.method} {request.url}")
    logger.info(f"Headers: {dict(request.headers)}")
    if request.data:
        logger.info(f"Request data: {len(request.data)} bytes")

@app.after_request
def log_response_info(response):
    logger.info(f"Response: {response.status_code}")
    return response

# --- Paths ---
BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
TMP_DIR = os.path.join(BASE_DIR, 'tmp')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
WEIGHTS_DIR = os.path.join(BASE_DIR, '../Weights') # Default weights directory

for d in (UPLOAD_DIR, TMP_DIR, STATIC_DIR):
    os.makedirs(d, exist_ok=True)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Not Found",
        "message": "The requested endpoint does not exist",
        "available_endpoints": [
            "GET / - Root endpoint",
            "GET /api/test - API test endpoint",
            "GET /health - Health check",
            "GET /data - Get OCR data",
            "GET /final_data - Get reordered OCR data",
            "POST /result - OCR processing (printed text)",
            "POST /handwritten_result - Handwritten OCR processing",
            "POST /audio - Audio generation",
            "POST /translate - Text translation",
            "POST /summarize - Text summarization",
            "GET /get_summary - Get saved summary"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        "error": "Internal Server Error",
        "message": "Something went wrong on the server"
    }), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}")
    return jsonify({
        "error": "Server Error",
        "message": str(e)
    }), 500

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

# Initialize Language-Specific OCR
print("Initializing Language-Specific Fine-tuned Models...")
try:
    # You can change this path to where your weights are stored
    language_ocr = LanguageSpecificOCR(base_weights_dir=WEIGHTS_DIR)
    available_fine_tuned_langs = language_ocr.get_available_languages()
    print(f"✅ Fine-tuned models loaded for: {available_fine_tuned_langs}")
except Exception as e:
    print(f"Warning: Language-specific OCR initialization failed: {e}")
    language_ocr = None
    available_fine_tuned_langs = []

# TrOCR model configuration
TROCR_MODEL_CONFIG = {
    "base": "microsoft/trocr-base-handwritten",
    "large": "microsoft/trocr-large-handwritten"
}

# Initialize TrOCR for handwritten text recognition
print("Initializing TrOCR for handwritten text...")
try:
    selected_model = "base"
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
        "max_tokens": 1000
    }

    try:
        response = requests.post(GROQ_ENDPOINT, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            corrected_text = response.json()["choices"][0]["message"]["content"].strip()
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

# Utility functions
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
    
    # Apply Otsu's thresholding
    _, thresh_otsu = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    enhanced = cv2.equalizeHist(sharpened)
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    return resized, thresh, enhanced, thresh_otsu

def run_ocr_language_specific_with_finetuned(image_np, box_id, language):
    """OCR with fine-tuned model first, then fallback to EasyOCR and Tesseract"""
    # Try fine-tuned model first if available
    if language_ocr and language in available_fine_tuned_langs:
        try:
            print(f"Trying fine-tuned {language} model for region {box_id}")
            text, confidence = language_ocr.recognize_text(image_np, language)
            if text and confidence > 0.75: # Confidence threshold
                print(f"Fine-tuned model success: {text} (confidence: {confidence:.3f})")
                return (None, text, confidence)
            else:
                print(f"Fine-tuned model low confidence: {confidence:.3f}, falling back to EasyOCR")
        except Exception as e:
            print(f"Fine-tuned model error: {e}, falling back to EasyOCR")

    # Fallback to original EasyOCR + Tesseract approach
    return run_ocr_language_specific_fast(image_np, box_id, language)

def run_ocr_language_specific_fast(image_np, box_id, language):
    """Fast OCR for specific language using optimized multi-reader approach"""
    resized, thresh, enhanced, thresh_otsu = enhance_image_for_ocr(image_np)
    images_to_process = [resized, thresh, enhanced, thresh_otsu]
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
                # Use appropriate language for Tesseract
                if language == 'fr':
                    tesseract_text = pytesseract.image_to_string(resized, lang='fra+eng', config='--psm 6')
                else:
                    tesseract_text = pytesseract.image_to_string(resized, lang=tesseract_lang, config='--psm 6')
            
            if tesseract_text.strip():
                results.append((None, tesseract_text.strip(), 0.5))
        except Exception as e:
            print(f"Tesseract error for {language}: {e}")

    if results:
        best = max(results, key=lambda x: x[2] if len(x) == 3 else -1)
        return best

    return None

# Text reordering functions
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
    """Process a single OCR region with fine-tuned models and fallback"""
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
            # Use fine-tuned model first, then fallback
            ocr = run_ocr_language_specific_with_finetuned(crop_np, f"region_{idx}", language)
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

# ========================
# ROUTES
# ========================

# Root route - enhanced for tunneling
@app.route('/', methods=['GET', 'OPTIONS'])
def home():
    """Root endpoint with tunnel-friendly response"""
    if request.method == 'OPTIONS':
        return jsonify({"message": "CORS preflight successful"}), 200

    logger.info("Root endpoint accessed")
    return jsonify({
        "message": "Flask OCR backend is running!",
        "status": "success",
        "version": "2.0",
        "timestamp": time.time(),
        "endpoints": {
            "health": "/health",
            "api_test": "/api/test",
            "ocr_printed": "/result",
            "ocr_handwritten": "/handwritten_result",
            "audio": "/audio",
            "translate": "/translate",
            "summarize": "/summarize",
            "get_summary": "/get_summary",
            "data": "/data",
            "final_data": "/final_data"
        },
        "request_info": {
            "method": request.method,
            "remote_addr": request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr),
            "user_agent": request.headers.get('User-Agent', 'Unknown')
        },
        "available_models": {
            "fine_tuned_languages": available_fine_tuned_langs,
            "easyocr_languages": list(lang_codes.keys()),
            "trocr_available": trocr_processor is not None
        }
    })

# Enhanced API test route
@app.route('/api/test', methods=['GET', 'POST', 'OPTIONS'])
def test():
    """Enhanced test endpoint that handles all HTTP methods for tunnel testing"""
    if request.method == 'OPTIONS':
        return jsonify({"message": "CORS preflight successful"}), 200

    logger.info(f"API test endpoint accessed via {request.method}")

    response_data = {
        "message": "API test successful",
        "status": "success",
        "method": request.method,
        "timestamp": time.time(),
        "request_info": {
            "remote_addr": request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr),
            "user_agent": request.headers.get('User-Agent', 'Unknown'),
            "headers": dict(request.headers)
        },
        "server_info": {
            "flask_running": True,
            "trocr_available": trocr_processor is not None,
            "fine_tuned_models": available_fine_tuned_langs,
            "easyocr_readers": len([reader_en, reader_hi, reader_ru, reader_fr, reader_es, reader_ko, reader_de, reader_it, reader_tr])
        }
    }

    if request.method == 'POST':
        try:
            if request.is_json:
                response_data["request_data"] = request.get_json()
            elif request.form:
                response_data["form_data"] = dict(request.form)
        except Exception as e:
            response_data["request_data_error"] = str(e)

    return jsonify(response_data)

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring and tunnel verification"""
    logger.info("Health check endpoint accessed")

    health_status = {
        "status": "healthy",
        "service": "OCR Flask Backend",
        "timestamp": time.time(),
        "uptime": "running",
        "version": "2.0",
        "components": {
            "flask": True,
            "easyocr": True,
            "trocr": trocr_processor is not None,
            "fine_tuned_models": language_ocr is not None,
            "tesseract": True,
            "translator": True,
            "pygame": True,
            "groq": True
        },
        "directories": {
            "uploads": os.path.exists(UPLOAD_DIR),
            "static": os.path.exists(STATIC_DIR),
            "tmp": os.path.exists(TMP_DIR),
            "weights": os.path.exists(WEIGHTS_DIR)
        },
        "models_loaded": {
            "easyocr_readers": 9,
            "fine_tuned_languages": available_fine_tuned_langs,
            "trocr_model": "base" if trocr_processor else None
        }
    }

    return jsonify(health_status), 200

# **UPDATED: Summarization Route - Only summarize if text > 50 words**
@app.route('/summarize', methods=['POST'])
def summarize_extracted_text():
    """Generate summary from extracted OCR text and save as summary.txt - Only if text > 50 words"""
    try:
        print("=== Text summarization requested ===")

        text_to_summarize = request.form.get('text', '').strip()

        if not text_to_summarize:
            print("Error: No text provided for summarization")
            return jsonify({'error': 'No text provided for summarization'}), 400

        print(f"Text to summarize ({len(text_to_summarize)} chars): {text_to_summarize[:100]}...")

        word_count = len(text_to_summarize.split())
        print(f"Word count: {word_count}")

        if word_count <= 50:
            print(f"Text has only {word_count} words (≤50), returning extracted text instead of summary")

            summary_path = os.path.join(STATIC_DIR, 'summary.txt')
            if save_summary_to_file(text_to_summarize, summary_path):
                print(f"Original text saved as summary.txt: {summary_path}")

                summary_data = {
                    'original_text': text_to_summarize,
                    'summary': text_to_summarize,
                    'original_length': len(text_to_summarize),
                    'summary_length': len(text_to_summarize),
                    'word_count': word_count,
                    'compression_ratio': 1.0,
                    'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'action_taken': 'extracted_text_returned',
                    'reason': f'Text has only {word_count} words (≤50), not enough for summarization'
                }

                summary_json_path = os.path.join(STATIC_DIR, 'summary.json')
                with open(summary_json_path, 'w', encoding='utf-8') as f:
                    json.dump(summary_data, f, ensure_ascii=False, indent=4)

                return jsonify({
                    'success': True,
                    'message': f'Text too short for summarization ({word_count} words ≤ 50). Extracted text returned.',
                    'summary': text_to_summarize,
                    'original_length': len(text_to_summarize),
                    'summary_length': len(text_to_summarize),
                    'word_count': word_count,
                    'compression_ratio': 1.0,
                    'action_taken': 'extracted_text_returned',
                    'summary_file': 'summary.txt',
                    'summary_url': f'/static/summary.txt?t={int(time.time())}'
                }), 200

            else:
                print("Failed to save extracted text to file")
                return jsonify({'error': 'Failed to save extracted text to file'}), 500

        print("Generating summary with Groq API...")
        summary = summarize_text(text_to_summarize)

        if not summary or summary.startswith("Error"):
            print("Summarization failed")
            return jsonify({'error': 'Failed to generate summary'}), 500

        summary_path = os.path.join(STATIC_DIR, 'summary.txt')
        if save_summary_to_file(summary, summary_path):
            print(f"Summary saved to: {summary_path}")
            print(f"Summary preview: {summary[:100]}...")

            summary_data = {
                'original_text': text_to_summarize,
                'summary': summary,
                'original_length': len(text_to_summarize),
                'summary_length': len(summary),
                'word_count': word_count,
                'compression_ratio': round(len(summary) / len(text_to_summarize), 2),
                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'model_used': 'llama3-70b-8192',
                'action_taken': 'summary_generated'
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
                'word_count': word_count,
                'compression_ratio': round(len(summary) / len(text_to_summarize), 2),
                'action_taken': 'summary_generated',
                'summary_file': 'summary.txt',
                'summary_url': f'/static/summary.txt?t={int(time.time())}'
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

# **UNCHANGED: Keep handwritten result route exactly as it is (NO spell check, NO automatic summarization)**
@app.route('/handwritten_result', methods=['POST'])
def handwritten_result():
    """Dedicated route for handwritten text processing using TrOCR with preprocessing"""
    start_time = time.time()

    language = request.form.get('language', 'AutoDetect')
    enable_spell_check = request.form.get('enable_spell_check', 'false').lower() == 'true'
    max_length = int(request.form.get('max_length', '256'))

    print(f"Handwritten processing - Language: {language}, Spell check: {enable_spell_check}")
    print(f"Max text length: {max_length}")

    file = request.files.get('file')
    if not file:
        print("Error: No image provided")
        return jsonify({'error': 'No image provided'}), 400

    fname = file.filename
    img_path = os.path.join(UPLOAD_DIR, fname)
    file.save(img_path)
    print(f"Processing image: {img_path}")

    try:
        print("Applying preprocessing to improve handwritten text detection...")
        preprocessed_image = preprocess_image(img_path)
        preprocess_path = os.path.join(STATIC_DIR, 'preprocess.png')
        cv2.imwrite(preprocess_path, preprocessed_image)
        print(f"Preprocessed image saved as: {preprocess_path}")
        processing_image_path = img_path
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        print("Continuing with original image...")
        processing_image_path = img_path

    if trocr_processor is None:
        print("Error: TrOCR not available")
        return jsonify({'error': 'TrOCR not available. Please check initialization.'}), 500

    try:
        print("Processing handwritten text with TrOCR...")
        print("Detecting text regions...")
        result = trocr_processor.process_image(processing_image_path, max_length)

        if not result:
            print("Error: Processing failed")
            return jsonify({'error': 'TrOCR processing failed'}), 500

        print(f"Words detected: {result['total_detections']}")
        print(f"Processing time: {result['processing_time']}s")

        if result['results']:
            print("Sample detections:")
            for i, detection in enumerate(result['results'][:3]):
                confidence = detection.get('confidence', 1.0)
                text = detection.get('detected_text', '')
                print(f"  {i+1}. '{text}' (confidence: {confidence:.3f})")

        output = result['results']
        pygame.mixer.init()

        # **NO SPELL CHECK FOR HANDWRITTEN - Mark all as not spell-checked**
        for result_item in output:
            result_item['spell_checked'] = False
            result_item['original_text'] = result_item.get('detected_text', '')

        total_processing_time = round(time.time() - start_time, 1)
        print(f"Total handwritten processing completed in {total_processing_time} seconds")

        json_path = os.path.join(STATIC_DIR, 'data.json')
        data_json = {
            'batch_info': {
                'total_images': 1,
                'processed_successfully': 1,
                'processed_at': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'language': language,
            'is_handwritten': True,
            'spell_check_enabled': False,
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
                'spell_check_enabled': False,
                'processing_time': total_processing_time,
                'trocr_processing_time': result['processing_time'],
                'total_detections': len(reordered_output),
                'fast_processing': False,
                'processing_method': 'EasyOCR (detection) + TrOCR (recognition)',
                'model_used': selected_model,
                'max_length': max_length,
                'reordered': True,
                'summary_generated': False,
                'results': reordered_output
            }

            final_json_path = os.path.join(STATIC_DIR, 'final_data.json')
            with open(final_json_path, 'w', encoding='utf-8') as jf:
                json.dump(final_data_json, jf, ensure_ascii=False, indent=4)
            print("Reordered handwritten final_data.json saved successfully (no automatic summarization)")

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

# **UPDATED: Modified main result route with fine-tuned models and whole text spell check**
@app.route('/result', methods=['POST'])
def result():
    start_time = time.time()

    language = request.form.get('language', 'AutoDetect')
    is_handwritten = request.form.get('is_handwritten', 'false').lower() == 'true'
    enable_spell_check = request.form.get('enable_spell_check', 'false').lower() == 'true'

    print(f"Language selected: {language}")
    print(f"Handwritten mode: {is_handwritten}")
    print(f"Spell check enabled: {enable_spell_check}")
    print(f"Fine-tuned models available for: {available_fine_tuned_langs}")

    if is_handwritten:
        print("Redirecting to handwritten processing route...")
        return handwritten_result()

    print("Processing printed text with fine-tuned models and fast parallel processing...")

    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No image provided'}), 400

    fname = file.filename
    img_path = os.path.join(UPLOAD_DIR, fname)
    file.save(img_path)

    # Send to CRAFT for boxes (only for printed text)
    try:
        with open(img_path, 'rb') as f:
            craft_resp = requests.post(CRAFT_URL, files={'image': f})
        if craft_resp.status_code != 200:
            return jsonify({'error': 'CRAFT service failed'}), 500
        boxes = craft_resp.json()
    except Exception as e:
        print(f"CRAFT service error: {e}")
        return jsonify({'error': f'CRAFT service failed: {str(e)}'}), 500

    try:
        pil = Image.open(img_path).convert('RGB')
        img_np = np.array(pil)
    except Exception as e:
        print(f"Error loading image: {e}")
        return jsonify({'error': 'Failed to load image'}), 500

    pygame.mixer.init()

    # Process printed text with fine-tuned models and fast parallel processing
    print(f"Processing {len(boxes)} text regions with fine-tuned models and parallel processing...")
    args_list = [(box, idx, img_np, language, False) for idx, box in enumerate(boxes)]

    output = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_region_fast, args_list))

    for result in results:
        if result is not None:
            output.append(result)

    # Detect dominant language AFTER processing all regions
    if language == 'AutoDetect' and output:
        print("AutoDetect mode: Analyzing dominant language from all text...")
        dominant_language = detect_dominant_language(output)
        print(f"Dominant language detected: {dominant_language}")
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
        for result in output:
            result['spell_checked'] = False
            result['original_text'] = result.get('detected_text', '')

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
        'processing_method': 'CRAFT + Fine-tuned Models + EasyOCR',
        'fine_tuned_models_used': available_fine_tuned_langs,
        'results': output
    }

    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(data_json, jf, ensure_ascii=False, indent=4)
    print("Printed text data.json saved successfully")

    # Reorder the results and save as final_data.json (NO automatic summarization)
    try:
        print("\n--- Starting text reordering ---")
        reordered_output = reorder_ocr_results(output, line_threshold=25)

        final_data_json = {
            'language': language,
            'is_handwritten': False,
            'spell_check_enabled': enable_spell_check,
            'processing_time': processing_time,
            'total_detections': len(reordered_output),
            'fast_processing': True,
            'processing_method': 'CRAFT + Fine-tuned Models + EasyOCR',
            'fine_tuned_models_used': available_fine_tuned_langs,
            'reordered': True,
            'summary_generated': False,
            'results': reordered_output
        }

        final_json_path = os.path.join(STATIC_DIR, 'final_data.json')
        with open(final_json_path, 'w', encoding='utf-8') as jf:
            json.dump(final_data_json, jf, ensure_ascii=False, indent=4)
        print("Reordered printed text final_data.json saved successfully (no automatic summarization)")

    except Exception as e:
        print(f"Error during text reordering: {e}")
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

        text_to_convert = request.form.get('text', '').strip()
        is_all_text = request.form.get('isAllText', 'false').lower() == 'true'
        is_translated_audio = request.form.get('isTranslatedAudioRequest', 'false').lower() == 'true'

        print(f"Form data received:")
        print(f"  - text: {text_to_convert[:50]}..." if text_to_convert else "  - text: (empty)")
        print(f"  - isAllText: {is_all_text}")
        print(f"  - isTranslatedAudioRequest: {is_translated_audio}")

        if not text_to_convert:
            print("Error: No text provided for audio generation")
            return jsonify({'error': 'No text provided for audio generation'}), 400

        try:
            detected_lang = detect_language(text_to_convert)
            print(f"Detected language for TTS: {detected_lang}")

            if is_translated_audio:
                tts_lang = 'en'
                print("Using English for translated text audio")
            else:
                tts_lang = lang_codes.get(detected_lang, 'en')
                print(f"Using TTS language code: {tts_lang}")
        except Exception as lang_error:
            print(f"Language detection failed: {lang_error}, defaulting to English")
            tts_lang = 'en'

        if is_translated_audio:
            audio_filename = 'translated_data.mp3'
        elif is_all_text:
            audio_filename = 'alldata.mp3'
        else:
            audio_filename = 'data.mp3'

        audio_path = os.path.join(STATIC_DIR, audio_filename)

        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
                print(f"Removed existing {audio_filename}")

            print(f"Generating TTS audio using gTTS with language: {tts_lang}")
            tts = gTTS(text=text_to_convert, lang=tts_lang, slow=False)
            tts.save(audio_path)
            time.sleep(0.5)

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

# **UPDATED: Translation route - Return extracted text for English language**
@app.route('/translate', methods=['POST'])
def translate_text():
    """Translate text sent in the request to English. Return extracted text if already English."""
    try:
        print("=== Translation requested by user ===")

        text_to_translate = request.form.get('text', '').strip()

        if not text_to_translate:
            print("Error: No text provided for translation")
            return jsonify({'error': 'No text provided for translation'}), 400

        print(f"Text to translate ({len(text_to_translate)} chars): {text_to_translate[:100]}...")

        try:
            source_lang = detect_language(text_to_translate)
            print(f"Detected source language: {source_lang}")

            if source_lang == 'en':
                translated_text = text_to_translate
                print("Text is already in English, returning extracted text instead of translation")

                translation_data = {
                    'original_text': text_to_translate,
                    'translated_text': translated_text,
                    'source_language': source_lang,
                    'target_language': 'en',
                    'action_taken': 'extracted_text_returned',
                    'reason': 'Text is already in English',
                    'timestamp': time.time()
                }

                translate_path = os.path.join(STATIC_DIR, 'translate.json')
                with open(translate_path, 'w', encoding='utf-8') as f:
                    json.dump(translation_data, f, ensure_ascii=False, indent=4)

                print("Extracted text saved to translate.json (no translation needed)")

                return jsonify({
                    'success': True,
                    'message': 'Text is already in English. Extracted text returned.',
                    'original_text': text_to_translate,
                    'translated_text': translated_text,
                    'source_language': source_lang,
                    'target_language': 'en',
                    'action_taken': 'extracted_text_returned'
                }), 200

            else:
                print(f"Translating from {source_lang} to English...")
                translation = translator.translate(text_to_translate, src=source_lang, dest='en')
                translated_text = translation.text
                print(f"Translation completed: {translated_text[:100]}...")

        except Exception as lang_error:
            print(f"Language detection or translation failed: {lang_error}")
            try:
                translation = translator.translate(text_to_translate, dest='en')
                translated_text = translation.text
                source_lang = 'auto'
                print(f"Auto-translation completed: {translated_text[:100]}...")
            except Exception as trans_error:
                print(f"Translation failed completely: {trans_error}")
                return jsonify({'error': f'Translation failed: {str(trans_error)}'}), 500

        translation_data = {
            'original_text': text_to_translate,
            'translated_text': translated_text,
            'source_language': source_lang,
            'target_language': 'en',
            'action_taken': 'translation_completed',
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
            'target_language': 'en',
            'action_taken': 'translation_completed'
        }), 200

    except Exception as e:
        print(f"❌ Translation error: {e}")
        return jsonify({'error': f'Translation failed: {str(e)}'}), 500

# Static files route
@app.route('/static/<filename>')
def serve_static(filename):
    """Serve static files with proper CORS headers"""
    try:
        file_path = os.path.join(STATIC_DIR, filename)
        if not os.path.exists(file_path):
            print(f"Static file not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404

        if filename.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            response = jsonify(data)
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response

        return send_from_directory(STATIC_DIR, filename)

    except Exception as e:
        print(f"Error serving static file {filename}: {e}")
        return jsonify({'error': f'Error serving file: {str(e)}'}), 500

@app.route('/api/final_data')
def get_final_data_api():
    """API endpoint to get final OCR data with proper error handling"""
    try:
        json_path = os.path.join(STATIC_DIR, 'final_data.json')
        if not os.path.exists(json_path):
            print("final_data.json not found")
            return jsonify({'error': 'Final data not available yet. Please try again.'}), 404

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"Successfully served final_data.json with {len(data.get('results', []))} results")
        return jsonify(data)

    except json.JSONDecodeError as e:
        print(f"JSON decode error in final_data.json: {e}")
        return jsonify({'error': 'Invalid JSON data'}), 500

    except Exception as e:
        print(f"Error reading final_data.json: {e}")
        return jsonify({'error': f'Error reading data: {str(e)}'}), 500

if __name__ == '__main__':
    print("=" * 80)
    print("Starting Flask OCR Backend Server with Multi-Language Support")
    print("=" * 80)
    print("Available endpoints:")
    print("  GET / - Root endpoint")
    print("  GET /api/test - API test endpoint")
    print("  GET /health - Health check")
    print("  GET /data - Get OCR data")
    print("  GET /final_data - Get reordered OCR data")
    print("  POST /result - OCR processing (printed text)")
    print("  POST /handwritten_result - Handwritten OCR processing")
    print("  POST /audio - Audio generation")
    print("  POST /translate - Text translation")
    print("  POST /summarize - Text summarization")
    print("  GET /get_summary - Get saved summary")
    print("  GET /static/<filename> - Static files")
    print("=" * 80)

    # Check component status
    print("Component Status:")
    print(f"  ✅ Flask: Ready")
    print(f"  ✅ EasyOCR: {9} readers loaded")
    print(f"  {'✅' if trocr_processor else '❌'} TrOCR: {'Ready' if trocr_processor else 'Not available'}")
    print(f"  {'✅' if language_ocr else '❌'} Fine-tuned Models: {available_fine_tuned_langs if language_ocr else 'Not available'}")
    print(f"  ✅ Groq: Ready for spell check and summarization")
    print(f"  ✅ Directories: Created")
    print(f"  📁 Weights Directory: {WEIGHTS_DIR}")
    print("=" * 80)

    # Directory structure information
    print("Expected Weights Directory Structure:")
    print("  📁 ../../Weights/")
    print("    📁 English/")
    print("      📄 best_en_ocr_model.pth")
    print("    📁 French/")
    print("      📄 best_fr_ocr_model.pth")
    print("    📁 German/")
    print("      📄 best_de_ocr_model.pth")
    print("    📁 Italian/")
    print("      📄 best_it_ocr_model.pth")
    print("    📁 Spanish/")
    print("      📄 best_es_ocr_model.pth")
    print("    📁 Korean/")
    print("      📄 best_ko_ocr_model.pth")
    print("    📁 Hindi/")
    print("      📄 best_hi_ocr_model.pth")
    print("    📁 Turkish/")
    print("      📄 best_tr_ocr_model.pth")
    print("    📁 Russian/")
    print("      📄 best_ru_ocr_model.pth")
    print("=" * 80)

    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )
