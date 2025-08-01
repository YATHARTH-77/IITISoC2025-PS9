import easyocr
from PIL import Image, ImageFile
import cv2
import os
import numpy as np
import json
import pytesseract
from langdetect import detect
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn as nn
from types import MethodType

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define SimpleCRNN model
class SimpleCRNN(nn.Module):
    def __init__(self, num_classes, img_h=32, hidden_size=256):
        super(SimpleCRNN, self).__init__()
        self.img_h = img_h
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
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
        
        self.rnn = nn.LSTM(512, hidden_size, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        conv_features = self.cnn(x)
        b, c, h, w = conv_features.size()
        if h == 1:
            conv_features = conv_features.squeeze(2)
            conv_features = conv_features.permute(0, 2, 1)
        else:
            conv_features = conv_features.view(b, c, h * w).permute(0, 2, 1)
        rnn_output, _ = self.rnn(conv_features)
        output = self.classifier(rnn_output)
        output = output.permute(1, 0, 2)  # For CTC
        return output

# Monkey-patch EasyOCR's get_recognizer function
def custom_get_recognizer(self, recog_network, network_params, character_list, weights_path, device, opt):
    if hasattr(self, 'custom_weights') and hasattr(self, 'custom_character_lists') and self.custom_weights and self.custom_character_lists and weights_path in self.custom_weights.values():
        lang = [k for k, v in self.custom_weights.items() if v == weights_path][0]
        num_classes = len(self.custom_character_lists[lang]) + 1  # +1 for CTC blank
        model = SimpleCRNN(num_classes=num_classes).to(device)
        checkpoint = torch.load(self.custom_weights[lang], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        from easyocr.model.vgg_model import CRNN_VGG_BiLSTM_CTC
        model = CRNN_VGG_BiLSTM_CTC(network_params).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
    
    model.eval()
    return model

# Monkey-patch Reader class to support custom weights
def custom_reader_init(self, lang_list, gpu=True, model_storage_directory=None, user_network_directory=None, recog_network='standard', download_enabled=True, custom_weights=None, custom_character_lists=None):
    from easyocr.utils import get_image_list, get_paragraph
    from easyocr.detection import get_detector, get_textbox
    from easyocr.recognition import get_text
    from easyocr.utils import group_text_box, get_image_list, reformat_input, reformat_input_batched
    from easyocr.config import get_model_path, get_character_list

    self.lang_list = lang_list
    self.gpu = gpu and torch.cuda.is_available()
    self.device = torch.device('cuda' if self.gpu else 'cpu')
    self.model_storage_directory = model_storage_directory
    self.user_network_directory = user_network_directory
    self.recog_network = recog_network
    self.download_enabled = download_enabled
    self.custom_weights = custom_weights
    self.custom_character_lists = custom_character_lists
    
    self.character_list = {}
    self.model_path = {}
    for lang in lang_list:
        self.character_list[lang] = get_character_list(lang, model_storage_directory, user_network_directory)
        self.model_path[lang] = get_model_path(lang, model_storage_directory, user_network_directory, download_enabled)
    
    self.detector = get_detector(self.model_path.get('detector', None), self.device, quantize=True)
    
    self.recognizer = {}
    for lang in lang_list:
        network_params = {'input_channel': 1, 'output_channel': 512, 'hidden_size': 512}
        self.recognizer[lang] = self.get_recognizer(recog_network, network_params, self.character_list[lang], self.model_path[lang], self.device, None)
    
    self.get_image_list = get_image_list
    self.get_paragraph = get_paragraph
    self.get_textbox = get_textbox
    self.get_text = get_text
    self.group_text_box = group_text_box
    self.reformat_input = reformat_input
    self.reformat_input_batched = reformat_input_batched

# Define paths
json_path = "result/coords_7d81721481ec43c08d2fe02ad90fdfab.json"
image_path = "result/res_7d81721481ec43c08d2fe02ad90fdfab.jpg"
output_folder = "output_folder"
weights_paths = {
    'en': "../Weights/English/best_english_ocr_model.pth",
    'ko': "../Weights/Korean/best_korean_ocr_model.pth",
    'hi': "../Weights/Hindi/best_hindi_ocr_model.pth",
    'ru': "../Weights/Russian/best_russian_ocr_model.pth",
    'es': "../Weights/Spanish/best_spanish_ocr_model.pth",
    'fr': "../Weights/French/best_french_ocr_model.pth",
    'de': "../Weights/German/best_german_ocr_model.pth",
    'it': "../Weights/Italian/best_italian_ocr_model.pth",
    'tr': "../Weights/Turkish/best_turkish_ocr_model.pth",
}

# Create output directories
os.makedirs(output_folder, exist_ok=True)

# Load character lists from checkpoints
custom_character_lists = {}
for lang, path in weights_paths.items():
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location='cpu')
        custom_character_lists[lang] = checkpoint['character_list']
    else:
        print(f"Warning: Weights not found for {lang}: {path}. Using default EasyOCR weights.")

# Monkey-patch EasyOCR Reader
easyocr.Reader.get_recognizer = MethodType(custom_get_recognizer, easyocr.Reader)
easyocr.Reader.__init__ = custom_reader_init

# Initialize OCR readers with custom weights
reader_ko_en = easyocr.Reader(['ko', 'en'], gpu=True, custom_weights=weights_paths, custom_character_lists=custom_character_lists)
reader_hi_en = easyocr.Reader(['hi', 'en'], gpu=True, custom_weights=weights_paths, custom_character_lists=custom_character_lists)
reader_ru_en = easyocr.Reader(['ru', 'en'], gpu=True, custom_weights=weights_paths, custom_character_lists=custom_character_lists)
reader_multi = easyocr.Reader(['es', 'fr', 'de', 'it', 'tr'], gpu=True, custom_weights=weights_paths, custom_character_lists=custom_character_lists)

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Language code to full name mapping
lang_names = {
    'ko': 'Korean', 'hi': 'Hindi', 'ru': 'Russian', 'es': 'Spanish', 'fr': 'French',
    'de': 'German', 'it': 'Italian', 'tr': 'Turkish', 'en': 'English'
}

# Load image to get dimensions
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")
try:
    with Image.open(image_path) as image:
        image_width, image_height = image.size
    print(f"Image dimensions: {image_width}x{image_height}")
except Exception as e:
    raise Exception(f"Failed to load image {image_path}: {e}")

# Load coordinates from JSON
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
if isinstance(data, list):
    polygons = [item["boxes"] for item in data if "boxes" in item]
    polygons = [poly for sublist in polygons for poly in sublist]
elif isinstance(data, dict) and "boxes" in data:
    polygons = data["boxes"]
else:
    raise ValueError("JSON format invalid.")

# Function to extract coordinates
def get_coordinates(polygon):
    if isinstance(polygon, dict) and "coordinates" in polygon:
        return polygon["coordinates"]
    raise KeyError(f"No 'coordinates' key found in {polygon}")

# Function to compute bounding rectangle
def get_bounding_rect(polygon):
    coords = get_coordinates(polygon)
    xs = [point[0] for point in coords]
    ys = [point[1] for point in coords]
    x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image_width, x_max)
    y_max = min(image_height, y_max)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Invalid bounding box: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
    return [x_min, y_min, x_max, y_max]

# Function to run OCR with all readers
def run_ocr_all(image_np, box_id):
    scale_factor = 2
    resized = cv2.resize(image_np, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    enhanced = cv2.equalizeHist(sharpened)
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    images_to_process = [resized, thresh, enhanced]
    results = []
    for img in images_to_process:
        results.extend(reader_ko_en.readtext(img))
        results.extend(reader_hi_en.readtext(img))
        results.extend(reader_ru_en.readtext(img))
        results.extend(reader_multi.readtext(img))

    if not results:
        text = pytesseract.image_to_string(resized, lang='hin+eng+kor+rus+spa+fra+deu+ita+tur', config='--psm 6')
        if text.strip():
            results.append((None, text.strip(), 0.5))

    if not results:
        cv2.imwrite(os.path.join(output_folder, f"no_text_detected_{box_id}.png"), image_np)

    if results:
        best_result = max(results, key=lambda x: x[2] if len(x) == 3 else -1)
        return best_result
    return None

# Function to detect language
def detect_language(text):
    try:
        lang_code = detect(text)
        return lang_names.get(lang_code, 'English')
    except:
        return 'English'

# Function to process a single region
def process_region(args):
    polygon, idx = args
    try:
        x_min, y_min, x_max, y_max = get_bounding_rect(polygon)
        print(f"Region {idx} - Coordinates: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")

        with Image.open(image_path) as region_image:
            if region_image is None:
                raise ValueError("Image.open returned None")
            cropped_image = region_image.crop((x_min, y_min, x_max, y_max)).convert('RGB')
            if cropped_image is None:
                raise ValueError("Cropping returned None")
            print(f"Region {idx} - Cropped image mode: {cropped_image.mode}, size: {cropped_image.size}")
            cropped_image_np = np.array(cropped_image)

        image_base = os.path.basename(image_path).split('.')[0]
        best_result = run_ocr_all(cropped_image_np, f"{image_base}_{idx}")
        if best_result and len(best_result) == 3:
            bbox, text, prob = best_result
            detected_lang = detect_language(text)
            print(f"Region {idx} - Detected Text: {text} (Language: {detected_lang}, Confidence: {prob:.2f})")
            return {
                "coordinates": get_coordinates(polygon),
                "detected_text": text,
                "detected_language": detected_lang,
                "confidence": prob,
            }
        else:
            return {
                "coordinates": get_coordinates(polygon),
                "detected_text": "No text detected",
                "detected_language": "N/A",
                "confidence": 0.0,
            }
    except Exception as e:
        print(f"Error processing region {idx}: {e}")
        return {
            "coordinates": get_coordinates(polygon) if "coordinates" in polygon else [],
            "detected_text": f"Error: {e}",
            "detected_language": "N/A",
            "confidence": 0.0,
        }

# Process regions in parallel
output_results = []
with ThreadPoolExecutor(max_workers=8) as executor:
    output_results = list(executor.map(process_region, zip(polygons, range(len(polygons)))))

# Save results to JSON
output_json_path = os.path.join(output_folder, "results.json")
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump({"image": image_path, "texts": output_results}, f, ensure_ascii=False, indent=4)

print(f"Results saved to {output_json_path}")