import easyocr
from PIL import Image
from googletrans import Translator
import cv2
import os
from gtts import gTTS
from langdetect import detect
import numpy as np
import json
import pytesseract
import torch
import torch.nn as nn
from torchvision import transforms

print("Running final updated russian.py - Version 2025-07-31 10:15 PM IST")  # Unique version marker

# Define paths
json_path = "../result/coords_b5c9010e9ecb4e818a50a6980ff64e3f.json"  # CRAFT coordinates JSON
image_path = "../result/res_b5c9010e9ecb4e818a50a6980ff64e3f.jpg"    # Input image
output_folder = "output_folder"
model_path = "../FineTune/Russian/best_russian_ocr_model.pth"  # Path to your fine-tuned weights
character_list_path = "../FineTune/Russian/training_data/character_list.txt"  # Path to character list
weights_path = "../FineTune/Russian/best_russian_ocr_model.pth"

# Create output directories if they don’t exist
os.makedirs(output_folder, exist_ok=True)
print(f"Output folder created or verified: {output_folder}")

# Initialize OCR reader for Russian and English
reader = easyocr.Reader(['ru', 'en'], gpu=True)
print("EasyOCR reader initialized for languages: ru, en")

# Set Tesseract path (adjust based on your system)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update this path
print("Tesseract path set")

# Initialize translator
translator = Translator()
print("Translator initialized")

# Language code mapping for gTTS
lang_codes = {
    'ko': 'ko', 'hi': 'hi', 'ru': 'ru', 'es': 'es', 'fr': 'fr',
    'de': 'de', 'it': 'it', 'tr': 'tr', 'en': 'en'
}

# Define the fine-tuned model
class SimpleCRNN(nn.Module):
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

# Try to load fine-tuned model if available
model = None
charset = None
preprocess = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists(model_path):
    try:
        print("Loading Russian fine-tuned model...")
        checkpoint = torch.load(model_path, map_location=device)
        print(f"Checkpoint keys: {checkpoint.keys()}")
        
        # Extract character list and model info
        character_list = checkpoint['character_list']
        print(f"Character list length: {len(character_list)}")
        
        # Get num_classes from checkpoint
        num_classes = checkpoint['model_state_dict']['classifier.bias'].shape[0]
        print(f"Setting num_classes to {num_classes} from checkpoint")
        
        # Create charset for decoding
        charset = [''] + character_list  # Add blank token at the beginning
        print(f"Final charset length: {len(charset)}")
        
        # Initialize and load model
        model = SimpleCRNN(num_classes=num_classes).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Image preprocessing for fine-tuned model
        preprocess = transforms.Compose([
            transforms.Resize((32, 100)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        print("Russian fine-tuned model loaded successfully!")
        
    except Exception as e:
        print(f"Failed to load fine-tuned model: {e}")
        print("Continuing with EasyOCR and Tesseract only...")
        model = None

# Decode model output (only if model is loaded)
def decode_output(output):
    if charset is None:
        return "", 0.0
    
    output = output.cpu().detach().numpy()
    pred = np.argmax(output, axis=2)
    text = ""
    for p in pred[0]:
        if p != 0 and (not text or text[-1] != charset[p]):  # Remove blanks and duplicates
            if p < len(charset):  # Safety check
                text += charset[p]
    confidence = np.mean(np.max(output, axis=2))  # Simple confidence metric
    return text, confidence

# Load the image
print(f"Checking image file: {os.path.exists(image_path)}")
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")
image = Image.open(image_path)
print("Image loaded successfully")

# Load coordinates from the JSON file
print(f"Checking JSON file: {os.path.exists(json_path)}")
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
print("JSON structure:", json.dumps(data, indent=2)[:500] + "..." if len(json.dumps(data, indent=2)) > 500 else json.dumps(data, indent=2))

if isinstance(data, list):
    polygons = [item["boxes"] for item in data if "boxes" in item]
    polygons = [poly for sublist in polygons for poly in sublist]
elif isinstance(data, dict) and "boxes" in data:
    polygons = data["boxes"]
else:
    raise ValueError("JSON format invalid. Expected a list of dicts with 'boxes' key or a dict with 'boxes' key.")
print(f"Extracted {len(polygons)} polygons from JSON")

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
    return [min(xs), min(ys), max(xs), max(ys)]

# Function to run OCR with fine-tuned model, EasyOCR, and Tesseract fallback
def run_ocr_all(image_np, box_id):
    print(f"Processing image for box_id: {box_id}")
    # Try fine-tuned model first (if available)
    if model is not None and preprocess is not None:
        try:
            img_pil = Image.fromarray(image_np)
            img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(img_tensor)
                text, confidence = decode_output(output)
            
            if text and confidence > 0.7:  # Adjust threshold as needed
                return (None, text, confidence)
        except Exception as e:
            print(f"Fine-tuned model failed for box {box_id}: {e}")

    # Enhance image for EasyOCR and Tesseract
    scale_factor = 2
    resized = cv2.resize(image_np, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    enhanced = cv2.equalizeHist(sharpened)
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Try EasyOCR with multiple enhanced versions
    images_to_process = [resized, thresh, enhanced]
    results = []
    for img in images_to_process:
        results.extend(reader.readtext(img))
    print(f"EasyOCR results count: {len(results)}")

    # Return best EasyOCR result if found
    if results:
        best_result = max(results, key=lambda x: x[2] if len(x) == 3 else -1)
        return best_result

    # Fallback to Tesseract
    text = pytesseract.image_to_string(resized, lang='rus+eng', config='--psm 6')
    if text.strip():
        return (None, text.strip(), 0.5)

    # Save image if no text detected
    cv2.imwrite(os.path.join(output_folder, f"no_text_detected_{box_id}.png"), image_np)
    print(f"Saved no_text_detected_{box_id}.png")
    return None

# Function to detect language
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'ru'  # Fallback to Russian for this script

# Function to generate speech
def generate_speech(text, lang_code, filename):
    try:
        if lang_code in lang_codes:
            tts = gTTS(text=text, lang=lang_codes[lang_code], slow=False)
            tts.save(filename)
            return True
    except Exception as e:
        print(f"Speech generation failed: {e}")
    return False

# Process each text region
output_results = []
image_base = os.path.basename(image_path).split('.')[0]

for idx, polygon in enumerate(polygons):
    try:
        print(f"Processing region {idx}")
        # Extract coordinates and crop image
        x_min, y_min, x_max, y_max = get_bounding_rect(polygon)
        cropped_image = image.crop((x_min, y_min, x_max, y_max)).convert('RGB')
        cropped_image_np = np.array(cropped_image)
        print(f"Cropped image shape for region {idx}: {cropped_image_np.shape}")

        # Run OCR
        best_result = run_ocr_all(cropped_image_np, f"{image_base}_{idx}")
        if best_result and len(best_result) == 3:
            bbox, text, prob = best_result
            print(f"Region {idx} - Detected Text: {text} (Confidence: {prob:.2f})")

            # Detect language
            detected_lang = detect_language(text)
            
            # Translate to English
            try:
                translated_text = translator.translate(text, dest='en').text
                print(f"Translated text for region {idx}: {translated_text}")
            except Exception as e:
                translated_text = f"Translation failed: {e}"
                print(f"Translation error for region {idx}: {e}")

            # Store result
            result_data = {
                "coordinates": get_coordinates(polygon),
                "detected_text": text,
                "translated_text": translated_text,
                "confidence": prob,
                "detected_language": detected_lang
            }
            
            # Generate speech file
            speech_filename = os.path.join(output_folder, f"speech_{image_base}_{idx}.mp3")
            if generate_speech(text, detected_lang, speech_filename):
                result_data["speech_file"] = speech_filename
            
            output_results.append(result_data)
        else:
            output_results.append({
                "coordinates": get_coordinates(polygon),
                "detected_text": "No text detected",
                "translated_text": "N/A",
                "confidence": 0.0,
                "detected_language": "unknown"
            })
            print(f"Region {idx} - No text detected")
    except KeyError as e:
        print(f"Error processing region {idx}: Missing coordinates - {e}")
        output_results.append({
            "coordinates": [],
            "detected_text": f"Error: {e}",
            "translated_text": "N/A",
            "confidence": 0.0,
            "detected_language": "unknown"
        })
    except Exception as e:
        print(f"Error processing region {idx}: {e}")
        output_results.append({
            "coordinates": get_coordinates(polygon) if "coordinates" in polygon else [],
            "detected_text": f"Error: {e}",
            "translated_text": "N/A",
            "confidence": 0.0,
            "detected_language": "unknown"
        })

# Save results to JSON
output_json_path = os.path.join(output_folder, "results_russian.json")
print(f"Saving results to: {output_json_path}")
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump({
        "image": image_path,
        "model_used": "fine-tuned + EasyOCR + Tesseract" if model is not None else "EasyOCR + Tesseract",
        "texts": output_results
    }, f, ensure_ascii=False, indent=4)

print(f"Results saved to {output_json_path}")
print(f"Processed {len(output_results)} text regions")
if model is not None:
    print("Used fine-tuned Russian model as primary OCR method")
else:
    print("Used EasyOCR and Tesseract (fine-tuned model not available)")