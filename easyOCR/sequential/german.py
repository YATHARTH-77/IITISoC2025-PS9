import easyocr
from PIL import Image
import cv2
import os
import numpy as np
import json
import pytesseract
import torch
import torch.nn as nn
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor

print("Running updated german_sequential.py - Version 2025-08-02 02:10 AM IST")  # Version marker

# Define paths
json_path = "../result/coords_b5c9010e9ecb4e818a50a6980ff64e3f.json"  # CRAFT coordinates JSON
image_path = "../result/res_b5c9010e9ecb4e818a50a6980ff64e3f.jpg"    # Input image
output_folder = "output_folder"
model_path = "../../Weights/German/best_german_ocr_model.pth"    # Path to your fine-tuned weights
character_list_path = "../../Weights/German/training_data/character_list.txt"  # Path to character list used during training
weights_path = "../../Weights/German/best_german_ocr_model.pth"

# Create output directories if they don’t exist
os.makedirs(output_folder, exist_ok=True)

# Initialize OCR reader for German and English
reader = easyocr.Reader(['de', 'en'], gpu=True)

# Set Tesseract path (adjust based on your system)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update this path

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
        # Flatten LSTM parameters to avoid memory fragmentation
        self.rnn.flatten_parameters()
        
        conv_features = self.cnn(x)
        b, c, h, w = conv_features.size()
        print(f"conv_features shape after CNN: {conv_features.shape}")  # Debug shape
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
        print(f"conv_features shape before RNN: {conv_features.shape}")  # Debug shape
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
        print("Loading German fine-tuned model...")
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
        
        print("German fine-tuned model loaded successfully!")
        
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
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")
image = Image.open(image_path)

# Load coordinates from the JSON file
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

# Function to run OCR with fine-tuned model, EasyOCR, and Tesseract sequentially
def run_ocr_all(image_np, box_id):
    # Preprocess image for fine-tuned model
    text_custom, conf_custom = "", 0.0
    if model is not None and preprocess is not None:
        try:
            img_pil = Image.fromarray(image_np)
            img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_tensor)
                text_custom, conf_custom = decode_output(output)
        except Exception as e:
            print(f"Fine-tuned model failed for box {box_id}: {e}")

    # Preprocess images for EasyOCR
    scale_factor = 2
    resized = cv2.resize(image_np, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    _, thresh_otsu = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    enhanced = cv2.equalizeHist(sharpened)
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    images_to_process = [resized, thresh, enhanced, thresh_otsu]

    # Run EasyOCR
    text_easyocr, conf_easyocr = "", 0.0
    try:
        results = []
        for img in images_to_process:
            results.extend(reader.readtext(img))
        if results:
            best_result = max(results, key=lambda x: x[2] if len(x) == 3 else -1)
            text_easyocr, conf_easyocr = best_result[1], best_result[2]
    except Exception as e:
        print(f"EasyOCR failed for box {box_id}: {e}")

    # Debug model contributions
    print(f"Region {box_id} - Fine-tuned: '{text_custom}' (Conf: {conf_custom:.2f}), EasyOCR: '{text_easyocr}' (Conf: {conf_easyocr:.2f})")

    # Define confidence thresholds
    threshold_custom = 0.7  # Threshold for fine-tuned model
    threshold_easyocr = 0.9  # Threshold for EasyOCR

    # Select best prediction based on confidence
    if conf_custom > threshold_custom:
        print(f"Region {box_id} - Using fine-tuned model result")
        return (None, text_custom, conf_custom)
    elif conf_easyocr > threshold_easyocr:
        print(f"Region {box_id} - Using EasyOCR result")
        return (None, text_easyocr, conf_easyocr)
    else:
        print(f"Region {box_id} - Using Tesseract result")
        text_tesseract = pytesseract.image_to_string(resized, lang='deu+eng', config='--psm 6')
        if text_tesseract.strip():
            return (None, text_tesseract.strip(), 0.5)
        else:
            # Save image if no text detected
            cv2.imwrite(os.path.join(output_folder, f"no_text_detected_{box_id}.png"), image_np)
            return None

# Function to process a single region
def process_region(args):
    polygon, idx = args
    try:
        # Reload image for this region to avoid threading issues
        with Image.open(image_path) as region_image:
            if region_image is None:
                raise ValueError("Image.open returned None")
            # Extract coordinates and crop image
            x_min, y_min, x_max, y_max = get_bounding_rect(polygon)
            cropped_image = region_image.crop((x_min, y_min, x_max, y_max)).convert('RGB')
            if cropped_image is None:
                raise ValueError("Cropping returned None")
            print(f"Region {idx} - Cropped image mode: {cropped_image.mode}, size: {cropped_image.size}")
            cropped_image_np = np.array(cropped_image)

        # Run OCR
        image_base = os.path.basename(image_path).split('.')[0]
        result = run_ocr_all(cropped_image_np, f"{image_base}_{idx}")
        if result and len(result) == 3:
            _, text, prob = result
            print(f"Region {idx} - Detected Text: {text} (Confidence: {prob:.2f})")

            # Store result
            return {
                "coordinates": get_coordinates(polygon),
                "detected_text": text,
                "confidence": prob,
            }
        else:
            return {
                "coordinates": get_coordinates(polygon),
                "detected_text": "No text detected",
                "confidence": 0.0,
            }
    except KeyError as e:
        print(f"Error processing region {idx}: Missing coordinates - {e}")
        return {
            "coordinates": [],
            "detected_text": f"Error: {e}",
            "confidence": 0.0,
        }
    except Exception as e:
        print(f"Error processing region {idx}: {e}")
        return {
            "coordinates": get_coordinates(polygon) if "coordinates" in polygon else [],
            "detected_text": f"Error: {e}",
            "confidence": 0.0,
        }

# Process each text region in parallel
output_results = []
with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust max_workers based on your system
    output_results = list(executor.map(process_region, zip(polygons, range(len(polygons)))))

# Save results to JSON
output_json_path = os.path.join(output_folder, "results_german.json")
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump({
        "image": image_path,
        "model_used": "fine-tuned + EasyOCR + Tesseract" if model is not None else "EasyOCR + Tesseract",
        "texts": output_results
    }, f, ensure_ascii=False, indent=4)

print(f"Results saved to {output_json_path}")
print(f"Processed {len(output_results)} text regions")
if model is not None:
    print("Used fine-tuned German model as primary OCR method")
else:
    print("Used EasyOCR and Tesseract (fine-tuned model not available)")