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

print("Running updated spanish.py - Version 2025-07-31 Updated")  # Version marker

# Define paths
json_path = "../result/coords_b5c9010e9ecb4e818a50a6980ff64e3f.json"  # CRAFT coordinates JSON
image_path = "../result/res_b5c9010e9ecb4e818a50a6980ff64e3f.jpg"    # Input image
output_folder = "output_folder"
model_path = "../../Weights/spanish/best_spanish_ocr_model.pth"    # Path to your fine-tuned weights
character_list_path = "../../Weights/spanish/training_data/character_list.txt"  # Path to character list used during training
weights_path = "../../Weights/spanish/best_spanish_ocr_model.pth"

# Create output directories if they don't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize OCR reader for Spanish and English
reader = easyocr.Reader(['es', 'en'], gpu=True)

# Set Tesseract path (adjust based on your system)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update this path

# Define the fine-tuned model (same architecture as French/German)
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

# Initialize model variables
model = None
charset = None
preprocess = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Try to load Spanish fine-tuned model
if os.path.exists(weights_path):
    try:
        print("Loading Spanish fine-tuned model...")
        checkpoint = torch.load(weights_path, map_location=device)
        print(f"Checkpoint keys: {checkpoint.keys()}")
        
        # Extract character list from checkpoint
        character_list = checkpoint['character_list']
        print(f"Character list: {character_list}")
        print(f"Length of character_list: {len(character_list)}")
        
        # Get num_classes directly from the saved model weights
        num_classes = checkpoint['model_state_dict']['classifier.bias'].shape[0]
        print(f"Setting num_classes to {num_classes} from checkpoint")
        
        # Create charset for decoding (add blank token at index 0)
        charset = [''] + character_list  # Add blank token at the beginning
        print(f"Final charset length: {len(charset)}")
        
        # Instantiate the model with the correct num_classes
        model = SimpleCRNN(num_classes=num_classes).to(device)
        print(f"Model classifier weight shape: {model.classifier.weight.shape}")
        
        # Load the state dictionary
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Image preprocessing for the fine-tuned model
        preprocess = transforms.Compose([
            transforms.Resize((32, 100)),  # Adjust size based on your training setup
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        print("Spanish fine-tuned model loaded successfully!")
        
    except Exception as e:
        print(f"Failed to load Spanish fine-tuned model: {e}")
        print("Continuing with EasyOCR and Tesseract only...")
        model = None
else:
    print(f"Spanish fine-tuned model not found at {weights_path}")
    print("Using EasyOCR and Tesseract only...")

# Decode model output
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

# Load coordinates from the JSON file and debug structure
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
print("JSON structure:", json.dumps(data, indent=2)[:300] + "..." if len(json.dumps(data, indent=2)) > 300 else json.dumps(data, indent=2))

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

# Function to run OCR with fine-tuned model, EasyOCR, and Tesseract fallback
def run_ocr_all(image_np, box_id):
    # Try fine-tuned model first (if available)
    if model is not None and preprocess is not None:
        try:
            # Convert to PIL Image for fine-tuned model
            img_pil = Image.fromarray(image_np)
            img_tensor = preprocess(img_pil).unsqueeze(0).to(device)

            # Run inference with fine-tuned model
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

    # Return best EasyOCR result if found
    if results:
        best_result = max(results, key=lambda x: x[2] if len(x) == 3 else -1)
        return best_result

    # Fallback to Tesseract if no EasyOCR results
    text = pytesseract.image_to_string(resized, lang='spa+eng', config='--psm 6')
    if text.strip():
        return (None, text.strip(), 0.5)  # Placeholder confidence

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
            cropped_image_np = np.array(cropped_image)

        # Run OCR
        image_base = os.path.basename(image_path).split('.')[0]
        best_result = run_ocr_all(cropped_image_np, f"{image_base}_{idx}")
        if best_result and len(best_result) == 3:
            bbox, text, prob = best_result
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
output_json_path = os.path.join(output_folder, "results_spanish.json")
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump({
        "image": image_path,
        "model_used": "fine-tuned + EasyOCR + Tesseract" if model is not None else "EasyOCR + Tesseract",
        "total_regions": len(output_results),
        "texts": output_results
    }, f, ensure_ascii=False, indent=4)

print(f"Results saved to {output_json_path}")
print(f"Processed {len(output_results)} text regions")
if model is not None:
    print("✓ Fine-tuned Spanish model was used as primary OCR method")
else:
    print("• Using EasyOCR and Tesseract (fine-tuned model not available)")