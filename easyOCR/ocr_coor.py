import easyocr
from PIL import Image
import numpy as np
import json
import os

# Define relative paths
json_path = "../CRAFT Detection Model/result/coords_WhatsApp Image 2025-07-14 at 23.35.32_9972846f.json"
image_path = "../CRAFT Detection Model/result/res_WhatsApp Image 2025-07-14 at 23.35.32_9972846f.jpg"
output_folder = "output_folder"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Initialize separate readers for different languages
reader_ko = easyocr.Reader(['ko'], gpu=True)  # Korean
reader_ru_en = easyocr.Reader(['ru', 'en'], gpu=True)  # Russian and English
reader_hi_en = easyocr.Reader(['hi', 'en'], gpu=True)  # Hindi and English
reader_es = easyocr.Reader(['es'], gpu=True)  # Spanish

# Load the image
image_dir = os.path.dirname(os.path.abspath(json_path))
image_path_full = os.path.join(image_dir, os.path.basename(image_path))
if not os.path.exists(image_dir):
    raise FileNotFoundError(f"Directory not found: {image_dir}. Please ensure the CRAFT Detection Model/result folder exists relative to {os.path.abspath('.')}")
if not os.path.exists(image_path_full):
    available_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    raise FileNotFoundError(f"Image not found at {image_path_full}. Available files in {image_dir}: {available_files}")
image = Image.open(image_path_full)

# Load coordinates from the JSON file and debug the structure
with open(json_path, 'r') as f:
    data = json.load(f)
print("Loaded JSON data:", json.dumps(data, indent=2))
polygons = []
for item in data:
    if isinstance(item, dict) and "boxes" in item:
        polygons.extend(item["boxes"])

# Function to extract coordinates with fallback
def get_coordinates(polygon):
    for key in ["coordinates", "points", "vertices"]:
        if key in polygon:
            return polygon[key]
    raise KeyError(f"No valid coordinate key found in {polygon}. Expected 'coordinates', 'points', or 'vertices'.")

# Function to compute the bounding rectangle from a polygon
def get_bounding_rect(polygon):
    coords = get_coordinates(polygon)
    xs = [point[0] for point in coords]
    ys = [point[1] for point in coords]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [x_min, y_min, x_max, y_max]

# Function to run OCR with all readers and get the best result
def run_ocr_all(cropped_image_np):
    results = []
    results.append(("ko", reader_ko.readtext(cropped_image_np)))
    results.append(("ru/en", reader_ru_en.readtext(cropped_image_np)))
    results.append(("hi/en", reader_hi_en.readtext(cropped_image_np)))
    results.append(("es", reader_es.readtext(cropped_image_np)))
    
    # Flatten and find the best result (highest confidence)
    all_results = [item for sublist in [r[1] for r in results] for item in sublist]
    if all_results:
        best_result = max(all_results, key=lambda x: x[2] if len(x) == 3 else -1)
        if len(best_result) == 3:
            bbox, text, prob = best_result
            return text, prob
    return "No text detected", None

# Process each polygon
output_results = []
for polygon in polygons:
    try:
        # Extract coordinates and compute bounding rectangle
        coords = get_coordinates(polygon)
        x_min, y_min, x_max, y_max = get_bounding_rect(polygon)
        
        # Crop the image to the bounding rectangle and convert to RGB
        cropped_image = image.crop((x_min, y_min, x_max, y_max)).convert('RGB')
        cropped_image_np = np.array(cropped_image)
        
        # Run OCR and get the best result
        text, confidence = run_ocr_all(cropped_image_np)
        
        # Store the result
        output_results.append({
            "polygon": coords,
            "text": text,
            "confidence": confidence
        })
    except KeyError as e:
        print(f"Skipping invalid polygon: {e}")
        continue

# Save the results to a JSON file in the output folder
output_json_path = os.path.join(output_folder, "results.json")
with open(output_json_path, 'w') as f:
    json.dump(output_results, f, ensure_ascii=False, indent=4)

# Display the results in the console
# print(json.dumps(output_results, ensure_ascii=False, indent=4))