import easyocr
from PIL import Image, ImageFile
import cv2
import os
import numpy as np
import json
import pytesseract
from langdetect import detect
from concurrent.futures import ThreadPoolExecutor

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define paths
json_path = "result/coords_7d81721481ec43c08d2fe02ad90fdfab.json"  # CRAFT coordinates JSON
image_path = "result/res_7d81721481ec43c08d2fe02ad90fdfab.jpg"     # Input image
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

# Language code to full name mapping
lang_names = {
    'ko': 'Korean', 'hi': 'Hindi', 'ru': 'Russian', 'es': 'Spanish', 'fr': 'French',
    'de': 'German', 'it': 'Italian', 'tr': 'Turkish', 'en': 'English'
}

# Load the image to get dimensions
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")
try:
    with Image.open(image_path) as image:
        image_width, image_height = image.size
    print(f"Image dimensions: {image_width}x{image_height}")
except Exception as e:
    raise Exception(f"Failed to load image {image_path} for dimension check: {e}")

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
    x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
    # Validate coordinates
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image_width, x_max)
    y_max = min(image_height, y_max)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Invalid bounding box: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
    return [x_min, y_min, x_max, y_max]

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
        lang_code = detect(text)
        return lang_names.get(lang_code, 'English')  # Default to English if not in mapping
    except:
        return 'English'  # Fallback to English

# Function to process a single region
def process_region(args):
    polygon, idx = args
    try:
        # Extract coordinates
        x_min, y_min, x_max, y_max = get_bounding_rect(polygon)
        print(f"Region {idx} - Coordinates: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")

        # Reload image for this region to avoid threading issues
        try:
            with Image.open(image_path) as region_image:
                if region_image is None:
                    raise ValueError("Image.open returned None")
                # Crop the image
                cropped_image = region_image.crop((x_min, y_min, x_max, y_max)).convert('RGB')
                if cropped_image is None:
                    raise ValueError("Cropping returned None")
                print(f"Region {idx} - Cropped image mode: {cropped_image.mode}, size: {cropped_image.size}")
        except Exception as e:
            print(f"Error cropping region {idx}: {e}")
            return {
                "coordinates": get_coordinates(polygon),
                "detected_text": f"Error cropping image: {e}",
                "detected_language": "N/A",
                "confidence": 0.0,
            }

        try:
            cropped_image_np = np.array(cropped_image)
        except Exception as e:
            print(f"Error converting cropped image to NumPy array for region {idx}: {e}")
            return {
                "coordinates": get_coordinates(polygon),
                "detected_text": f"Error converting image to NumPy: {e}",
                "detected_language": "N/A",
                "confidence": 0.0,
            }

        # Run OCR
        image_base = os.path.basename(image_path).split('.')[0]
        best_result = run_ocr_all(cropped_image_np, f"{image_base}_{idx}")
        if best_result and len(best_result) == 3:
            bbox, text, prob = best_result
            detected_lang = detect_language(text)
            print(f"Region {idx} - Detected Text: {text} (Language: {detected_lang}, Confidence: {prob:.2f})")

            # Store result
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
    except KeyError as e:
        print(f"Error processing region {idx}: Missing coordinates - {e}")
        return {
            "coordinates": [],
            "detected_text": f"Error: {e}",
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

# Process text regions in parallel
output_results = []
with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust max_workers based on your system
    output_results = list(executor.map(process_region, zip(polygons, range(len(polygons)))))

# Save results to JSON
output_json_path = os.path.join(output_folder, "results.json")
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump({"image": image_path, "texts": output_results}, f, ensure_ascii=False, indent=4)

print(f"Results saved to {output_json_path}")