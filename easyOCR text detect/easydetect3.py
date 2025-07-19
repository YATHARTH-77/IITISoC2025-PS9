import os
import json
import numpy as np
from PIL import Image, ImageDraw
import easyocr
import torch

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Define input and output folders
input_folder = "input_folder"  # Replace with your input folder path
output_folder = "output_folder"  # Replace with your output folder path
os.makedirs(output_folder, exist_ok=True)

# List all image files in the input folder
supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(supported_extensions)]

# Initialize four EasyOCR Readers
# Reader 1: Korean and English
reader_ko = easyocr.Reader(['ko', 'en'], gpu=torch.cuda.is_available())
# Reader 2: Hindi and English (for Devanagari)
reader_hi = easyocr.Reader(['hi', 'en'], gpu=torch.cuda.is_available())
# Reader 3: Russian and English (for Cyrillic)
reader_ru = easyocr.Reader(['ru', 'en'], gpu=torch.cuda.is_available())
# Reader 4: English, Spanish, Italian, French, German, Turkish
reader_multi = easyocr.Reader(['en', 'es', 'it', 'fr', 'de', 'tr'], gpu=torch.cuda.is_available())

# Function to check if two boxes overlap significantly
def boxes_overlap(box1, box2, threshold=0.5):
    # box1, box2 are [x1, y1, x2, y2, x3, y3, x4, y4]
    # Convert to bounding rectangle for simplicity
    def get_rect(box):
        x_coords = box[0::2]
        y_coords = box[1::2]
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
    
    rect1 = get_rect(box1)
    rect2 = get_rect(box2)
    
    x1_1, y1_1, x2_1, y2_1 = rect1
    x1_2, y1_2, x2_2, y2_2 = rect2
    
    # Calculate intersection
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return False
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    iou = intersection_area / min(area1, area2)
    
    return iou > threshold

# Process each image
for img_path in image_paths:
    print(f"\nProcessing {img_path}")
    base_name, ext = os.path.splitext(os.path.basename(img_path))
    
    # Load image
    pil_img = Image.open(img_path).convert('RGB')
    img_array = np.array(pil_img)
    img_height, img_width = img_array.shape[:2]

    # Run EasyOCR with all four readers
    results_ko = reader_ko.readtext(img_path, rotation_info=[0, 90, 180, 270], low_text=0.3)
    results_hi = reader_hi.readtext(img_path, rotation_info=[0, 90, 180, 270], low_text=0.3)
    results_ru = reader_ru.readtext(img_path, rotation_info=[0, 90, 180, 270], low_text=0.3)
    results_multi = reader_multi.readtext(img_path, rotation_info=[0, 90, 180, 270], low_text=0.3)
    
    # Extract and convert bounding boxes
    pixel_boxes = []
    
    # Process Korean + English results
    for (box, text, score) in results_ko:
        flat_box = [coord for point in box for coord in point]
        if len(flat_box) == 8:
            pixel_box = [int(coord) for coord in flat_box]
            pixel_boxes.append(pixel_box)
        else:
            print(f"Invalid box format (Korean/English): {box}")
    
    # Process Hindi + English results, avoiding duplicates
    for (box, text, score) in results_hi:
        flat_box = [coord for point in box for coord in point]
        if len(flat_box) != 8:
            print(f"Invalid box format (Hindi/English): {box}")
            continue
        pixel_box = [int(coord) for coord in flat_box]
        if not any(boxes_overlap(pixel_box, existing_box) for existing_box in pixel_boxes):
            pixel_boxes.append(pixel_box)
    
    # Process Russian + English results, avoiding duplicates
    for (box, text, score) in results_ru:
        flat_box = [coord for point in box for coord in point]
        if len(flat_box) != 8:
            print(f"Invalid box format (Russian/English): {box}")
            continue
        pixel_box = [int(coord) for coord in flat_box]
        if not any(boxes_overlap(pixel_box, existing_box) for existing_box in pixel_boxes):
            pixel_boxes.append(pixel_box)
    
    # Process other languages' results, avoiding duplicates
    for (box, text, score) in results_multi:
        flat_box = [coord for point in box for coord in point]
        if len(flat_box) != 8:
            print(f"Invalid box format (Multi): {box}")
            continue
        pixel_box = [int(coord) for coord in flat_box]
        if not any(boxes_overlap(pixel_box, existing_box) for existing_box in pixel_boxes):
            pixel_boxes.append(pixel_box)
    
    print(f"Pixel boxes: {pixel_boxes}")
    
    # Draw boxes on image
    draw = ImageDraw.Draw(pil_img)
    for box in pixel_boxes:
        # Draw polygon (8 coordinates: x1, y1, x2, y2, x3, y3, x4, y4)
        draw.polygon(box, outline="red", width=2)
    
    # Save output image
    output_img_path = os.path.join(output_folder, f"{base_name}_easyocr_boxes{ext}")
    pil_img.save(output_img_path)
    
    # Save boxes as JSON
    boxes_data = [{"box": box} for box in pixel_boxes]
    output_json_path = os.path.join(output_folder, f"{base_name}_easyocr_boxes.json")
    with open(output_json_path, 'w') as f:
        json.dump({"boxes": boxes_data}, f, indent=4)
    
    print(f"Saved to {output_img_path} and {output_json_path}")

print("Processing complete.")