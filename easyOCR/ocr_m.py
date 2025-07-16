import easyocr
from PIL import Image

# Initialize separate readers due to language compatibility restrictions
reader_ko_en = easyocr.Reader(['ko', 'en'], gpu=True)  # Korean with English only
reader_hi_en = easyocr.Reader(['hi', 'en'], gpu=True)  # Hindi with English only
reader_ru_en = easyocr.Reader(['ru', 'en'], gpu=True)  # Russian with English only
reader_multi = easyocr.Reader(['es', 'fr', 'de', 'it','tr'], gpu=True)  # Other compatible languages

# Function to run OCR with all readers and get the best result
def run_ocr_all(image_path):
    results = []
    print(f"Processing image: {image_path}")
    
    # Process with Korean + English reader
    ko_en_results = reader_ko_en.readtext(image_path)
    results.extend(ko_en_results)
    
    # Process with Hindi + English reader
    hi_en_results = reader_hi_en.readtext(image_path)
    results.extend(hi_en_results)
    
    # Process with Russian + English reader
    ru_en_results = reader_ru_en.readtext(image_path)
    results.extend(ru_en_results)
    
    # Process with multi-language reader
    multi_results = reader_multi.readtext(image_path)
    results.extend(multi_results)
    
    # Find the best result (highest confidence)
    best_result = max(results, key=lambda x: x[2] if len(x) == 3 else -1, default=None)
    return best_result

# Example usage
image_path = '../paddleOCR/test_images/e_h3.png'  # Replace with your test image
image = Image.open(image_path)
image.show()  # View the image for verification

# Run OCR and get the best result
best_result = run_ocr_all(image_path)

# Display only the best detected text
if best_result and len(best_result) == 3:
    bbox, text, prob = best_result
    print(f"Detected Text: {text}")
else:
    print("No text detected or incomplete data")