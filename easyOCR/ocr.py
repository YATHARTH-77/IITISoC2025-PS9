import easyocr
from PIL import Image

# Initialize separate readers due to language compatibility restrictions
reader_ko = easyocr.Reader(['ko'], gpu=True)  # Korean
reader_ru_en = easyocr.Reader(['ru', 'en'], gpu=True)  # Russian and English
reader_hi_en = easyocr.Reader(['hi', 'en'], gpu=True)  # Hindi and English
reader_es = easyocr.Reader(['es'], gpu=True)  # Spanish

# Function to run OCR with all readers and get the best result
def run_ocr_all(image_path):
    results = []
    print(f"Processing image: {image_path}")
    results.append(("ko", reader_ko.readtext(image_path)))
    results.append(("ru/en", reader_ru_en.readtext(image_path)))
    results.append(("hi/en", reader_hi_en.readtext(image_path)))
    results.append(("es", reader_es.readtext(image_path)))
    
    # Flatten and find the best result (highest confidence)
    all_results = [item for sublist in [r[1] for r in results] for item in sublist]
    best_result = max(all_results, key=lambda x: x[2] if len(x) == 3 else -1, default=None)
    return best_result

# Example usage
image_path = 'test_images/image_e2.png'  # Replace with your test image
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