import cv2
from paddleocr import PaddleOCR
from PIL import Image

# Initialize separate readers based on PaddleOCR model groups with CPU mode
reader_ko = PaddleOCR(lang='korean', use_angle_cls=True, use_gpu=False)  # Korean
reader_hi = PaddleOCR(lang='devanagari', use_angle_cls=True, use_gpu=False)  # Hindi
reader_ru = PaddleOCR(lang='cyrillic', use_angle_cls=True, use_gpu=False)  # Russian
reader_multi = PaddleOCR(lang='latin', use_angle_cls=True, use_gpu=False)  # Spanish, French, German, Italian, Turkish

# Function to run OCR with all readers and get the best result
def run_ocr_all(image_path):
    results = []
    print(f"Processing image: {image_path}")
    
    # Process with Korean reader
    ko_results = reader_ko.ocr(image_path, cls=True)
    if ko_results and len(ko_results) > 0 and ko_results[0]:
        results.extend([item[1] for item in ko_results[0]])
    else:
        print("No results from Korean reader")

    # Process with Hindi reader
    hi_results = reader_hi.ocr(image_path, cls=True)
    if hi_results and len(hi_results) > 0 and hi_results[0]:
        results.extend([item[1] for item in hi_results[0]])
    else:
        print("No results from Hindi reader")

    # Process with Russian reader
    ru_results = reader_ru.ocr(image_path, cls=True)
    if ru_results and len(ru_results) > 0 and ru_results[0]:
        results.extend([item[1] for item in ru_results[0]])
    else:
        print("No results from Russian reader")

    # Process with multi-language (Latin) reader
    multi_results = reader_multi.ocr(image_path, cls=True)
    if multi_results and len(multi_results) > 0 and multi_results[0]:
        results.extend([item[1] for item in multi_results[0]])
    else:
        print("No results from Latin reader")

    # Debug: Print all results for inspection
    print("All results:", results)
    
    # Find the best result (highest confidence)
    best_result = max(results, key=lambda x: x[1] if len(x) == 2 else -1, default=None)
    return best_result

# Example usage
image_path = 'test_images/image.png'  # Replace with your test image
image = Image.open(image_path)
image.show()  # View the image for verification

# Run OCR and get the best result
best_result = run_ocr_all(image_path)

# Display only the best detected text
if best_result and len(best_result) == 2:
    text, prob = best_result
    print(f"Detected Text: {text}")
else:
    print("No text detected or incomplete data")