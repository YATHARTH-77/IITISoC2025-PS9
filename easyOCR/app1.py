import os
import json
import requests
from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import easyocr

app = Flask(__name__)

# --- Paths ---
CRAFT_URL = 'http://localhost:6000/detect'
BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
STATIC_DIR = os.path.join(BASE_DIR, 'static') 

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# --- EasyOCR caching ---
readers = {}
def run_easyocr(image_np, language='en'):
    if language not in readers:
        readers[language] = easyocr.Reader([language], gpu=True)
    result = readers[language].readtext(image_np)
    if not result:
        return "", 0.0
    best = max(result, key=lambda x: x[2])
    return best[1], float(best[2])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    file = request.files.get('file')
    lang = request.form.get('language', 'en')
    if not file:
        return "[Error] No file provided", 400

    filename = file.filename
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)

    # --- Step 1: Get boxes from CRAFT ---
    with open(save_path, 'rb') as f:
        resp = requests.post(CRAFT_URL, files={'image': f})
    if resp.status_code != 200:
        return "[Error] CRAFT failed", 500

    boxes = resp.json()

    # --- Step 2: OCR on each box ---
    img = Image.open(save_path).convert('RGB')
    results = []

    for idx, box in enumerate(boxes):
        try:
            if isinstance(box[0], list) and len(box) > 2:
                xs = [pt[0] for pt in box]
                ys = [pt[1] for pt in box]
                x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
            elif isinstance(box[0], list):
                (x0, y0), (x1, y1) = box
                x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
            else:
                x0, y0, x1, y1 = map(int, box)

            if x1 <= x0 or y1 <= y0:
                continue

            cropped = img.crop((x0, y0, x1, y1))
            cropped_np = np.array(cropped)
            text, conf = run_easyocr(cropped_np, language=lang)

            results.append({
                'box': box,
                'text': text,
                'confidence': conf
            })

        except Exception as e:
            print(f"[Merged] Skipped box {idx} due to error: {e}")
            continue

    # --- Step 3: Save result as data.json in easyocr/static ---
    output_path = os.path.join(STATIC_DIR, 'data.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"[Saved] OCR result saved as: {output_path}")
    return '',204
@app.route('/output', methods=['GET'])
def result_get():
 return render_template('Result.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
