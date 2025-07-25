# craft_service_app.py

from flask import Flask, request, jsonify
from craft_detection import detect_text
from preprocessing import preprocess_image
import os
import cv2
import json

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR  = os.path.join(BASE_DIR, '..', 'easyOCR', 'static')
os.makedirs(TMP_DIR, exist_ok=True)

@app.route('/detect', methods=['POST','GET'])
def detect():
    img_file = request.files.get('image')
    if not img_file:
        return jsonify({'error': 'No image provided 111'}), 400

    # 1. Save raw upload
    raw_path = os.path.join(TMP_DIR, "preprocess.png")
    img_file.save(raw_path)
    print(f"[CRAFT] Saved raw image to: {raw_path}  Exists? {os.path.exists(raw_path)}")

    # 2. Preprocess
    pre_img = preprocess_image(raw_path)  # returns cv2 image
    preproc_path = os.path.join(TMP_DIR, "craft.png")
    success = cv2.imwrite(preproc_path, pre_img)
    print(f"[CRAFT] Saved preprocessed to: {preproc_path}  Success? {success}")

    # 3. Detect boxes (no polygons)
    _, boxes = detect_text(
        preproc_path,
        output_dir=TMP_DIR,
        cuda=False,
        poly=False
    )

    # 4. Save coordinates
    coords_path = os.path.join(TMP_DIR, "craft.json")
    with open(coords_path, 'w') as f:
        json.dump(boxes, f, indent=2)

    print(f"[CRAFT] Returning {len(boxes)} boxes")
    return jsonify(boxes)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6000)
