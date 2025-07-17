
import cv2 as cv
import numpy as np
import os

def preprocess_image(image_path):
    img = cv.imread(image_path)
    if img is None:
        raise ValueError("Image not found.")

    h, w = img.shape[:2]
    scale = max(h, w)

    bilateral_d = 1 if scale < 600 else 13
    median_k = 1 if scale < 600 else 15
    sharpen_amount = 1.5 if scale < 600 else 2.5
    sharpen_subtract =0.5  if scale < 600 else 1.5
    kernel_size = (1, 1) 

    blu = cv.bilateralFilter(img, bilateral_d, 75, 75)
    im = cv.addWeighted(img, sharpen_amount, blu, -sharpen_subtract, 0)

    blur = cv.medianBlur(im, median_k)
    sharpened = cv.addWeighted(img, sharpen_amount, blur, -sharpen_subtract, 0)

    kernel = np.ones(kernel_size, np.uint8)
    cleaned = cv.morphologyEx(sharpened, cv.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv.morphologyEx(cleaned, cv.MORPH_CLOSE, kernel, iterations=1)

    return cleaned

def process_folder(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
           
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(input_dir, filename)
            try:
                processed = preprocess_image(path)
                out_path = os.path.join(output_dir, f"pre_{filename}")
                cv.imwrite(out_path, processed)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

def display_comparison(original, processed):
    resized_orig = cv.resize(original, (750, 750))
    resized_proc = cv.resize(processed, (750,750))
    combined = np.hstack((resized_orig, resized_proc))
    cv.imshow("Original (Left) | Processed (Right)", combined)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Example usage
#image_path = "D:\Coding\IITISoC2025-PS9\Preprocess\Datasets\English\\114.jpg"
#processed = preprocess_image(image_path)
#im=cv.imread(image_path)
#display_comparison(im,processed)
process_folder('input','preprocessed output')
#filename = os.path.basename(image_path)
#output_path = os.path.join('preprocessed output', f"pre_{filename}")
#cv.imwrite(output_path, processed)
