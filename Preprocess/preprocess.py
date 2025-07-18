
import cv2 as cv
import numpy as np
import os
def preprocess_image(image_path):
    img = cv.imread(image_path)
    if img is None:
        raise ValueError("Image not found.")

    h, w = img.shape[:2]
    scale = max(h, w)

    # Scaling logic
    median_k = 3 if scale < 600 else 15
    sharpen_amount = 1.7 if scale < 600 else 2.5
    sharpen_subtract = 0.7 if scale < 600 else 1.5
    morph_kernel_size = (1, 1)
    bet= 200 if scale < 600 else 215

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 2. Remove background noise 
    blur = cv.medianBlur(gray, median_k)

    # 3. Contrast enhancement (visibly darkens text, brightens background)
    norm= cv.normalize(blur, None, alpha=0, beta=bet, norm_type=cv.NORM_MINMAX)

    # 4. Sharpen text edges
    sharpened = cv.addWeighted(gray, sharpen_amount, norm, -sharpen_subtract, 0)

    
    kernel = np.ones(morph_kernel_size, np.uint8)
    morph = cv.morphologyEx(sharpened, cv.MORPH_OPEN, kernel, iterations=1)
    morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, kernel, iterations=1)


    if scale>600 : 
        processed = cv.adaptiveThreshold(morph, 185, cv.ADAPTIVE_THRESH_MEAN_C,
                                     cv.THRESH_BINARY, 23, 27)
    else:
        processed = cv.adaptiveThreshold(morph, 205, cv.ADAPTIVE_THRESH_MEAN_C,
                                     cv.THRESH_BINARY, 7, 7)
    
    return processed
def display_comparison(original, processed):
    resized_orig = cv.resize(original, (750, 750))
    resized_proc = cv.resize(processed, (750, 750))
    combined = np.hstack((resized_orig, cv.cvtColor(resized_proc, cv.COLOR_GRAY2BGR)))
    cv.imshow("Original (Left) | Processed (Right)", combined)
    cv.waitKey(0)
    cv.destroyAllWindows()
def process_folder(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
           
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(input_dir, filename)
            try:
                processed = preprocess_image(path)
                out_path = os.path.join(output_dir, filename)
                cv.imwrite(out_path, processed)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
#image_path = ''
#processed = preprocess_image(image_path)
#im = cv.imread(image_path)
#display_comparison(im, processed)
#fill output folder before \image path
#output_path=f'\{image_path}'
#saving the image by specifying image path and output path above 
#cv.imwrite(output_path,processed)
process_folder('Preprocess\input','CRAFT Detection Model\\test_images')