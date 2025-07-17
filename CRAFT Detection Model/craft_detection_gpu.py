import os
import cv2
import torch
import numpy as np
import json
from craft import CRAFT
from craft_utils import getDetBoxes
from imgproc import loadImage, resize_aspect_ratio, normalizeMeanVariance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = False  # Disable CuDNN to avoid potential errors
print(f"Using device: {device}")

def copyStateDict(state_dict):
    """Handle state dictionary keys for model loading."""
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = {}
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def detect_text(image_path, output_dir, trained_model='weights/craft_mlt_25k.pth', 
                text_threshold=0.7, link_threshold=0.4, low_text=0.4, 
                canvas_size=1024, mag_ratio=1.0, poly=False):
    """
    Detect text in an image using the CRAFT model and output results.
    
    Args:
        image_path (str): Path to the input image.
        output_dir (str): Directory to save output image and JSON.
        trained_model (str): Path to pre-trained CRAFT model weights.
        text_threshold (float): Confidence threshold for text detection.
        link_threshold (float): Confidence threshold for link detection.
        low_text (float): Low-bound score for text regions.
        canvas_size (int): Maximum image size for inference.
        mag_ratio (float): Image magnification ratio.
        poly (bool): Return polygons instead of rectangles if True.
    
    Returns:
        tuple: Scaled boxes and polygons coordinates, or None if error occurs.
    """
    print(f"Using device: {device}")
    
    # Validate image
    img_check = cv2.imread(image_path)
    if img_check is None:
        print(f"Error: Could not load image at {image_path}")
        return None, None

    # Load the CRAFT model
    net = CRAFT()
    print(f'Loading weights from {trained_model}')
    try:
        state_dict = torch.load(trained_model, map_location=device)
        net.load_state_dict(copyStateDict(state_dict))
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return None, None

    net.to(device)
    net.eval()

    # Load and preprocess the image
    try:
        image = loadImage(image_path)
        img_resized, ratio, _ = resize_aspect_ratio(image, canvas_size, cv2.INTER_LINEAR, mag_ratio)
        x = normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0)
        print(f"Input tensor shape: {x.shape}")
        x = x.to(device)
    except Exception as e:
        print(f"Error in image preprocessing: {e}")
        return None, None

    # Perform inference
    try:
        with torch.no_grad():
            y, _ = net(x)
        score_text = y[0, :, :, 0].cpu().numpy()
        score_link = y[0, :, :, 1].cpu().numpy()
    except Exception as e:
        print(f"Error during inference: {e}")
        return None, None

    # Generate detection boxes
    try:
        boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    except Exception as e:
        print(f"Error generating detection boxes: {e}")
        return None, None

    # Manually scale coordinates to original image size
    scale_factor = 2 / ratio
    scaled_boxes = []
    scaled_polys = []
    for box in boxes:
        if box is not None:
            scaled_box = [[int(x * scale_factor), int(y * scale_factor)] for x, y in box]
            scaled_boxes.append(scaled_box)
        else:
            scaled_boxes.append(None)
    for poly in polys:
        if poly is not None:
            scaled_poly = [[int(x * scale_factor), int(y * scale_factor)] for x, y in poly]
            scaled_polys.append(scaled_poly)
        else:
            scaled_polys.append(None)

    # Replace None polys with corresponding boxes
    for k in range(len(scaled_polys)):
        if scaled_polys[k] is None and scaled_boxes[k] is not None:
            scaled_polys[k] = scaled_boxes[k]

    # Draw boxes on the original image
    img = cv2.imread(image_path)
    for poly in scaled_polys:
        if poly is not None:
            poly_array = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [poly_array], True, (0, 255, 0), 2)

    # Prepare JSON output
    json_data = [{
        "source_image": os.path.basename(image_path),
        "boxes": [
            {"coordinates": box, "type": "rectangle" if not poly else "polygon"}
            for box in (scaled_boxes if not poly else scaled_polys) if box is not None
        ]
    }]

    # Save results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_image = os.path.join(output_dir, f"res_{os.path.basename(image_path)}")
    cv2.imwrite(output_image, img)
    json_path = os.path.join(output_dir, f"coords_{os.path.basename(image_path).rsplit('.', 1)[0]}.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)

    print(f"Output image saved to {output_image}")
    print(f"Coordinates saved to {json_path}")
    return scaled_boxes, scaled_polys

if __name__ == "__main__":
    image_dir = 'test_images'
    output_dir = '../easyOCR/result'
    trained_model = 'weights/craft_mlt_25k.pth'
    poly = False

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, filename)
            print(f"Processing {filename}...")
            detect_text(image_path, output_dir, trained_model, poly=poly)