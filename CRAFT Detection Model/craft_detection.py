import torch
import cv2
import numpy as np
from craft import CRAFT
from craft_utils import getDetBoxes
from imgproc import resize_aspect_ratio, normalizeMeanVariance

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = {}
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def load_craft_model(model_path='weights/craft_mlt_25k.pth'):
    """
    Loads the CRAFT model into memory and returns it.
    """
    net = CRAFT()
    print(f'Loading weights from {model_path}')
    state_dict = torch.load(model_path, map_location='cpu')
    net.load_state_dict(copyStateDict(state_dict))
    net.eval()
    print("CRAFT model loaded successfully.")
    return net

def get_text_boxes(net, image, text_threshold=0.7, link_threshold=0.4, low_text=0.4, 
                   canvas_size=1280, mag_ratio=1.5, poly=False):
    """
    Detects text in a preprocessed image using the loaded CRAFT model.
    """
    # --- THE DEFINITIVE FIX ---
    # The 'image' we receive from the preprocessor is in BGR format.
    # We convert it to RGB right before the model's internal processing functions need it.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    img_resized, ratio, _ = resize_aspect_ratio(image, canvas_size, cv2.INTER_LINEAR, mag_ratio)
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0)

    with torch.no_grad():
        y, _ = net(x)
    
    score_text = y[0, :, :, 0].cpu().numpy()
    score_link = y[0, :, :, 1].cpu().numpy()

    boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    
    scale_factor = 2 / ratio
    scaled_polys = []
    for poly_coords in polys:
        if poly_coords is not None:
            scaled_poly = [[int(x * scale_factor), int(y * scale_factor)] for x, y in poly_coords]
            scaled_polys.append(scaled_poly)

    return scaled_polys
