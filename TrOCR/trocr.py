#!/usr/bin/env python3
"""
TrOCR Handwritten Text Recognition with Word Coordinates
Uses Microsoft's TrOCR model + text detection for coordinate extraction
Outputs JSON in EasyOCR format with word bounding boxes and recognized text
"""

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageDraw
import cv2
import numpy as np
import argparse
import os
import sys
import json
from datetime import datetime
import easyocr
import time

class TrOCRWithCoordinates:
    def __init__(self, model_name="microsoft/trocr-large-handwritten"):
        """
        Initialize the TrOCR model and text detector
        
        Args:
            model_name (str): TrOCR model name
        """
        print(f"Loading TrOCR model: {model_name}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load TrOCR processor and model
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        
        # Initialize EasyOCR for text detection (coordinates)
        print("Loading text detector...")
        self.detector = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        
        print("Models loaded successfully!")
    
    def detect_text_regions(self, image_path):
        """
        Detect text regions and get bounding boxes using EasyOCR
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            list: List of detected text regions with coordinates
        """
        try:
            # Use EasyOCR to detect text regions
            results = self.detector.readtext(image_path, paragraph=False, width_ths=0.7)
            
            text_regions = []
            for i, (bbox, text, confidence) in enumerate(results):
                # Convert bbox to the exact format needed
                bbox_array = np.array(bbox).astype(int)
                
                text_regions.append({
                    'id': i,
                    'bbox': bbox_array.tolist(),  # Keep original polygon format
                    'confidence': float(confidence),
                    'detected_text': text  # EasyOCR's detection for comparison
                })
            
            return text_regions
        except Exception as e:
            print(f"Error detecting text regions: {e}")
            return []
    
    def crop_text_region(self, image, bbox, padding=5):
        """
        Crop text region from image with padding
        
        Args:
            image (PIL.Image): Source image
            bbox (list): Bounding box as polygon points
            padding (int): Padding around the text region
            
        Returns:
            PIL.Image: Cropped image region
        """
        # Convert polygon to bounding rectangle
        bbox_array = np.array(bbox)
        x_min = int(min(bbox_array[:, 0]))
        y_min = int(min(bbox_array[:, 1]))
        x_max = int(max(bbox_array[:, 0]))
        y_max = int(max(bbox_array[:, 1]))
        
        # Add padding
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(image.width, x_max + padding)
        y_max = min(image.height, y_max + padding)
        
        return image.crop((x_min, y_min, x_max, y_max))
    
    def recognize_text_region(self, image_region, max_length=256):
        """
        Recognize text from a cropped image region using TrOCR
        
        Args:
            image_region (PIL.Image): Cropped image region
            max_length (int): Maximum length of generated text
            
        Returns:
            str: Recognized text
        """
        try:
            # Process image region with TrOCR
            pixel_values = self.processor(image_region, return_tensors="pt").pixel_values.to(self.device)
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values, 
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode the generated text
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return generated_text.strip()
            
        except Exception as e:
            print(f"Error recognizing text region: {e}")
            return ""
    
    def split_into_words(self, text, bbox):
        """
        Split recognized text into individual words with estimated coordinates
        
        Args:
            text (str): Recognized text
            bbox (list): Bounding box as polygon points
            
        Returns:
            list: List of word dictionaries with coordinates in EasyOCR format
        """
        words = text.split()
        if not words:
            return []
        
        # Convert polygon to bounding rectangle
        bbox_array = np.array(bbox)
        x_min = min(bbox_array[:, 0])
        y_min = min(bbox_array[:, 1])
        x_max = max(bbox_array[:, 0])
        y_max = max(bbox_array[:, 1])
        
        region_width = x_max - x_min
        
        word_list = []
        total_chars = sum(len(word) for word in words) + len(words) - 1  # Include spaces
        
        current_x = x_min
        for i, word in enumerate(words):
            # Estimate word width based on character count
            word_width = (len(word) / total_chars) * region_width
            
            # Create bounding box in EasyOCR format (4 corner points)
            word_bbox = [
                [int(current_x), int(y_min)],
                [int(current_x + word_width), int(y_min)],
                [int(current_x + word_width), int(y_max)],
                [int(current_x), int(y_max)]
            ]
            
            word_dict = {
                "box": word_bbox,
                "detected_text": word,
                "confidence": 1.0,  # TrOCR doesn't provide word-level confidence
                "language": "en",
                "is_handwritten": True,  # Since we're using TrOCR for handwriting
                "spell_checked": False,
                "original_text": word
            }
            word_list.append(word_dict)
            
            # Move to next word position (add space width)
            space_width = (1 / total_chars) * region_width if i < len(words) - 1 else 0
            current_x += word_width + space_width
        
        return word_list
    
    def process_image(self, image_path, max_length=256):
        """
        Process entire image and return JSON in EasyOCR format
        
        Args:
            image_path (str): Path to the image file
            max_length (int): Maximum length for text recognition
            
        Returns:
            dict: JSON structure in EasyOCR format with detected words and coordinates
        """
        start_time = time.time()
        
        try:
            # Load image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Detect text regions
            print("Detecting text regions...")
            text_regions = self.detect_text_regions(image_path)
            
            if not text_regions:
                print("No text regions detected")
                return self.create_empty_result(image_path, start_time)
            
            print(f"Found {len(text_regions)} text regions")
            
            # Process each text region with TrOCR
            all_results = []
            for i, region in enumerate(text_regions):
                print(f"Processing region {i+1}/{len(text_regions)}...")
                
                # Crop the text region
                cropped_image = self.crop_text_region(image, region['bbox'])
                
                # Recognize text with TrOCR
                trocr_text = self.recognize_text_region(cropped_image, max_length)
                
                if trocr_text:
                    # Split into words with coordinates
                    words = self.split_into_words(trocr_text, region['bbox'])
                    all_results.extend(words)
                else:
                    # If TrOCR failed, use EasyOCR result
                    word_dict = {
                        "box": region['bbox'],
                        "detected_text": region['detected_text'],
                        "confidence": region['confidence'],
                        "language": "en",
                        "is_handwritten": True,
                        "spell_checked": False,
                        "original_text": region['detected_text']
                    }
                    all_results.append(word_dict)
            
            processing_time = time.time() - start_time
            
            # Create final JSON structure in EasyOCR format
            result = {
                "language": "en",
                "is_handwritten": True,
                "spell_check_enabled": False,
                "processing_time": round(processing_time, 1),
                "total_detections": len(all_results),
                "fast_processing": True,
                "results": all_results
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return self.create_empty_result(image_path, start_time, str(e))
    
    def create_empty_result(self, image_path, start_time, error=None):
        """Create empty result structure"""
        processing_time = time.time() - start_time
        
        result = {
            "language": "en",
            "is_handwritten": True,
            "spell_check_enabled": False,
            "processing_time": round(processing_time, 1),
            "total_detections": 0,
            "fast_processing": True,
            "results": []
        }
        
        if error:
            result["error"] = error
            
        return result
    
    def batch_process(self, input_folder, output_folder):
        """
        Process multiple images and save JSON results
        
        Args:
            input_folder (str): Path to folder containing images
            output_folder (str): Path to save JSON results
        """
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        if not os.path.exists(input_folder):
            print(f"Error: Input folder '{input_folder}' does not exist")
            return
        
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all image files
        image_files = [f for f in os.listdir(input_folder) 
                      if os.path.splitext(f.lower())[1] in supported_formats]
        
        if not image_files:
            print(f"No supported image files found in '{input_folder}'")
            return
        
        print(f"Processing {len(image_files)} images...")
        results = []
        
        for i, filename in enumerate(image_files, 1):
            image_path = os.path.join(input_folder, filename)
            print(f"\n[{i}/{len(image_files)}] Processing: {filename}")
            
            try:
                result = self.process_image(image_path)
                results.append({
                    "filename": filename,
                    "data": result
                })
                
                # Save individual JSON file
                output_filename = os.path.splitext(filename)[0] + '.json'
                output_path = os.path.join(output_folder, output_filename)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)
                
                print(f"  Words detected: {result['total_detections']}")
                print(f"  Processing time: {result['processing_time']}s")
                print(f"  Saved: {output_path}")
                
            except Exception as e:
                print(f"  Error processing {filename}: {e}")
        
        # Save combined results
        combined_output = os.path.join(output_folder, 'combined_results.json')
        with open(combined_output, 'w', encoding='utf-8') as f:
            json.dump({
                'batch_info': {
                    'total_images': len(image_files),
                    'processed_successfully': len(results),
                    'processed_at': datetime.now().isoformat()
                },
                'results': results
            }, f, indent=4, ensure_ascii=False)
        
        print(f"\nBatch processing complete!")
        print(f"Individual results saved in: {output_folder}")
        print(f"Combined results saved: {combined_output}")

def main():
    parser = argparse.ArgumentParser(description="TrOCR with EasyOCR Format JSON Output")
    parser.add_argument("input", help="Path to image file or folder containing images")
    parser.add_argument("-o", "--output", required=True, 
                       help="Output JSON file (for single image) or folder (for batch)")
    parser.add_argument("-m", "--model", 
                       choices=["base", "large"], 
                       default="large",
                       help="TrOCR model size (default: large)")
    parser.add_argument("--max-length", type=int, default=256,
                       help="Maximum length of generated text (default: 256)")
    
    args = parser.parse_args()
    
    # Set model name
    model_names = {
        "base": "microsoft/trocr-base-handwritten",
        "large": "microsoft/trocr-large-handwritten"
    }
    model_name = model_names[args.model]
    
    try:
        # Initialize processor
        processor = TrOCRWithCoordinates(model_name)
        
        if os.path.isfile(args.input):
            # Single image processing
            print(f"\nProcessing image: {args.input}")
            result = processor.process_image(args.input, args.max_length)
            
            # Save JSON result
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            
            print(f"\nResults saved to: {args.output}")
            print(f"Words detected: {result['total_detections']}")
            print(f"Processing time: {result['processing_time']}s")
            
            # Print sample results
            if result['results']:
                print(f"Sample detections:")
                for i, detection in enumerate(result['results'][:3]):
                    print(f"  {i+1}. '{detection['detected_text']}' (confidence: {detection['confidence']:.3f})")
            
        elif os.path.isdir(args.input):
            # Batch processing
            print(f"\nBatch processing folder: {args.input}")
            processor.batch_process(args.input, args.output)
            
        else:
            print(f"Error: '{args.input}' is neither a valid file nor directory")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("TrOCR with EasyOCR Format JSON Output")
        print("\nInstallation:")
        print("  pip install torch transformers pillow opencv-python easyocr")
        print("\nUsage examples:")
        print("  python recog.py image.jpg -o result.json")
        print("  python recog.py /path/to/images/ -o /path/to/output/")
        print("  python recog.py image.jpg -o result.json --model base")
        print("\nFor help: python recog.py -h")
    else:
        main()