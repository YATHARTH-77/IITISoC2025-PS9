import json
import sys
from typing import List, Dict, Any

def get_text_position(box: List[List[int]]) -> tuple:
    """
    Extract the top-left position from a bounding box.
    Returns (y, x) for sorting - y first for top-to-bottom, then x for left-to-right.
    """
    # Box format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    # Get the top-left corner (minimum y, then minimum x)
    top_y = min(point[1] for point in box)
    left_x = min(point[0] for point in box)
    return (top_y, left_x)

def group_by_lines(text_items: List[Dict], line_threshold: int = 15) -> List[List[Dict]]:
    """
    Group text items that are on the same line based on their y-coordinates.
    Items with y-coordinates within line_threshold pixels are considered on the same line.
    """
    if not text_items:
        return []
    
    # Sort by y-coordinate first
    sorted_items = sorted(text_items, key=lambda item: get_text_position(item['box'])[0])
    
    lines = []
    current_line = [sorted_items[0]]
    current_y = get_text_position(sorted_items[0]['box'])[0]
    
    for item in sorted_items[1:]:
        item_y = get_text_position(item['box'])[0]
        
        # If the y-coordinate is within threshold, add to current line
        if abs(item_y - current_y) <= line_threshold:
            current_line.append(item)
        else:
            # Start a new line
            lines.append(current_line)
            current_line = [item]
            current_y = item_y
    
    # Don't forget the last line
    if current_line:
        lines.append(current_line)
    
    return lines

def sort_line_items(line_items: List[Dict]) -> List[Dict]:
    """
    Sort items within a line from left to right based on x-coordinates.
    """
    return sorted(line_items, key=lambda item: get_text_position(item['box'])[1])

def reorder_ocr_data(input_file: str, output_file: str, line_threshold: int = 15):
    """
    Read OCR JSON data, reorder it in natural reading order, and save to output file.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        line_threshold: Maximum y-coordinate difference to consider items on same line
    """
    try:
        # Read input JSON
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("Expected JSON to contain a list of text items")
        
        print(f"Processing {len(data)} text items...")
        
        # Group items by lines
        lines = group_by_lines(data, line_threshold)
        print(f"Found {len(lines)} lines of text")
        
        # Sort items within each line and flatten
        reordered_data = []
        for i, line in enumerate(lines):
            sorted_line = sort_line_items(line)
            reordered_data.extend(sorted_line)
            
            # Debug: print line content
            line_text = ' '.join(item['detected_text'] for item in sorted_line)
            print(f"Line {i+1}: {line_text}")
        
        # Save reordered data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(reordered_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nReordered data saved to: {output_file}")
        
        # Print the reconstructed text
        full_text = ' '.join(item['detected_text'] for item in reordered_data)
        print(f"Reconstructed text: {full_text}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file '{input_file}'.")
    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python ocr_reorder.py <input_json_file> <output_json_file> [line_threshold]")
        print("Example: python ocr_reorder.py data.json reordered_data.json 30")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    line_threshold = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    
    reorder_ocr_data(input_file, output_file, line_threshold)

if __name__ == "__main__":
    main()