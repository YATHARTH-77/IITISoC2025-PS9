# optimized_data_prep.py
import os
import cv2
import numpy as np
import lmdb
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import string
import gc

class OptimizedDatasetPrep:
    def __init__(self, batch_size=15000):
        self.batch_size = batch_size
        
    def load_dataset(self, labels_file, images_dir):
        """Load dataset with validation"""
        print("Loading dataset...")
        data_pairs = []
        character_set = set()
        
        with open(labels_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_name, text = parts
                    img_path = os.path.join(images_dir, img_name)
                    
                    if os.path.exists(img_path):
                        # Quick validation - check if image can be loaded
                        try:
                            img = cv2.imread(img_path)
                            if img is not None:
                                data_pairs.append((img_path, text))
                                character_set.update(text)
                            else:
                                print(f"Warning: Could not load image {img_path}")
                        except Exception as e:
                            print(f"Error with image {img_path}: {e}")
                    else:
                        print(f"Missing image: {img_path}")
                else:
                    print(f"Invalid line {line_num}: {line.strip()}")
        
        print(f"Successfully loaded {len(data_pairs)} valid image-text pairs")
        print(f"Character set size: {len(character_set)}")
        return data_pairs, character_set
    
    def create_lmdb_batch(self, data_pairs, output_path, batch_size=1000):
        """Create LMDB with batch processing"""
        print(f"Creating LMDB at {output_path} with batch size {batch_size}")
        
        # Remove existing LMDB if it exists
        if os.path.exists(output_path):
            import shutil
            shutil.rmtree(output_path)
        
        # Calculate map size (estimate 2MB per image)
        map_size = len(data_pairs) * 2 * 1024 * 1024  # 2MB per image
        map_size = max(map_size, 1024 * 1024 * 1024)  # Minimum 1GB
        
        env = lmdb.open(output_path, map_size=map_size)
        
        try:
            # Process in batches
            for start_idx in range(0, len(data_pairs), batch_size):
                end_idx = min(start_idx + batch_size, len(data_pairs))
                batch = data_pairs[start_idx:end_idx]
                
                print(f"Processing batch {start_idx//batch_size + 1}/{(len(data_pairs)-1)//batch_size + 1}")
                
                with env.begin(write=True) as txn:
                    for i, (img_path, label) in enumerate(tqdm(batch, desc="Batch progress")):
                        try:
                            # Load and resize image
                            img = cv2.imread(img_path)
                            if img is None:
                                continue
                            
                            # Standard height for OCR
                            h, w = img.shape[:2]
                            new_h = 32
                            new_w = int(w * (new_h / h))
                            
                            # Limit maximum width
                            if new_w > 200:
                                new_w = 200
                            
                            img_resized = cv2.resize(img, (new_w, new_h))
                            
                            # Encode image
                            _, img_encoded = cv2.imencode('.jpg', img_resized, 
                                                        [cv2.IMWRITE_JPEG_QUALITY, 90])
                            img_bytes = img_encoded.tobytes()
                            
                            # Store in LMDB
                            global_idx = start_idx + i
                            img_key = f'image-{global_idx:09d}'.encode()
                            label_key = f'label-{global_idx:09d}'.encode()
                            
                            txn.put(img_key, img_bytes)
                            txn.put(label_key, label.encode('utf-8'))
                            
                        except Exception as e:
                            print(f"Error processing {img_path}: {e}")
                            continue
                
                # Force garbage collection after each batch
                gc.collect()
            
            # Store dataset info
            with env.begin(write=True) as txn:
                info = {
                    'num_samples': len(data_pairs),
                    'image_shape': [32, new_w, 3]
                }
                txn.put('info'.encode(), pickle.dumps(info))
        
        finally:
            env.close()
        
        print(f"LMDB creation completed: {len(data_pairs)} samples")

def main():
    # Paths
    labels_file = "es/image_list.txt"
    images_dir = "es/"
    output_dir = "training_data"
    
    # Create output directory
    os.makedirs(f"{output_dir}/lmdb", exist_ok=True)
    
    # Initialize processor
    processor = OptimizedDatasetPrep(batch_size=250)  # Smaller batches
    
    # Load dataset
    data_pairs, character_set = processor.load_dataset(labels_file, images_dir)
    
    if len(data_pairs) == 0:
        print("No valid data pairs found! Check your images directory and labels.txt")
        return
    
    # Create character list
    chars = set(string.ascii_letters + string.digits + string.punctuation + ' ')
    chars.update(character_set)
    
    with open(f"{output_dir}/character_list.txt", 'w', encoding='utf-8') as f:
        for char in sorted(chars):
            f.write(char + '\n')
    
    print(f"Character list saved with {len(chars)} characters")
    
    # Split dataset
    train_data, temp_data = train_test_split(data_pairs, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    print(f"Dataset split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Create LMDB datasets
    print("\nCreating training LMDB...")
    processor.create_lmdb_batch(train_data, f"{output_dir}/lmdb/train")
    
    print("\nCreating validation LMDB...")
    processor.create_lmdb_batch(val_data, f"{output_dir}/lmdb/val")
    
    print("\nCreating test LMDB...")
    processor.create_lmdb_batch(test_data, f"{output_dir}/lmdb/test")
    
    print("\nData preparation completed successfully!")
    print(f"Files created in: {output_dir}/")

if __name__ == "__main__":
    main()