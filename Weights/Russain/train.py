import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import lmdb
import pickle
from PIL import Image
import json
from tqdm import tqdm
import torch.nn.functional as F

class CustomLMDBDataset(Dataset):
    """Custom LMDB Dataset for our training data"""
    
    def __init__(self, lmdb_path, character_list, max_len=25):
        self.lmdb_path = lmdb_path
        self.character_list = character_list
        self.max_len = max_len
        
        # Create character mappings
        self.char_to_idx = {char: idx+1 for idx, char in enumerate(character_list)}  # +1 for blank
        self.char_to_idx['<blank>'] = 0
        self.char_to_idx['<UNK>'] = 0  # Unknown character
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        # Open LMDB (deferred to __getitem__ to avoid pickling)
        self.env = None
        with lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False) as env:
            with env.begin() as txn:
                info = pickle.loads(txn.get('info'.encode()))
                self.num_samples = info['num_samples']
        
        print(f"Dataset loaded: {self.num_samples} samples from {lmdb_path}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        
        with self.env.begin() as txn:
            # Get image
            img_key = f'image-{idx:09d}'.encode()
            img_bytes = txn.get(img_key)
            
            if img_bytes is None:
                return torch.zeros(3, 32, 100), torch.zeros(self.max_len, dtype=torch.long), 0
            
            # Decode image
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                return torch.zeros(3, 32, 100), torch.zeros(self.max_len, dtype=torch.long), 0
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get label
            label_key = f'label-{idx:09d}'.encode()
            label_bytes = txn.get(label_key)
            
            if label_bytes is None:
                return torch.zeros(3, 32, 100), torch.zeros(self.max_len, dtype=torch.long), 0
            
            label = label_bytes.decode('utf-8')
            
            # Process image
            img_tensor = self.process_image(img)
            
            # Process label
            label_tensor, label_length = self.process_label(label)
            
            return img_tensor, label_tensor, label_length
    
    def process_image(self, img):
        """Process image for training"""
        # Resize to standard height
        h, w = img.shape[:2]
        target_h = 32
        target_w = int(w * (target_h / h))
        
        # Limit max width
        max_w = 200
        if target_w > max_w:
            target_w = max_w
        
        img_resized = cv2.resize(img, (target_w, target_h))
        
        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Convert to tensor (C, H, W)
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1)
        
        return img_tensor
    
    def process_label(self, label):
        """Convert label to tensor"""
        # Convert characters to indices
        indices = []
        for char in label:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                indices.append(self.char_to_idx['<UNK>'])
        
        # Pad or truncate
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        
        label_tensor = torch.tensor(indices, dtype=torch.long)
        label_length = len(indices)
        
        return label_tensor, label_length

def collate_fn(batch):
    """Custom collate function for variable width images"""
    images, labels, lengths = zip(*batch)
    
    # Find max width in batch
    max_width = max([img.shape[2] for img in images])
    
    # Pad images to same width
    padded_images = []
    for img in images:
        c, h, w = img.shape
        if w < max_width:
            padding = torch.zeros(c, h, max_width - w)
            padded_img = torch.cat([img, padding], dim=2)
        else:
            padded_img = img
        padded_images.append(padded_img)
    
    # Stack tensors
    image_batch = torch.stack(padded_images)
    
    # Pad labels to same length
    max_label_len = max([len(label) for label in labels])
    padded_labels = []
    for label in labels:
        if len(label) < max_label_len:
            padding = torch.zeros(max_label_len - len(label), dtype=torch.long)
            padded_label = torch.cat([label, padding])
        else:
            padded_label = label
        padded_labels.append(padded_label)
    
    label_batch = torch.stack(padded_labels)
    length_batch = torch.tensor(lengths, dtype=torch.long)
    
    return image_batch, label_batch, length_batch

class SimpleCRNN(nn.Module):
    """Simplified CRNN model for OCR"""
    
    def __init__(self, num_classes, img_h=32, hidden_size=256):
        super(SimpleCRNN, self).__init__()
        
        self.img_h = img_h
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, kernel_size=2, padding=0),
        )
        
        # RNN layers
        self.rnn = nn.LSTM(512, hidden_size, bidirectional=True, batch_first=True)
        
        # Output layer
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        conv_features = self.cnn(x)
        b, c, h, w = conv_features.size()
        conv_features = conv_features.view(b, c, w).permute(0, 2, 1)
        rnn_output, _ = self.rnn(conv_features)
        output = self.classifier(rnn_output)
        output = output.permute(1, 0, 2)
        output = F.log_softmax(output, dim=2)
        return output

class EasyOCRTrainer:
    """Training class for EasyOCR fine-tuning"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load character list
        self.character_list = self.load_character_list()
        self.num_classes = len(self.character_list) + 1  # +1 for CTC blank
        
        print(f"Number of classes: {self.num_classes}")
        print(f"Characters: {len(self.character_list)}")
    
    def load_character_list(self):
        with open(self.config['character_list_path'], 'r', encoding='utf-8') as f:
            chars = [line.strip() for line in f.readlines()]
        return chars
    
    def create_data_loaders(self):
        train_dataset = CustomLMDBDataset(
            self.config['train_lmdb_path'], 
            self.character_list
        )
        val_dataset = CustomLMDBDataset(
            self.config['val_lmdb_path'], 
            self.character_list
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid pickling issues on Windows
            collate_fn=collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid pickling issues on Windows
            collate_fn=collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        return train_loader, val_loader
    
    def create_model(self):
        model = SimpleCRNN(self.num_classes)
        model.to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model created:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def train_epoch(self, model, train_loader, optimizer, criterion):
        model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (images, labels, lengths) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            lengths = lengths.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(images)
            output_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long)
            loss = criterion(outputs, labels, output_lengths, lengths)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def validate_epoch(self, model, val_loader, criterion):
        model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for images, labels, lengths in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)
                
                outputs = model(images)
                output_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long)
                loss = criterion(outputs, labels, output_lengths, lengths)
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def train(self):
        print("Starting EasyOCR fine-tuning...")
        
        model = self.create_model()
        train_loader, val_loader = self.create_data_loaders()
        
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5, verbose=True
        )
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        print(f"Training for {self.config['num_epochs']} epochs...")
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            print("-" * 50)
            
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            val_loss = self.validate_epoch(model, val_loader, criterion)
            
            scheduler.step(val_loss)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'character_list': self.character_list,
                    'config': self.config
                }, 'best_italian_ocr_model.pth')
                print(f"🎉 Best model saved! Val Loss: {val_loss:.4f}")
            
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, f'checkpoint_epoch_{epoch+1}.pth')
        
        print("\n" + "="*50)
        print("Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print("Model saved as: best_italian_ocr_model.pth")
        
        return train_losses, val_losses

def main():
    config = {
        'character_list_path': 'training_data/character_list.txt',
        'train_lmdb_path': 'training_data/lmdb/train',
        'val_lmdb_path': 'training_data/lmdb/val',
        'batch_size': 16,
        'learning_rate': 0.001,
        'num_epochs': 50,
        'max_image_width': 200,
        'image_height': 32
    }
    
    print("=== EasyOCR italian Fine-tuning ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    required_files = [
        config['character_list_path'],
        config['train_lmdb_path'],
        config['val_lmdb_path']
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Error: Required file/directory not found: {file_path}")
            return
        print(f"✅ Found: {file_path}")
    
    with open('training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nConfiguration saved to: training_config.json")
    
    trainer = EasyOCRTrainer(config)
    train_losses, val_losses = trainer.train()
    
    print("\n🎉 Training pipeline completed successfully!")

if __name__ == "__main__":
    main()