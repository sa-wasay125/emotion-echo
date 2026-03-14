"""
Complete Face Emotion Training Script
Run from ml/ directory: python training/train_face.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import sys
import os
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.face_preprocess import FacePreprocessor
from models.face_model.face_model import FaceEmotionModel


class FaceEmotionDataset(Dataset):
    """Dataset for face emotion images"""

    def __init__(self, image_paths, labels, preprocessor):
        self.image_paths = image_paths
        self.labels = labels
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = self.preprocessor.preprocess_image(img_path)
            if image is None:
                image = torch.zeros(3, 224, 224)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = torch.zeros(3, 224, 224)

        return image, label


def load_fer2013_split(data_dir, split='training'):
    """Load FER2013 dataset split"""
    emotion_map = {
        'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
        'sad': 4, 'surprise': 5, 'neutral': 6
    }

    split_dir = Path(data_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    image_paths = []
    labels = []

    for emotion, label_id in emotion_map.items():
        emotion_dir = split_dir / emotion
        if not emotion_dir.exists():
            print(f"⚠️ Warning: {emotion} directory not found")
            continue

        for img_path in list(emotion_dir.glob("*.png")) + list(emotion_dir.glob("*.jpg")):
            image_paths.append(str(img_path))
            labels.append(label_id)

    return image_paths, labels


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({
            'loss': f"{running_loss / (pbar.n + 1):.4f}",
            'acc': f"{100 * correct / total:.2f}%"
        })

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, epoch):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]  ")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                'loss': f"{running_loss / (pbar.n + 1):.4f}",
                'acc': f"{100 * correct / total:.2f}%"
            })

    val_loss = running_loss / len(dataloader)
    val_acc = 100 * correct / total
    return val_loss, val_acc


def main():
    print("\n" + "=" * 60)
    print("🎭 FACE EMOTION MODEL TRAINING")
    print("=" * 60 + "\n")

    # Configuration (CPU FORCED)
    config = {
        'data_dir': 'datasets/face/fer2013',
        'batch_size': 32,
        'num_epochs': 10,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'device': 'cpu',   # ✅ FORCED CPU
        'save_dir': 'models/face_model',
        'num_workers': 4 if os.name != 'nt' else 0,
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Create directories
    Path(config['save_dir']).mkdir(parents=True, exist_ok=True)
    Path('experiments/face').mkdir(parents=True, exist_ok=True)

    # Initialize preprocessor
    print("Initializing preprocessor...")
    preprocessor = FacePreprocessor(
        target_size=(224, 224),
        grayscale=False,
        use_face_detection=False
    )

    # Load datasets
    print("\nLoading datasets...")
    try:
        train_paths, train_labels = load_fer2013_split(config['data_dir'], 'training')
        val_paths, val_labels = load_fer2013_split(config['data_dir'], 'validation')

        print(f"✓ Training samples: {len(train_paths)}")
        print(f"✓ Validation samples: {len(val_paths)}")

        if len(train_paths) == 0 or len(val_paths) == 0:
            print("\n❌ Error: No data found!")
            print("Please run: python prepare_datasets.py first")
            return
    except Exception as e:
        print(f"\n❌ Error loading datasets: {e}")
        print("Please run: python prepare_datasets.py first")
        return

    # Create datasets
    print("\nCreating PyTorch datasets...")
    train_dataset = FaceEmotionDataset(train_paths, train_labels, preprocessor)
    val_dataset = FaceEmotionDataset(val_paths, val_labels, preprocessor)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=False
    )

    # Initialize model
    print(f"\nInitializing model on {config['device']}...")
    model = FaceEmotionModel(num_classes=7, freeze_backbone=False)
    model = model.to(config['device'])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])

    # Training loop
    print(f"\n{'=' * 60}")
    print("STARTING TRAINING")
    print(f"{'=' * 60}\n")

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"{'=' * 60}")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config['device'], epoch
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, config['device'], epoch
        )
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"\nResults:")
        print(f"  Train → Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"  Val   → Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = Path(config['save_dir']) / 'best_model.pth'
            model.save_model(str(save_path))
            print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%)")

    final_path = Path(config['save_dir']) / 'final_model.pth'
    model.save_model(str(final_path))

    history_path = Path(config['save_dir']) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'=' * 60}")
    print("✅ TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Models saved to: {config['save_dir']}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
