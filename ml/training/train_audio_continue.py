"""
Continue training audio model from saved checkpoint
Run from ml/: python training/train_audio_continue.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.audio_preprocess import AudioPreprocessor, parse_ravdess_filename
from models.audio_model.audio_model_simple import SimpleAudioEmotionModel


class AudioEmotionDataset(Dataset):
    def __init__(self, audio_paths, labels, preprocessor):
        self.audio_paths = audio_paths
        self.labels = labels
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        try:
            waveform = self.preprocessor.preprocess_audio(
                self.audio_paths[idx], augment=False
            )
            if waveform.dim() == 2:
                waveform = waveform.squeeze(0)
        except Exception:
            waveform = torch.zeros(16000 * 3)
        return waveform, self.labels[idx]


def load_ravdess_dataset(data_dir):
    emotion_map = {
        'neutral': 0, 'calm': 1, 'happy': 2, 'sad': 3,
        'angry': 4, 'fearful': 5, 'disgust': 6, 'surprised': 7
    }
    data_path = Path(data_dir) / 'Audio_Speech_Actors_01-24'
    audio_paths, labels = [], []
    for actor_dir in data_path.glob('Actor_*'):
        for audio_file in actor_dir.glob('*.wav'):
            meta = parse_ravdess_filename(audio_file.name)
            if meta['emotion'] in emotion_map:
                audio_paths.append(str(audio_file))
                labels.append(emotion_map[meta['emotion']])
    return audio_paths, labels


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for waveforms, labels in pbar:
        waveforms, labels = waveforms.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(waveforms)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        _, pred = torch.max(out, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
        pbar.set_postfix({
            'loss': f"{loss_sum / (pbar.n + 1):.4f}",
            'acc': f"{100 * correct / total:.2f}%"
        })
    return loss_sum / len(loader), 100 * correct / total


def validate(model, loader, criterion, device, epoch):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]  ")
        for waveforms, labels in pbar:
            waveforms, labels = waveforms.to(device), labels.to(device)
            out = model(waveforms)
            loss = criterion(out, labels)
            loss_sum += loss.item()
            _, pred = torch.max(out, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            pbar.set_postfix({
                'loss': f"{loss_sum / (pbar.n + 1):.4f}",
                'acc': f"{100 * correct / total:.2f}%"
            })
    return loss_sum / len(loader), 100 * correct / total


def main():
    print("\n" + "="*60)
    print("🎙️ AUDIO TRAINING - CONTINUING FROM CHECKPOINT")
    print("="*60 + "\n")

    config = {
        'data_dir':      'datasets/audio/ravdess',
        'checkpoint':    'models/audio_model/best_model.pth',
        'save_dir':      'models/audio_model',
        'history_path':  'models/audio_model/training_history.json',
        # ── training hyper-params ──────────────────────────────
        'batch_size':    8,      # bigger than before → better gradients
        'num_epochs':    100,    # plenty of room to improve
        'learning_rate': 5e-4,   # lower LR for fine-tuning
        'weight_decay':  1e-4,
        # ── hardware ──────────────────────────────────────────
        'device':        'cpu',  # RTX 5060 not yet supported
        'num_workers':   0,
    }

    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    Path(config['save_dir']).mkdir(parents=True, exist_ok=True)

    # ── Load existing history ─────────────────────────────────
    history_file = Path(config['history_path'])
    if history_file.exists():
        with open(history_file) as f:
            history = json.load(f)
        start_epoch = len(history['train_loss']) + 1
        best_val_acc = max(history['val_acc'])
        print(f"\n✓ Resuming from epoch {start_epoch}")
        print(f"✓ Previous best val acc: {best_val_acc:.2f}%")
    else:
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        start_epoch = 1
        best_val_acc = 0.0
        print("\n⚠️  No history found - starting fresh")

    # ── Dataset ───────────────────────────────────────────────
    print("\nLoading RAVDESS dataset...")
    preprocessor = AudioPreprocessor(target_sr=16000, duration=3.0)
    audio_paths, labels = load_ravdess_dataset(config['data_dir'])
    print(f"✓ Total files: {len(audio_paths)}")

    train_p, tmp_p, train_l, tmp_l = train_test_split(
        audio_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_p, _, val_l, _ = train_test_split(
        tmp_p, tmp_l, test_size=0.5, random_state=42, stratify=tmp_l
    )
    print(f"✓ Train: {len(train_p)} | Val: {len(val_p)}")

    train_loader = DataLoader(
        AudioEmotionDataset(train_p, train_l, preprocessor),
        batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers']
    )
    val_loader = DataLoader(
        AudioEmotionDataset(val_p, val_l, preprocessor),
        batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers']
    )

    # ── Model ─────────────────────────────────────────────────
    print("\nLoading model from checkpoint...")
    checkpoint = torch.load(
        config['checkpoint'], map_location=config['device']
    )
    model = SimpleAudioEmotionModel(
        num_classes=checkpoint['num_classes'],
        n_mels=checkpoint.get('n_mels', 128)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config['device'])
    print(f"✓ Model loaded - {sum(p.numel() for p in model.parameters()):,} params")

    criterion  = nn.CrossEntropyLoss()
    optimizer  = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7
    )

    # ── Training loop ─────────────────────────────────────────
    end_epoch  = start_epoch + config['num_epochs'] - 1
    print(f"\n{'='*60}")
    print(f"TRAINING: epochs {start_epoch} → {end_epoch}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(start_epoch, end_epoch + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{end_epoch}")
        print(f"{'='*60}")

        t0 = time.time()
        tr_loss, tr_acc = train_epoch(
            model, train_loader, criterion, optimizer, config['device'], epoch
        )
        vl_loss, vl_acc = validate(
            model, val_loader,   criterion,           config['device'], epoch
        )
        scheduler.step(vl_acc)

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(vl_loss)
        history['val_acc'].append(vl_acc)

        print(f"\nResults:")
        print(f"  Train → Loss: {tr_loss:.4f} | Acc: {tr_acc:.2f}%")
        print(f"  Val   → Loss: {vl_loss:.4f} | Acc: {vl_acc:.2f}%")
        print(f"  Time : {time.time() - t0:.1f}s")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            model.save_model(str(Path(config['save_dir']) / 'best_model.pth'))
            print(f"  ✓ NEW BEST → {vl_acc:.2f}%")

        # Save history every epoch (safe against crashes)
        with open(config['history_path'], 'w') as f:
            json.dump(history, f, indent=2)

    # ── Final save ────────────────────────────────────────────
    model.save_model(str(Path(config['save_dir']) / 'final_model.pth'))
    total = time.time() - start_time

    print(f"\n{'='*60}")
    print("✅ TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best val accuracy : {best_val_acc:.2f}%")
    print(f"Total time        : {total/60:.1f} min")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()