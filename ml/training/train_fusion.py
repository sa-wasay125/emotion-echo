"""
Fusion Model Training
Combines face + audio emotion predictions into a single final emotion.
Run from ml/: python training/train_fusion.py

Strategy: Both models output softmax probabilities over their emotion sets.
We map them to a shared 7-emotion space, then train a small MLP to learn
optimal fusion weights using synthetic paired samples.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import time
from pathlib import Path
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.face_model.face_model import FaceEmotionModel
from models.audio_model.audio_model_simple import SimpleAudioEmotionModel
from preprocessing.face_preprocess import FacePreprocessor
from preprocessing.audio_preprocess import AudioPreprocessor, parse_ravdess_filename
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── Shared emotion space (7 emotions, dropping RAVDESS "calm") ───────────────
SHARED_EMOTIONS = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Mapping: face model output index → shared emotion index
# FER2013: angry=0, disgust=1, fear=2, happy=3, sad=4, surprise=5, neutral=6
FACE_TO_SHARED = {
    0: 0,  # angry    → angry
    1: 1,  # disgust  → disgust
    2: 2,  # fear     → fearful
    3: 3,  # happy    → happy
    4: 5,  # sad      → sad
    5: 6,  # surprise → surprised
    6: 4,  # neutral  → neutral
}

# Mapping: audio model output index → shared emotion index
# RAVDESS: neutral=0, calm=1, happy=2, sad=3, angry=4, fearful=5, disgust=6, surprised=7
AUDIO_TO_SHARED = {
    0: 4,  # neutral  → neutral
    1: 4,  # calm     → neutral (closest mapping)
    2: 3,  # happy    → happy
    3: 5,  # sad      → sad
    4: 0,  # angry    → angry
    5: 2,  # fearful  → fearful
    6: 1,  # disgust  → disgust
    7: 6,  # surprised→ surprised
}


def remap_probs(probs: torch.Tensor, mapping: dict, n_shared: int = 7) -> torch.Tensor:
    """Remap probability vector to shared emotion space."""
    batch_size = probs.shape[0]
    shared = torch.zeros(batch_size, n_shared)
    for src_idx, dst_idx in mapping.items():
        if src_idx < probs.shape[1]:
            shared[:, dst_idx] += probs[:, src_idx]
    # Renormalize
    row_sums = shared.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return shared / row_sums


# ── Fusion MLP ────────────────────────────────────────────────────────────────
class FusionMLP(nn.Module):
    """
    Input:  concatenated face_probs (7) + audio_probs (7) = 14 dims
    Output: shared emotion logits (7)
    """
    def __init__(self, input_dim=14, hidden_dim=64, num_classes=7, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, face_probs, audio_probs):
        x = torch.cat([face_probs, audio_probs], dim=1)
        return self.net(x)

    def predict(self, face_probs, audio_probs):
        self.eval()
        with torch.no_grad():
            logits = self.forward(face_probs, audio_probs)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)
        return pred, conf, probs

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_classes': 7,
            'emotions': SHARED_EMOTIONS,
            'face_to_shared': FACE_TO_SHARED,
            'audio_to_shared': AUDIO_TO_SHARED,
        }, path)
        print(f"✓ Model saved to {path}")

    @classmethod
    def load_model(cls, path, device='cpu'):
        ckpt = torch.load(path, map_location=device)
        model = cls(num_classes=ckpt['num_classes'])
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model


# ── Synthetic Dataset ─────────────────────────────────────────────────────────
class SyntheticFusionDataset(Dataset):
    """
    Generates synthetic face+audio probability pairs for each shared emotion.
    Since we don't have paired face+audio samples, we simulate realistic
    per-emotion probability distributions using Dirichlet noise.
    """
    def __init__(self, n_samples_per_class=500, noise_level=0.3):
        self.data = []
        self.labels = []
        n_classes = len(SHARED_EMOTIONS)

        rng = np.random.RandomState(42)

        for emotion_idx in range(n_classes):
            for _ in range(n_samples_per_class):
                # --- Face probs: peaked at correct class + noise ---
                face_base = np.ones(n_classes) * noise_level
                face_base[emotion_idx] += 3.0
                face_probs = rng.dirichlet(face_base)

                # --- Audio probs: slightly noisier (lower accuracy model) ---
                audio_base = np.ones(n_classes) * noise_level
                audio_base[emotion_idx] += 2.0  # less confident
                audio_probs = rng.dirichlet(audio_base)

                self.data.append((face_probs, audio_probs))
                self.labels.append(emotion_idx)

        # Add hard cases: conflicting modalities (teaches robustness)
        for emotion_idx in range(n_classes):
            wrong_idx = (emotion_idx + rng.randint(1, n_classes)) % n_classes
            for _ in range(n_samples_per_class // 4):
                # Face correct, audio wrong
                face_base = np.ones(n_classes) * 0.2
                face_base[emotion_idx] += 3.0
                face_probs = rng.dirichlet(face_base)

                audio_base = np.ones(n_classes) * 0.2
                audio_base[wrong_idx] += 2.5
                audio_probs = rng.dirichlet(audio_base)

                self.data.append((face_probs, audio_probs))
                self.labels.append(emotion_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        face_p, audio_p = self.data[idx]
        return (
            torch.tensor(face_p, dtype=torch.float32),
            torch.tensor(audio_p, dtype=torch.float32),
            self.labels[idx]
        )


# ── Live Embedding Dataset (uses real models on RAVDESS) ──────────────────────
class LiveFusionDataset(Dataset):
    """
    Generates real face+audio embeddings using trained models on RAVDESS.
    For face: uses a neutral/random face image since RAVDESS is audio-only.
    For audio: uses actual RAVDESS audio files.
    Falls back to synthetic if models fail.
    """
    def __init__(self, audio_model, audio_preprocessor, audio_paths, audio_labels):
        self.samples = []
        n_shared = len(SHARED_EMOTIONS)
        print("  Generating audio embeddings from RAVDESS...")
        audio_model.eval()

        for path, label in tqdm(zip(audio_paths, audio_labels),
                                total=len(audio_paths), desc="  Embedding"):
            try:
                waveform = audio_preprocessor.preprocess_audio(path, augment=False)
                if waveform.dim() == 2:
                    waveform = waveform.squeeze(0)
                waveform = waveform.unsqueeze(0)  # batch dim

                with torch.no_grad():
                    logits = audio_model(waveform)
                    audio_probs_raw = torch.softmax(logits, dim=1)

                # Remap to shared space
                audio_probs = remap_probs(audio_probs_raw, AUDIO_TO_SHARED, n_shared)

                # For face: simulate a face prediction peaked at the same emotion
                # (in real inference, this comes from the actual face frame)
                rng = np.random.RandomState(label)
                face_base = np.ones(n_shared) * 0.3
                face_base[label] += 2.5
                face_probs = torch.tensor(
                    rng.dirichlet(face_base), dtype=torch.float32
                ).unsqueeze(0)

                self.samples.append((
                    face_probs.squeeze(0),
                    audio_probs.squeeze(0),
                    label
                ))
            except Exception:
                continue

        print(f"  ✓ Generated {len(self.samples)} live samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Training / Validation ─────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for face_p, audio_p, labels in loader:
        face_p  = face_p.to(device)
        audio_p = audio_p.to(device)
        labels  = labels.to(device)

        optimizer.zero_grad()
        out  = model(face_p, audio_p)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        _, pred = torch.max(out, 1)
        total   += labels.size(0)
        correct += (pred == labels).sum().item()

    return loss_sum / len(loader), 100 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for face_p, audio_p, labels in loader:
            face_p  = face_p.to(device)
            audio_p = audio_p.to(device)
            labels  = labels.to(device)

            out  = model(face_p, audio_p)
            loss = criterion(out, labels)

            loss_sum += loss.item()
            _, pred = torch.max(out, 1)
            total   += labels.size(0)
            correct += (pred == labels).sum().item()

    return loss_sum / len(loader), 100 * correct / total


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("🔀 FUSION MODEL TRAINING")
    print("="*60 + "\n")

    config = {
        'audio_checkpoint':     'models/audio_model/best_model.pth',
        'save_dir':             'models/fusion_model',
        'history_path':         'models/fusion_model/training_history.json',
        'data_dir':             'datasets/audio/ravdess',
        # training
        'batch_size':           64,
        'num_epochs':           80,
        'learning_rate':        1e-3,
        'weight_decay':         1e-4,
        'n_synthetic_per_class': 600,
        'device':               'cpu',
    }

    for k, v in config.items():
        print(f"  {k}: {v}")

    Path(config['save_dir']).mkdir(parents=True, exist_ok=True)

    device = config['device']

    # ── Load audio model ──────────────────────────────────────
    print("\nLoading audio model...")
    audio_ckpt = torch.load(config['audio_checkpoint'], map_location=device)
    audio_model = SimpleAudioEmotionModel(
        num_classes=audio_ckpt['num_classes'],
        n_mels=audio_ckpt.get('n_mels', 128)
    )
    audio_model.load_state_dict(audio_ckpt['model_state_dict'])
    audio_model.eval()
    print(f"✓ Audio model loaded ({audio_ckpt['num_classes']} classes)")

    # ── Build dataset ─────────────────────────────────────────
    print("\nBuilding fusion dataset...")

    # Part 1: Synthetic samples (core training signal)
    print("  Generating synthetic samples...")
    synthetic_ds = SyntheticFusionDataset(
        n_samples_per_class=config['n_synthetic_per_class']
    )
    print(f"  ✓ Synthetic samples: {len(synthetic_ds)}")

    # Part 2: Live audio embeddings from RAVDESS
    print("  Loading RAVDESS for live embeddings...")
    audio_preprocessor = AudioPreprocessor(target_sr=16000, duration=3.0)
    ravdess_path = Path(config['data_dir']) / 'Audio_Speech_Actors_01-24'

    # Map RAVDESS emotion IDs to shared space
    ravdess_emotion_map = {
        'neutral': 4, 'calm': 4, 'happy': 3, 'sad': 5,
        'angry': 0, 'fearful': 2, 'disgust': 1, 'surprised': 6
    }
    audio_paths, audio_labels = [], []
    for actor_dir in ravdess_path.glob('Actor_*'):
        for f in actor_dir.glob('*.wav'):
            meta = parse_ravdess_filename(f.name)
            if meta['emotion'] in ravdess_emotion_map:
                audio_paths.append(str(f))
                audio_labels.append(ravdess_emotion_map[meta['emotion']])

    live_ds = LiveFusionDataset(
        audio_model, audio_preprocessor, audio_paths, audio_labels
    )

    # Combine both datasets
    from torch.utils.data import ConcatDataset
    full_ds = ConcatDataset([synthetic_ds, live_ds])
    print(f"\n✓ Total samples: {len(full_ds)}")

    # Train / val split (80/20)
    n_total = len(full_ds)
    n_val   = int(n_total * 0.2)
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"✓ Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'],
                              shuffle=False, num_workers=0)

    # ── Fusion model ──────────────────────────────────────────
    model     = FusionMLP(input_dim=14, hidden_dim=64,
                          num_classes=7, dropout=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=config['learning_rate'],
                           weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Fusion MLP: {total_params:,} parameters")

    # ── Training loop ─────────────────────────────────────────
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    start_time   = time.time()

    print(f"\n{'='*60}")
    print(f"TRAINING: {config['num_epochs']} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, config['num_epochs'] + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = validate(model, val_loader, criterion, device)
        scheduler.step(vl_acc)

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(vl_loss)
        history['val_acc'].append(vl_acc)

        marker = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            model.save_model(str(Path(config['save_dir']) / 'best_model.pth'))
            marker = " ← NEW BEST"

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{config['num_epochs']} | "
                  f"Train: {tr_acc:.1f}% loss={tr_loss:.4f} | "
                  f"Val: {vl_acc:.1f}% loss={vl_loss:.4f}{marker}")

        with open(config['history_path'], 'w') as f:
            json.dump(history, f, indent=2)

    model.save_model(str(Path(config['save_dir']) / 'final_model.pth'))

    print(f"\n{'='*60}")
    print("✅ FUSION TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best val accuracy : {best_val_acc:.2f}%")
    print(f"Total time        : {(time.time()-start_time)/60:.1f} min")
    print(f"Emotions          : {SHARED_EMOTIONS}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()