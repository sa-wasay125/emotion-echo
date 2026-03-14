"""
Simple Audio Emotion Model without Transformers dependency
Uses CNN on mel-spectrograms - faster and no compatibility issues
"""

import torch
import torch.nn as nn
import torchaudio
from typing import Dict, Tuple
import numpy as np


class SimpleAudioEmotionModel(nn.Module):
    """
    CNN-based audio emotion detection using mel-spectrograms
    No transformer dependency - pure PyTorch
    """
    
    EMOTION_LABELS = [
        'neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'
    ]
    
    def __init__(self, num_classes: int = 8, n_mels: int = 128):
        super().__init__()
        
        self.num_classes = num_classes
        self.n_mels = n_mels
        
        # Mel-spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=n_mels
        )
        
        # CNN Feature Extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def extract_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to mel-spectrogram"""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-9)
        
        if mel_spec.dim() == 2:
            mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)
        elif mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(1)
        
        return mel_spec
    
    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass"""
        mel_spec = self.extract_mel_spectrogram(input_values)
        features = self.features(mel_spec)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return logits
    
    def predict(self, input_values: torch.Tensor, attention_mask: torch.Tensor = None, return_probs: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with confidence"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_values, attention_mask)
            if return_probs:
                probs = torch.softmax(logits, dim=-1)
                predicted = torch.argmax(probs, dim=-1)
                return predicted, probs
            else:
                predicted = torch.argmax(logits, dim=-1)
                return predicted, logits
    
    def predict_with_labels(self, input_values: torch.Tensor, attention_mask: torch.Tensor = None) -> Dict[str, any]:
        """Predict with formatted output"""
        predicted, probs = self.predict(input_values, attention_mask, return_probs=True)
        predicted_np = predicted.cpu().numpy()
        probs_np = probs.cpu().numpy()
        
        results = []
        for i in range(len(predicted_np)):
            emotion_id = int(predicted_np[i])
            emotion = self.EMOTION_LABELS[emotion_id]
            confidence = float(probs_np[i, emotion_id])
            
            emotion_probs = {
                label: float(probs_np[i, j])
                for j, label in enumerate(self.EMOTION_LABELS)
            }
            
            results.append({
                'emotion': emotion,
                'confidence': confidence,
                'probabilities': emotion_probs
            })
        
        return results[0] if len(results) == 1 else results
    
    def get_embeddings(self, input_values: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Extract embeddings for fusion"""
        self.eval()
        with torch.no_grad():
            mel_spec = self.extract_mel_spectrogram(input_values)
            features = self.features(mel_spec)
            embeddings = features.view(features.size(0), -1)
        return embeddings
    
    def save_model(self, save_path: str):
        """Save model"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_classes': self.num_classes,
            'n_mels': self.n_mels,
            'emotion_labels': self.EMOTION_LABELS
        }, save_path)
        print(f"✓ Model saved to {save_path}")
    
    @classmethod
    def load_model(cls, load_path: str, device: str = 'cpu'):
        """Load model"""
        print(f"Loading model from {load_path}...")
        checkpoint = torch.load(load_path, map_location=device)
        model = cls(num_classes=checkpoint['num_classes'], n_mels=checkpoint.get('n_mels', 128))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print(f"✓ Model loaded successfully")
        return model


# Alias for compatibility
AudioEmotionModel = SimpleAudioEmotionModel