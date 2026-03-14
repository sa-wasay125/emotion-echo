"""
Fusion Model - MLP for combining face + audio predictions
"""

import torch
import torch.nn as nn


class FusionMLP(nn.Module):
    """
    Multimodal fusion model.
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
        """Forward pass through fusion MLP."""
        x = torch.cat([face_probs, audio_probs], dim=1)
        return self.net(x)

    def predict(self, face_probs, audio_probs):
        """Get prediction with confidence."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(face_probs, audio_probs)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)
        return pred, conf, probs

    def save_model(self, path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_classes': 7,
        }, path)
        print(f"✓ Model saved to {path}")

    @classmethod
    def load_model(cls, path, device='cpu'):
        """Load model from checkpoint."""
        ckpt = torch.load(path, map_location=device)
        model = cls(num_classes=ckpt.get('num_classes', 7))
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model