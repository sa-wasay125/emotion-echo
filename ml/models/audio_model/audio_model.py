"""
Audio emotion detection model using Wav2Vec2
Fixed for compatibility with transformers library
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
import numpy as np

# Fixed import for Wav2Vec2
try:
    from transformers import Wav2Vec2Model
    print("✓ Wav2Vec2Model imported successfully")
except ImportError:
    try:
        from transformers import Wav2Vec2ForCTC as Wav2Vec2Model
        print("⚠️ Using Wav2Vec2ForCTC as fallback")
    except Exception as e:
        print(f"❌ Error importing Wav2Vec2: {e}")
        print("Installing required package...")
        import subprocess
        subprocess.check_call(["pip", "install", "--upgrade", "transformers"])
        from transformers import Wav2Vec2Model

class AudioEmotionModel(nn.Module):
    """
    Audio emotion detection using Wav2Vec2
    
    Architecture: Wav2Vec2 encoder -> Pooling -> Classifier
    """
    
    EMOTION_LABELS = [
        'neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'
    ]
    
    def __init__(
        self,
        num_classes: int = 8,
        pretrained_model: str = "facebook/wav2vec2-base",
        freeze_encoder: bool = False,
        pooling: str = "mean"
    ):
        """
        Args:
            num_classes: Number of emotion classes
            pretrained_model: HuggingFace model identifier
            freeze_encoder: Freeze pretrained encoder
            pooling: Pooling strategy ('mean', 'max', 'attention')
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.pretrained_model = pretrained_model
        self.pooling = pooling
        
        # Load pretrained Wav2Vec2
        print(f"Loading {pretrained_model}...")
        try:
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model)
            hidden_size = self.wav2vec2.config.hidden_size
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            print("Using default configuration...")
            from transformers import Wav2Vec2Config
            config = Wav2Vec2Config()
            self.wav2vec2 = Wav2Vec2Model(config)
            hidden_size = config.hidden_size
        
        # Optionally freeze encoder
        if freeze_encoder:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
        
        # Pooling layer
        if pooling == "attention":
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def pool_features(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Pool sequence features into fixed-size representation
        
        Args:
            hidden_states: Tensor (B, T, H)
            attention_mask: Mask for valid frames (B, T)
            
        Returns:
            Pooled features (B, H)
        """
        if self.pooling == "mean":
            if attention_mask is not None:
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_hidden / sum_mask
            else:
                pooled = torch.mean(hidden_states, dim=1)
        
        elif self.pooling == "max":
            pooled = torch.max(hidden_states, dim=1).values
        
        elif self.pooling == "attention":
            # Attention-based pooling
            attention_weights = self.attention(hidden_states)  # (B, T, 1)
            attention_weights = torch.softmax(attention_weights, dim=1)
            pooled = torch.sum(hidden_states * attention_weights, dim=1)
        
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        return pooled
    
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_values: Raw waveform (B, T)
            attention_mask: Attention mask (B, T)
            
        Returns:
            Logits (B, num_classes)
        """
        # Extract features with Wav2Vec2
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask
        )
        
        hidden_states = outputs.last_hidden_state  # (B, T', H)
        
        # Pool features
        pooled = self.pool_features(hidden_states, attention_mask)
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits
    
    def predict(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor = None,
        return_probs: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict emotion with confidence
        
        Args:
            input_values: Raw waveform (B, T)
            attention_mask: Attention mask (B, T)
            return_probs: Return softmax probabilities
            
        Returns:
            Tuple of (predicted_class, confidence/probabilities)
        """
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
    
    def predict_with_labels(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> Dict[str, any]:
        """
        Predict emotion and return formatted output
        
        Args:
            input_values: Raw waveform (B, T)
            attention_mask: Attention mask (B, T)
            
        Returns:
            Dictionary with emotion, confidence, and probabilities
        """
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
    
    def get_embeddings(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Extract feature embeddings (for fusion model)
        
        Args:
            input_values: Raw waveform (B, T)
            attention_mask: Attention mask (B, T)
            
        Returns:
            Feature embeddings (B, hidden_dim)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.wav2vec2(
                input_values=input_values,
                attention_mask=attention_mask
            )
            hidden_states = outputs.last_hidden_state
            embeddings = self.pool_features(hidden_states, attention_mask)
        
        return embeddings
    
    def save_model(self, save_path: str):
        """Save model weights"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_classes': self.num_classes,
            'pretrained_model': self.pretrained_model,
            'pooling': self.pooling,
            'emotion_labels': self.EMOTION_LABELS
        }, save_path)
        print(f"✓ Model saved to {save_path}")
    
    @classmethod
    def load_model(cls, load_path: str, device: str = 'cpu'):
        """Load model weights"""
        print(f"Loading model from {load_path}...")
        checkpoint = torch.load(load_path, map_location=device)
        
        model = cls(
            num_classes=checkpoint['num_classes'],
            pretrained_model=checkpoint['pretrained_model'],
            pooling=checkpoint.get('pooling', 'mean')
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"✓ Model loaded successfully")
        return model


class LightweightAudioModel(nn.Module):
    """
    Lightweight CNN-based model for audio emotion detection
    Uses mel-spectrogram features instead of raw waveform
    """
    
    EMOTION_LABELS = [
        'neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'
    ]
    
    def __init__(self, num_classes: int = 8, n_mels: int = 128):
        super().__init__()
        
        self.num_classes = num_classes
        
        # CNN for mel-spectrogram
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_spec: Mel spectrogram (B, 1, n_mels, T)
        """
        x = self.features(mel_spec)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class AudioEmotionInference:
    """
    High-level inference interface for audio emotion detection
    """
    
    def __init__(
        self,
        model_path: str = None,
        device: str = 'cpu',
        use_lightweight: bool = False
    ):
        """
        Args:
            model_path: Path to saved model weights
            device: 'cpu' or 'cuda'
            use_lightweight: Use lightweight CNN instead of Wav2Vec2
        """
        self.device = device
        
        if model_path:
            self.model = AudioEmotionModel.load_model(model_path, device)
        else:
            if use_lightweight:
                self.model = LightweightAudioModel()
            else:
                self.model = AudioEmotionModel()
            self.model.to(device)
            self.model.eval()
    
    def predict(
        self,
        preprocessed_audio: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> Dict[str, any]:
        """
        Predict emotion from preprocessed audio
        
        Args:
            preprocessed_audio: Waveform tensor (T,) or (B, T)
            attention_mask: Attention mask (optional)
            
        Returns:
            Prediction dictionary
        """
        # Add batch dimension if needed
        if preprocessed_audio.dim() == 1:
            preprocessed_audio = preprocessed_audio.unsqueeze(0)
        
        preprocessed_audio = preprocessed_audio.to(self.device)
        
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        return self.model.predict_with_labels(preprocessed_audio, attention_mask)


if __name__ == "__main__":
    # Test model
    print("\nTesting AudioEmotionModel...")
    model = AudioEmotionModel()
    print(f"✓ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    dummy_input = torch.randn(1, 16000 * 3)  # 3 seconds at 16kHz
    output = model.predict_with_labels(dummy_input)
    print(f"✓ Test prediction: {output}")