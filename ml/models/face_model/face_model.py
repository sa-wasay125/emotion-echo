"""
Face emotion detection model using Vision Transformer
Optimized for CPU inference with pretrained weights
"""

import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig
from typing import Dict, Tuple
import numpy as np

class FaceEmotionModel(nn.Module):
    """
    Face emotion detection using Vision Transformer (ViT)
    
    Supports both training and inference modes
    """
    
    EMOTION_LABELS = [
        'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'
    ]
    
    def __init__(
        self,
        num_classes: int = 7,
        pretrained_model: str = "google/vit-base-patch16-224",
        freeze_backbone: bool = False
    ):
        """
        Args:
            num_classes: Number of emotion classes
            pretrained_model: HuggingFace model identifier
            freeze_backbone: Freeze pretrained weights
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.pretrained_model = pretrained_model
        
        # Load pretrained ViT
        self.model = ViTForImageClassification.from_pretrained(
            pretrained_model,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.model.vit.parameters():
                param.requires_grad = False
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            pixel_values: Input tensor (B, C, H, W)
            
        Returns:
            Logits (B, num_classes)
        """
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits
    
    def predict(
        self,
        pixel_values: torch.Tensor,
        return_probs: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict emotion with confidence
        
        Args:
            pixel_values: Input tensor (B, C, H, W)
            return_probs: Return softmax probabilities
            
        Returns:
            Tuple of (predicted_class, confidence/probabilities)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(pixel_values)
            
            if return_probs:
                probs = torch.softmax(logits, dim=-1)
                predicted = torch.argmax(probs, dim=-1)
                confidence = torch.max(probs, dim=-1).values
                return predicted, probs
            else:
                predicted = torch.argmax(logits, dim=-1)
                return predicted, logits
    
    def predict_with_labels(
        self,
        pixel_values: torch.Tensor
    ) -> Dict[str, any]:
        """
        Predict emotion and return formatted output
        
        Args:
            pixel_values: Input tensor (B, C, H, W)
            
        Returns:
            Dictionary with emotion, confidence, and probabilities
        """
        predicted, probs = self.predict(pixel_values, return_probs=True)
        
        # Convert to numpy for easier handling
        predicted_np = predicted.cpu().numpy()
        probs_np = probs.cpu().numpy()
        
        results = []
        for i in range(len(predicted_np)):
            emotion_id = int(predicted_np[i])
            emotion = self.EMOTION_LABELS[emotion_id]
            confidence = float(probs_np[i, emotion_id])
            
            # All emotion probabilities
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
        pixel_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract feature embeddings (for fusion model)
        
        Args:
            pixel_values: Input tensor (B, C, H, W)
            
        Returns:
            Feature embeddings (B, hidden_dim)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.model.vit(pixel_values=pixel_values)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings
    
    def save_model(self, save_path: str):
        """Save model weights"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_classes': self.num_classes,
            'pretrained_model': self.pretrained_model,
            'emotion_labels': self.EMOTION_LABELS
        }, save_path)
    
    @classmethod
    def load_model(cls, load_path: str, device: str = 'cpu'):
        """Load model weights"""
        checkpoint = torch.load(load_path, map_location=device)
        
        model = cls(
            num_classes=checkpoint['num_classes'],
            pretrained_model=checkpoint['pretrained_model']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model


class LightweightFaceModel(nn.Module):
    """
    Lightweight CNN-based model for faster CPU inference
    Alternative to ViT for resource-constrained scenarios
    """
    
    EMOTION_LABELS = [
        'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'
    ]
    
    def __init__(self, num_classes: int = 7, input_channels: int = 3):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Efficient CNN architecture
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def predict_with_labels(self, pixel_values: torch.Tensor) -> Dict[str, any]:
        """Predict with label mapping"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(pixel_values)
            probs = torch.softmax(logits, dim=-1)
            predicted = torch.argmax(probs, dim=-1)
        
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
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings for fusion"""
        self.eval()
        with torch.no_grad():
            x = self.features(x)
            x = x.view(x.size(0), -1)
        return x


# Inference wrapper for easy deployment
class FaceEmotionInference:
    """
    High-level inference interface for face emotion detection
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
            use_lightweight: Use lightweight CNN instead of ViT
        """
        self.device = device
        
        if model_path:
            self.model = FaceEmotionModel.load_model(model_path, device)
        else:
            # Use pretrained without fine-tuning
            if use_lightweight:
                self.model = LightweightFaceModel()
            else:
                self.model = FaceEmotionModel()
            self.model.to(device)
            self.model.eval()
    
    def predict(
        self,
        preprocessed_image: torch.Tensor
    ) -> Dict[str, any]:
        """
        Predict emotion from preprocessed image
        
        Args:
            preprocessed_image: Tensor (C, H, W) or (B, C, H, W)
            
        Returns:
            Prediction dictionary
        """
        # Add batch dimension if needed
        if preprocessed_image.dim() == 3:
            preprocessed_image = preprocessed_image.unsqueeze(0)
        
        preprocessed_image = preprocessed_image.to(self.device)
        
        return self.model.predict_with_labels(preprocessed_image)


if __name__ == "__main__":
    # Test model
    model = FaceEmotionModel()
    print(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model.predict_with_labels(dummy_input)
    print(f"Test prediction: {output}") 