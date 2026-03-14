"""
Face Emotion Prediction - Inference Wrapper
Provides clean API for face emotion detection from images.
Usage:
    from predict_face import predict_face_emotion
    result = predict_face_emotion('path/to/image.jpg')
    # Returns: {'emotion': 'happy', 'confidence': 0.95, 'probabilities': {...}}
"""

import torch
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.face_preprocess import FacePreprocessor
from models.face_model.face_model import FaceEmotionModel


class FaceEmotionPredictor:
    """Face emotion prediction model wrapper."""
    
    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    def __init__(self, model_path='models/face_model/best_model.pth', device='cpu'):
        """Initialize the face emotion predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.preprocessor = FacePreprocessor()
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        self.model = FaceEmotionModel(num_classes=checkpoint['num_classes'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
    def predict(self, image_path):
        """Predict emotion from face image.
        
        Args:
            image_path: Path to image file or numpy array
            
        Returns:
            dict: {
                'emotion': str,           # Predicted emotion name
                'confidence': float,      # Confidence score (0-1)
                'probabilities': dict,    # All emotion probabilities
                'success': bool,          # Whether prediction succeeded
                'error': str or None      # Error message if failed
            }
        """
        try:
            # Preprocess image
            image_tensor = self.preprocessor.preprocess_image(image_path)
            if image_tensor is None:
                return {
                    'success': False,
                    'error': 'Failed to preprocess image (no face detected or invalid image)'
                }
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                logits = self.model(image_tensor)
                probs = torch.softmax(logits, dim=1)
                confidence, pred_idx = torch.max(probs, dim=1)
            
            # Build result
            emotion = self.EMOTIONS[pred_idx.item()]
            conf_value = confidence.item()
            
            # Get all probabilities
            prob_dict = {
                emotion: float(probs[0][i].item())
                for i, emotion in enumerate(self.EMOTIONS)
            }
            
            return {
                'success': True,
                'emotion': emotion,
                'confidence': conf_value,
                'probabilities': prob_dict,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Prediction failed: {str(e)}'
            }


# Global predictor instance (loaded once for efficiency)
_predictor = None


def predict_face_emotion(image_path, model_path='models/face_model/best_model.pth'):
    """Convenience function for face emotion prediction.
    
    Args:
        image_path: Path to face image
        model_path: Path to model checkpoint (optional)
        
    Returns:
        dict: Prediction results
    """
    global _predictor
    
    if _predictor is None:
        _predictor = FaceEmotionPredictor(model_path=model_path)
    
    return _predictor.predict(image_path)


def main():
    """CLI interface for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Face Emotion Prediction')
    parser.add_argument('image', type=str, help='Path to face image')
    parser.add_argument('--model', type=str, default='models/face_model/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--verbose', action='store_true', help='Show all probabilities')
    
    args = parser.parse_args()
    
    # Predict
    result = predict_face_emotion(args.image, args.model)
    
    if result['success']:
        print(f"\n✓ Emotion: {result['emotion']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        
        if args.verbose:
            print("\n  All probabilities:")
            for emotion, prob in sorted(result['probabilities'].items(),
                                       key=lambda x: x[1], reverse=True):
                print(f"    {emotion:<10} {prob:.1%}")
    else:
        print(f"\n✗ Error: {result['error']}")
    
    print()


if __name__ == '__main__':
    main()