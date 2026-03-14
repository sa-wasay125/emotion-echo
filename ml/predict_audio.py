"""
Audio Emotion Prediction - Inference Wrapper
Provides clean API for audio emotion detection.
Usage:
    from predict_audio import predict_audio_emotion
    result = predict_audio_emotion('path/to/audio.wav')
    # Returns: {'emotion': 'happy', 'confidence': 0.85, 'probabilities': {...}}
"""

import torch
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.audio_preprocess import AudioPreprocessor
from models.audio_model.audio_model_simple import SimpleAudioEmotionModel


class AudioEmotionPredictor:
    """Audio emotion prediction model wrapper."""
    
    EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    
    def __init__(self, model_path='models/audio_model/best_model.pth', device='cpu'):
        """Initialize the audio emotion predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.preprocessor = AudioPreprocessor(target_sr=16000, duration=3.0)
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        self.model = SimpleAudioEmotionModel(
            num_classes=checkpoint['num_classes'],
            n_mels=checkpoint.get('n_mels', 128)
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
    def predict(self, audio_path):
        """Predict emotion from audio file.
        
        Args:
            audio_path: Path to audio file (.wav, .mp3, etc.)
            
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
            # Preprocess audio
            waveform = self.preprocessor.preprocess_audio(audio_path, augment=False)
            if waveform is None:
                return {
                    'success': False,
                    'error': 'Failed to preprocess audio (invalid file or format)'
                }
            
            # Ensure correct shape
            if waveform.dim() == 2:
                waveform = waveform.squeeze(0)
            
            # Add batch dimension
            waveform = waveform.unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                logits = self.model(waveform)
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


def predict_audio_emotion(audio_path, model_path='models/audio_model/best_model.pth'):
    """Convenience function for audio emotion prediction.
    
    Args:
        audio_path: Path to audio file
        model_path: Path to model checkpoint (optional)
        
    Returns:
        dict: Prediction results
    """
    global _predictor
    
    if _predictor is None:
        _predictor = AudioEmotionPredictor(model_path=model_path)
    
    return _predictor.predict(audio_path)


def main():
    """CLI interface for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Audio Emotion Prediction')
    parser.add_argument('audio', type=str, help='Path to audio file')
    parser.add_argument('--model', type=str, default='models/audio_model/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--verbose', action='store_true', help='Show all probabilities')
    
    args = parser.parse_args()
    
    # Predict
    result = predict_audio_emotion(args.audio, args.model)
    
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