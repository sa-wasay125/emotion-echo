"""
Fusion Emotion Prediction - Multimodal Inference Wrapper
Combines face + audio for final emotion prediction.
Usage:
    from predict_fusion import predict_fusion_emotion
    result = predict_fusion_emotion('face.jpg', 'audio.wav')
    # Returns: {'emotion': 'happy', 'confidence': 0.97, ...}
"""

import torch
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.face_preprocess import FacePreprocessor
from preprocessing.audio_preprocess import AudioPreprocessor
from models.face_model.face_model import FaceEmotionModel
from models.audio_model.audio_model_simple import SimpleAudioEmotionModel
from models.fusion_model.fusion_model import FusionMLP


class FusionEmotionPredictor:
    """Multimodal fusion emotion predictor."""
    
    # Shared emotion space (7 emotions)
    EMOTIONS = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    
    # Mapping from face model indices to shared space
    FACE_TO_SHARED = {
        0: 0,  # angry    → angry
        1: 1,  # disgust  → disgust
        2: 2,  # fear     → fearful
        3: 3,  # happy    → happy
        4: 5,  # sad      → sad
        5: 6,  # surprise → surprised
        6: 4,  # neutral  → neutral
    }
    
    # Mapping from audio model indices to shared space
    AUDIO_TO_SHARED = {
        0: 4,  # neutral   → neutral
        1: 4,  # calm      → neutral
        2: 3,  # happy     → happy
        3: 5,  # sad       → sad
        4: 0,  # angry     → angry
        5: 2,  # fearful   → fearful
        6: 1,  # disgust   → disgust
        7: 6,  # surprised → surprised
    }
    
    def __init__(self,
                 face_model_path='models/face_model/best_model.pth',
                 audio_model_path='models/audio_model/best_model.pth',
                 fusion_model_path='models/fusion_model/best_model.pth',
                 device='cpu'):
        """Initialize fusion predictor with all three models.
        
        Args:
            face_model_path: Path to face model checkpoint
            audio_model_path: Path to audio model checkpoint
            fusion_model_path: Path to fusion model checkpoint
            device: 'cpu' or 'cuda'
        """
        self.device = device
        
        # Preprocessors
        self.face_preprocessor = FacePreprocessor()
        self.audio_preprocessor = AudioPreprocessor(target_sr=16000, duration=3.0)
        
        # Load face model
        face_ckpt = torch.load(face_model_path, map_location=device)
        self.face_model = FaceEmotionModel(num_classes=face_ckpt['num_classes'])
        self.face_model.load_state_dict(face_ckpt['model_state_dict'])
        self.face_model.to(device)
        self.face_model.eval()
        
        # Load audio model
        audio_ckpt = torch.load(audio_model_path, map_location=device)
        self.audio_model = SimpleAudioEmotionModel(
            num_classes=audio_ckpt['num_classes'],
            n_mels=audio_ckpt.get('n_mels', 128)
        )
        self.audio_model.load_state_dict(audio_ckpt['model_state_dict'])
        self.audio_model.to(device)
        self.audio_model.eval()
        
        # Load fusion model
        self.fusion_model = FusionMLP.load_model(fusion_model_path, device=device)
        self.fusion_model.to(device)
        
    def remap_probs(self, probs, mapping):
        """Remap probability vector to shared emotion space."""
        shared = torch.zeros(1, 7)
        for src_idx, dst_idx in mapping.items():
            if src_idx < probs.shape[1]:
                shared[0, dst_idx] += probs[0, src_idx]
        # Renormalize
        shared = shared / shared.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return shared
    
    def predict(self, face_path=None, audio_path=None):
        """Predict emotion from face and/or audio.
        
        Args:
            face_path: Path to face image (optional if audio provided)
            audio_path: Path to audio file (optional if face provided)
            
        Returns:
            dict: {
                'emotion': str,              # Final fused emotion
                'confidence': float,         # Fusion confidence (0-1)
                'probabilities': dict,       # Fusion probabilities
                'face_emotion': str,         # Face-only prediction
                'face_confidence': float,    # Face confidence
                'audio_emotion': str,        # Audio-only prediction
                'audio_confidence': float,   # Audio confidence
                'mode': str,                 # 'fusion', 'face_only', or 'audio_only'
                'success': bool,
                'error': str or None
            }
        """
        try:
            face_probs = None
            audio_probs = None
            face_result = {}
            audio_result = {}
            
            # Process face
            if face_path:
                face_tensor = self.face_preprocessor.preprocess_image(face_path)
                if face_tensor is not None:
                    face_tensor = face_tensor.unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        face_logits = self.face_model(face_tensor)
                        face_probs_raw = torch.softmax(face_logits, dim=1)
                        face_probs = self.remap_probs(face_probs_raw, self.FACE_TO_SHARED)
                        
                        face_conf, face_idx = torch.max(face_probs, dim=1)
                        face_result = {
                            'emotion': self.EMOTIONS[face_idx.item()],
                            'confidence': face_conf.item()
                        }
            
            # Process audio
            if audio_path:
                audio_waveform = self.audio_preprocessor.preprocess_audio(audio_path, augment=False)
                if audio_waveform is not None:
                    if audio_waveform.dim() == 2:
                        audio_waveform = audio_waveform.squeeze(0)
                    audio_waveform = audio_waveform.unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        audio_logits = self.audio_model(audio_waveform)
                        audio_probs_raw = torch.softmax(audio_logits, dim=1)
                        audio_probs = self.remap_probs(audio_probs_raw, self.AUDIO_TO_SHARED)
                        
                        audio_conf, audio_idx = torch.max(audio_probs, dim=1)
                        audio_result = {
                            'emotion': self.EMOTIONS[audio_idx.item()],
                            'confidence': audio_conf.item()
                        }
            
            # Determine mode and fuse
            if face_probs is not None and audio_probs is not None:
                # Both available - use fusion
                pred_idx, conf, probs = self.fusion_model.predict(face_probs, audio_probs)
                mode = 'fusion'
                
            elif face_probs is not None:
                # Face only
                pred_idx, conf = torch.max(face_probs, dim=1)
                probs = face_probs
                mode = 'face_only'
                
            elif audio_probs is not None:
                # Audio only
                pred_idx, conf = torch.max(audio_probs, dim=1)
                probs = audio_probs
                mode = 'audio_only'
                
            else:
                return {
                    'success': False,
                    'error': 'Failed to process both face and audio inputs'
                }
            
            # Build final result
            emotion = self.EMOTIONS[pred_idx.item() if isinstance(pred_idx, torch.Tensor) else pred_idx]
            confidence = conf.item() if isinstance(conf, torch.Tensor) else conf
            
            prob_dict = {
                emotion: float(probs[0][i].item())
                for i, emotion in enumerate(self.EMOTIONS)
            }
            
            return {
                'success': True,
                'emotion': emotion,
                'confidence': confidence,
                'probabilities': prob_dict,
                'face_emotion': face_result.get('emotion'),
                'face_confidence': face_result.get('confidence'),
                'audio_emotion': audio_result.get('emotion'),
                'audio_confidence': audio_result.get('confidence'),
                'mode': mode,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Fusion prediction failed: {str(e)}'
            }


# Global predictor instance
_predictor = None


def predict_fusion_emotion(face_path=None, audio_path=None,
                           face_model='models/face_model/best_model.pth',
                           audio_model='models/audio_model/best_model.pth',
                           fusion_model='models/fusion_model/best_model.pth'):
    """Convenience function for fusion emotion prediction.
    
    Args:
        face_path: Path to face image (optional)
        audio_path: Path to audio file (optional)
        face_model: Path to face model checkpoint
        audio_model: Path to audio model checkpoint
        fusion_model: Path to fusion model checkpoint
        
    Returns:
        dict: Prediction results
    """
    global _predictor
    
    if _predictor is None:
        _predictor = FusionEmotionPredictor(
            face_model_path=face_model,
            audio_model_path=audio_model,
            fusion_model_path=fusion_model
        )
    
    return _predictor.predict(face_path, audio_path)


def main():
    """CLI interface for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multimodal Fusion Emotion Prediction')
    parser.add_argument('--face', type=str, help='Path to face image')
    parser.add_argument('--audio', type=str, help='Path to audio file')
    parser.add_argument('--verbose', action='store_true', help='Show all details')
    
    args = parser.parse_args()
    
    if not args.face and not args.audio:
        parser.error('At least one of --face or --audio is required')
    
    # Predict
    result = predict_fusion_emotion(face_path=args.face, audio_path=args.audio)
    
    if result['success']:
        print(f"\n{'='*50}")
        print(f"Mode: {result['mode'].upper()}")
        print(f"{'='*50}")
        print(f"\n✓ Final Emotion: {result['emotion']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        
        if result['face_emotion']:
            print(f"\n  Face prediction: {result['face_emotion']} ({result['face_confidence']:.1%})")
        if result['audio_emotion']:
            print(f"  Audio prediction: {result['audio_emotion']} ({result['audio_confidence']:.1%})")
        
        if args.verbose:
            print("\n  Fusion probabilities:")
            for emotion, prob in sorted(result['probabilities'].items(),
                                       key=lambda x: x[1], reverse=True):
                print(f"    {emotion:<10} {prob:.1%}")
        
        print(f"\n{'='*50}\n")
    else:
        print(f"\n✗ Error: {result['error']}\n")


if __name__ == '__main__':
    main()