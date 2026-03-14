"""
Audio preprocessing module for EmotionEcho
Handles audio loading, resampling, feature extraction
"""

import numpy as np
import librosa
import torch
import torchaudio
from pathlib import Path
from typing import Tuple, Optional, Union

class AudioPreprocessor:
    """Preprocessing pipeline for audio emotion detection"""
    
    def __init__(
        self,
        target_sr: int = 16000,
        duration: Optional[float] = None,
        use_augmentation: bool = False
    ):
        """
        Args:
            target_sr: Target sampling rate (16kHz for Wav2Vec2)
            duration: Fixed duration in seconds (None = variable length)
            use_augmentation: Apply data augmentation during training
        """
        self.target_sr = target_sr
        self.duration = duration
        self.use_augmentation = use_augmentation
        
        # Torchaudio resampler (lazy initialization)
        self._resampler = None
    
    def load_audio(
        self,
        audio_path: str,
        offset: float = 0.0,
        duration: Optional[float] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file using librosa
        
        Args:
            audio_path: Path to audio file
            offset: Start position in seconds
            duration: Duration to load (None = full file)
            
        Returns:
            Tuple of (waveform, sample_rate)
        """
        audio, sr = librosa.load(
            audio_path,
            sr=None,  # Keep original sample rate
            offset=offset,
            duration=duration
        )
        return audio, sr
    
    def resample(
        self,
        waveform: Union[np.ndarray, torch.Tensor],
        orig_sr: int
    ) -> torch.Tensor:
        """
        Resample audio to target sample rate
        
        Args:
            waveform: Input audio
            orig_sr: Original sample rate
            
        Returns:
            Resampled waveform as torch tensor
        """
        if orig_sr == self.target_sr:
            if isinstance(waveform, np.ndarray):
                return torch.from_numpy(waveform).float()
            return waveform.float()
        
        # Convert to torch if needed
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()
        
        # Lazy initialization of resampler
        if self._resampler is None or self._resampler.orig_freq != orig_sr:
            self._resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sr,
                new_freq=self.target_sr
            )
        
        return self._resampler(waveform)
    
    def pad_or_trim(
        self,
        waveform: torch.Tensor,
        target_length: int
    ) -> torch.Tensor:
        """
        Pad or trim waveform to target length
        
        Args:
            waveform: Input waveform
            target_length: Target number of samples
            
        Returns:
            Fixed-length waveform
        """
        current_length = waveform.shape[-1]
        
        if current_length < target_length:
            # Pad with zeros
            padding = target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif current_length > target_length:
            # Trim from center
            start = (current_length - target_length) // 2
            waveform = waveform[..., start:start + target_length]
        
        return waveform
    
    def augment_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply audio augmentation (for training)
        
        Args:
            waveform: Input waveform
            
        Returns:
            Augmented waveform
        """
        if not self.use_augmentation:
            return waveform
        
        # Random volume adjustment
        if np.random.rand() < 0.5:
            volume_factor = np.random.uniform(0.8, 1.2)
            waveform = waveform * volume_factor
        
        # Random noise injection
        if np.random.rand() < 0.3:
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise
        
        return waveform
    
    def preprocess_audio(
        self,
        audio_path: str,
        augment: bool = False
    ) -> torch.Tensor:
        """
        Full preprocessing pipeline: load -> resample -> normalize
        
        Args:
            audio_path: Path to audio file
            augment: Apply augmentation
            
        Returns:
            Preprocessed waveform tensor (1, T)
        """
        # Load audio
        waveform, sr = self.load_audio(audio_path, duration=self.duration)
        
        # Resample
        waveform = self.resample(waveform, sr)
        
        # Apply fixed duration if specified
        if self.duration is not None:
            target_length = int(self.duration * self.target_sr)
            waveform = self.pad_or_trim(waveform, target_length)
        
        # Augmentation
        if augment:
            waveform = self.augment_audio(waveform)
        
        # Normalize
        waveform = waveform / (torch.abs(waveform).max() + 1e-8)
        
        # Add batch dimension if needed
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        return waveform
    
    def extract_mel_spectrogram(
        self,
        waveform: torch.Tensor,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 512
    ) -> torch.Tensor:
        """
        Extract mel spectrogram features (alternative to raw waveform)
        
        Args:
            waveform: Input waveform
            n_mels: Number of mel bands
            n_fft: FFT size
            hop_length: Hop length
            
        Returns:
            Mel spectrogram (1, n_mels, T)
        """
        mel_spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
        mel_spec = mel_spec_transform(waveform)
        
        # Convert to log scale
        mel_spec = torch.log(mel_spec + 1e-9)
        
        return mel_spec
    
    def preprocess_batch(
        self,
        audio_paths: list,
        augment: bool = False
    ) -> torch.Tensor:
        """
        Process multiple audio files into a batch
        
        Args:
            audio_paths: List of audio file paths
            augment: Apply augmentation
            
        Returns:
            Batch tensor (N, T) or (N, 1, T)
        """
        waveforms = []
        for path in audio_paths:
            waveform = self.preprocess_audio(path, augment=augment)
            waveforms.append(waveform)
        
        # Stack into batch (handle variable lengths if needed)
        if self.duration is not None:
            return torch.stack([w.squeeze(0) for w in waveforms])
        else:
            # Pad to max length in batch
            max_length = max(w.shape[-1] for w in waveforms)
            padded = [self.pad_or_trim(w.squeeze(0), max_length) for w in waveforms]
            return torch.stack(padded)


# Dataset utilities
def parse_ravdess_filename(filename: str) -> dict:
    """
    Parse RAVDESS filename to extract metadata
    
    Filename format: modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
    
    Args:
        filename: RAVDESS filename
        
    Returns:
        Dictionary with metadata
    """
    parts = Path(filename).stem.split('-')
    
    emotion_map = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    
    return {
        'modality': parts[0],
        'vocal_channel': parts[1],
        'emotion': emotion_map.get(parts[2], 'unknown'),
        'emotion_id': parts[2],
        'intensity': parts[3],
        'statement': parts[4],
        'repetition': parts[5],
        'actor': parts[6]
    }


def prepare_ravdess_dataset(data_dir: str) -> list:
    """
    Scan RAVDESS directory and create dataset list
    
    Args:
        data_dir: Root directory of RAVDESS dataset
        
    Returns:
        List of dictionaries with file paths and labels
    """
    data_list = []
    data_path = Path(data_dir)
    
    for audio_file in data_path.rglob("*.wav"):
        metadata = parse_ravdess_filename(audio_file.name)
        data_list.append({
            'path': str(audio_file),
            'emotion': metadata['emotion'],
            'actor': metadata['actor'],
            'intensity': metadata['intensity']
        })
    
    return data_list


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = AudioPreprocessor(target_sr=16000, duration=3.0)
    
    # Example: preprocess single audio
    # waveform = preprocessor.preprocess_audio("path/to/audio.wav")
    # print(f"Preprocessed waveform shape: {waveform.shape}")