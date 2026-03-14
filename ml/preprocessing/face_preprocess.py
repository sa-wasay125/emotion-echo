"""
Face preprocessing module for EmotionEcho
Handles face detection, alignment, and normalization
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import torch
from torchvision import transforms
from PIL import Image

class FacePreprocessor:
    """Preprocessing pipeline for face emotion detection"""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        grayscale: bool = False,
        use_face_detection: bool = True
    ):
        """
        Args:
            target_size: Output image dimensions (H, W)
            grayscale: Convert to grayscale (for FER2013 compatibility)
            use_face_detection: Use OpenCV face detector
        """
        self.target_size = target_size
        self.grayscale = grayscale
        self.use_face_detection = use_face_detection
        
        # Load Haar Cascade for face detection
        if use_face_detection:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Transform pipeline for model input
        transform_list = [
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ]
        
        if grayscale:
            transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        else:
            transform_list.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ))
        
        self.transform = transforms.Compose(transform_list)
    
    def detect_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect and crop face from image
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Cropped face region or None if no face detected
        """
        if not self.use_face_detection:
            return image
        
        # Convert to grayscale for detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
        
        # Get largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        # Add padding (10%)
        padding = int(0.1 * max(w, h))
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        return image[y1:y2, x1:x2]
    
    def preprocess_image(self, image_path: str) -> Optional[torch.Tensor]:
        """
        Full preprocessing pipeline: load -> detect -> normalize -> tensor
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed tensor (C, H, W) or None if processing failed
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Detect and crop face
        face = self.detect_face(image)
        if face is None:
            # Fallback: use full image
            face = image
        
        # Convert to RGB/grayscale PIL Image
        if self.grayscale:
            if len(face.shape) == 3:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            pil_image = Image.fromarray(face).convert('L')
        else:
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_rgb)
        
        # Apply transforms
        tensor = self.transform(pil_image)
        
        return tensor
    
    def preprocess_batch(self, image_paths: list) -> torch.Tensor:
        """
        Process multiple images into a batch
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Batch tensor (N, C, H, W)
        """
        tensors = []
        for path in image_paths:
            tensor = self.preprocess_image(path)
            if tensor is not None:
                tensors.append(tensor)
        
        if len(tensors) == 0:
            raise ValueError("No valid images in batch")
        
        return torch.stack(tensors)
    
    def preprocess_from_array(self, image: np.ndarray) -> Optional[torch.Tensor]:
        """
        Preprocess image from numpy array (for video frames)
        
        Args:
            image: Input image array (BGR)
            
        Returns:
            Preprocessed tensor (C, H, W) or None
        """
        # Detect and crop face
        face = self.detect_face(image)
        if face is None:
            face = image
        
        # Convert to RGB/grayscale PIL Image
        if self.grayscale:
            if len(face.shape) == 3:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            pil_image = Image.fromarray(face).convert('L')
        else:
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_rgb)
        
        # Apply transforms
        tensor = self.transform(pil_image)
        
        return tensor


# Dataset preparation utilities
def prepare_fer2013_dataset(
    csv_path: str,
    output_dir: str,
    split: str = 'Training'
) -> None:
    """
    Convert FER2013 CSV to image files
    
    Args:
        csv_path: Path to fer2013.csv
        output_dir: Directory to save images
        split: 'Training', 'PublicTest', or 'PrivateTest'
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    df = df[df['Usage'] == split]
    
    emotion_map = {
        0: 'angry',
        1: 'disgust',
        2: 'fear',
        3: 'happy',
        4: 'sad',
        5: 'surprise',
        6: 'neutral'
    }
    
    output_path = Path(output_dir) / split.lower()
    
    for emotion_id, emotion_name in emotion_map.items():
        (output_path / emotion_name).mkdir(parents=True, exist_ok=True)
    
    for idx, row in df.iterrows():
        emotion = emotion_map[row['emotion']]
        pixels = np.array([int(p) for p in row['pixels'].split()], dtype=np.uint8)
        image = pixels.reshape(48, 48)
        
        save_path = output_path / emotion / f"{idx}.png"
        cv2.imwrite(str(save_path), image)
    
    print(f"Processed {len(df)} images for {split}")


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = FacePreprocessor(grayscale=False)
    
    # Example: preprocess single image
    # tensor = preprocessor.preprocess_image("path/to/image.jpg")
    # print(f"Preprocessed tensor shape: {tensor.shape}")