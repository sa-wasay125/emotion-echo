# EmotionEcho ML Backend Integration Guide

## 📦 Deliverables

You have **3 production-ready inference scripts** + **3 trained models**:

### Inference Scripts (ml/)
1. **`predict_face.py`** - Face emotion detection
2. **`predict_audio.py`** - Audio emotion detection  
3. **`predict_fusion.py`** - Multimodal fusion (recommended)

### Trained Models (ml/models/)
1. **`face_model/best_model.pth`** - ViT-based face model (100% test accuracy)
2. **`audio_model/best_model.pth`** - CNN audio model (54.6% validation)
3. **`fusion_model/best_model.pth`** - MLP fusion (97.3% validation)

---

## 🚀 Quick Start

### Installation

```bash
cd ml
pip install -r requirements.txt
```

### Test the Models

```bash
# Test face model
python predict_face.py path/to/face.jpg --verbose

# Test audio model  
python predict_audio.py path/to/audio.wav --verbose

# Test fusion (recommended)
python predict_fusion.py --face path/to/face.jpg --audio path/to/audio.wav --verbose
```

---

## 💻 Backend Integration

### Option 1: Direct Python Import (FastAPI)

```python
# In your FastAPI app
import sys
sys.path.insert(0, 'ml')  # Add ml/ to path

from predict_fusion import predict_fusion_emotion

@app.post("/api/detect-emotion")
async def detect_emotion(face_file: UploadFile, audio_file: UploadFile):
    # Save uploaded files temporarily
    face_path = save_temp_file(face_file)
    audio_path = save_temp_file(audio_file)
    
    # Predict (models load once, cached globally)
    result = predict_fusion_emotion(
        face_path=face_path,
        audio_path=audio_path
    )
    
    # Clean up temp files
    os.remove(face_path)
    os.remove(audio_path)
    
    if result['success']:
        return {
            "emotion": result['emotion'],
            "confidence": result['confidence'],
            "face_emotion": result['face_emotion'],
            "audio_emotion": result['audio_emotion']
        }
    else:
        raise HTTPException(status_code=400, detail=result['error'])
```

### Option 2: CLI Subprocess (Any Backend)

```python
import subprocess
import json

def predict_emotion(face_path, audio_path):
    cmd = [
        'python', 'ml/predict_fusion.py',
        '--face', face_path,
        '--audio', audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Parse stdout and return result
    return parse_output(result.stdout)
```

---

## 📊 API Response Format

### Fusion Prediction (Recommended)

```python
result = predict_fusion_emotion(face_path='face.jpg', audio_path='audio.wav')

# Returns:
{
    'success': True,
    'emotion': 'happy',                    # Final fused emotion
    'confidence': 0.97,                    # Fusion confidence (0-1)
    'probabilities': {                     # All emotion probabilities
        'happy': 0.97,
        'neutral': 0.02,
        'sad': 0.01,
        # ... (7 emotions total)
    },
    'face_emotion': 'happy',               # Face-only prediction
    'face_confidence': 0.99,
    'audio_emotion': 'happy',              # Audio-only prediction
    'audio_confidence': 0.85,
    'mode': 'fusion',                      # 'fusion', 'face_only', or 'audio_only'
    'error': None
}
```

### Face-Only Prediction

```python
result = predict_face_emotion('face.jpg')

# Returns:
{
    'success': True,
    'emotion': 'sad',
    'confidence': 0.92,
    'probabilities': {
        'sad': 0.92,
        'neutral': 0.04,
        # ... (7 emotions)
    },
    'error': None
}
```

### Audio-Only Prediction

```python
result = predict_audio_emotion('audio.wav')

# Returns:
{
    'success': True,
    'emotion': 'angry',
    'confidence': 0.84,
    'probabilities': {
        'angry': 0.84,
        'disgust': 0.10,
        # ... (8 emotions)
    },
    'error': None
}
```

---

## 🎯 Emotion Labels

### Face Model (7 emotions)
`['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']`

### Audio Model (8 emotions)
`['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']`

### Fusion Model (7 emotions - shared space)
`['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']`

**Note:** Fusion automatically remaps face + audio to a shared 7-emotion space.

---

## ⚙️ Configuration

### Model Paths

Default paths (relative to `ml/`):
```python
face_model_path = 'models/face_model/best_model.pth'
audio_model_path = 'models/audio_model/best_model.pth'
fusion_model_path = 'models/fusion_model/best_model.pth'
```

Override if needed:
```python
result = predict_fusion_emotion(
    face_path='face.jpg',
    audio_path='audio.wav',
    face_model='custom/path/face.pth',
    audio_model='custom/path/audio.pth',
    fusion_model='custom/path/fusion.pth'
)
```

### Device (CPU vs GPU)

Models default to CPU. For GPU:
```python
from predict_fusion import FusionEmotionPredictor

predictor = FusionEmotionPredictor(device='cuda')
result = predictor.predict(face_path='face.jpg', audio_path='audio.wav')
```

---

## 🔍 Error Handling

All prediction functions return `{'success': bool, 'error': str or None}`.

**Common errors:**
- `'Failed to preprocess image (no face detected)'` - No face in image
- `'Failed to preprocess audio (invalid file)'` - Corrupt/unsupported audio
- `'Prediction failed: ...'` - Model inference error

**Example error handling:**

```python
result = predict_fusion_emotion(face_path='face.jpg', audio_path='audio.wav')

if not result['success']:
    if 'no face detected' in result['error']:
        return {"error": "Please provide a clear face image"}
    elif 'invalid file' in result['error']:
        return {"error": "Audio file format not supported"}
    else:
        return {"error": "Processing failed"}
```

---

## 📝 File Format Support

### Face Input
- **Supported:** JPG, PNG, BMP (any OpenCV-readable format)
- **Requirements:** Must contain a detectable face
- **Preprocessing:** Auto face detection → crop → resize to 224×224

### Audio Input
- **Supported:** WAV, MP3, FLAC (any librosa-readable format)
- **Requirements:** Speech/voice audio (not music)
- **Preprocessing:** Resample to 16kHz → pad/trim to 3 seconds

---

## 🚨 Performance Notes

### Latency (CPU)
- Face prediction: ~200-500ms
- Audio prediction: ~100-300ms
- Fusion prediction: ~400-800ms (face + audio + fusion)

### Memory
- Face model: ~330MB RAM
- Audio model: ~10MB RAM
- Fusion model: ~1MB RAM
- **Total:** ~350MB for all three loaded

### Optimization Tips

1. **Keep models loaded** (don't reload per request)
   ```python
   # ✅ Good - loads once, reuses
   predictor = FusionEmotionPredictor()
   result1 = predictor.predict(face1, audio1)
   result2 = predictor.predict(face2, audio2)
   
   # ❌ Bad - reloads models every time
   result1 = predict_fusion_emotion(face1, audio1)  # loads models
   result2 = predict_fusion_emotion(face2, audio2)  # loads again!
   ```

2. **Use batch processing** if processing multiple files
3. **GPU acceleration** if available (CUDA) - 5-10x faster

---

## 🧪 Testing Checklist

- [ ] Face model works on sample images
- [ ] Audio model works on sample WAV files
- [ ] Fusion model combines both correctly
- [ ] Error handling for missing face
- [ ] Error handling for corrupt audio
- [ ] Response format matches your API spec
- [ ] Latency is acceptable (<1s for fusion)
- [ ] Memory usage is stable (no leaks)

---

## 📞 Support

**ML Model Questions:**
- Check `ml/ML_DOCUMENTATION.md` for training details
- Review `ml/ML_HANDOFF_GUIDE.md` for technical specs

**Integration Issues:**
- Verify `requirements.txt` dependencies installed
- Ensure model files exist in `ml/models/`
- Check file paths are relative to `ml/` directory

---

## ✅ Production Checklist

Before deploying:

1. **Dependencies:** All packages in `requirements.txt` installed
2. **Models:** All `.pth` files in correct directories
3. **Permissions:** Read access to `ml/models/` and `ml/preprocessing/`
4. **Testing:** All three prediction scripts tested with sample data
5. **Error handling:** Graceful failures for bad inputs
6. **Logging:** Log predictions for monitoring/debugging
7. **Rate limiting:** Prevent abuse (models are CPU-intensive)

---

## 🎯 Recommended Architecture

```
┌─────────────────────────────────────────┐
│         FastAPI Backend                 │
│                                         │
│  ┌─────────────────────────────────┐  │
│  │ Global Model Instances (cached) │  │
│  │ - FusionEmotionPredictor        │  │
│  │   (loads all 3 models once)     │  │
│  └─────────────────────────────────┘  │
│                                         │
│  POST /detect-emotion                  │
│  ├─ Save uploaded files (temp)         │
│  ├─ Call predictor.predict()           │
│  ├─ Return JSON response               │
│  └─ Clean up temp files                │
│                                         │
└─────────────────────────────────────────┘
           ↓ uses ↓
┌─────────────────────────────────────────┐
│         ML Package (ml/)                │
│                                         │
│  predict_fusion.py                      │
│  ├─ FaceEmotionPredictor                │
│  ├─ AudioEmotionPredictor               │
│  └─ FusionMLP                           │
│                                         │
│  models/                                │
│  ├─ face_model/best_model.pth           │
│  ├─ audio_model/best_model.pth          │
│  └─ fusion_model/best_model.pth         │
└─────────────────────────────────────────┘
```

**Good luck with integration! The ML models are ready to go.** 🚀