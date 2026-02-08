"""
Flask API Server for CCTV Detection Model
Handles video uploads and returns predictions from the PyTorch model
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import tempfile
import traceback
from torchvision import models, transforms
from PIL import Image

# Import new intelligent components
from SuspiciousLogicEngine import SuspiciousLogicEngine, AlertLevel
from CCTVPreprocessor import CCTVPreprocessor, PreprocessingConfig

# ============================================================================
# MODEL ARCHITECTURE - EXACT COPY from testingconvnext.py
# ============================================================================

from torchvision import models

class ConvNext_LSTM(nn.Module):
    """ConvNext + LSTM Model - EXACT COPY from testingconvnext.py"""
    def __init__(self, num_classes):
        super().__init__()
        convnext = models.convnext_tiny(pretrained=False)
        convnext.classifier = nn.Identity()
        self.cnn = convnext
        self.lstm = nn.LSTM(768, 256, batch_first=True)
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        feats = self.cnn(x)
        feats = feats.reshape(B, T, -1)
        lstm_out, _ = self.lstm(feats)
        last = lstm_out[:, -1, :]
        x = torch.relu(self.fc1(last))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = os.getenv('MODEL_PATH', 'cctv_convnext_lstm.pt')
SEQ_LEN = 16
IMG_SIZE = 224
NUM_CLASSES = 5
CLASS_NAMES = ['Emergency', 'Robbery', 'Trespassing', 'Violence', 'Weaponized']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# INTELLIGENT COMPONENTS INITIALIZATION
# ============================================================================

# Initialize Suspicious Logic Engine
suspicious_logic_engine = SuspiciousLogicEngine(
    class_names=CLASS_NAMES,
    confidence_threshold=0.65,
    high_confidence_threshold=0.85,
    temporal_buffer_size=5,
    stabilization_required=3,
    uncertain_threshold=0.50
)

# Initialize CCTV Preprocessor
preprocessing_config = PreprocessingConfig(
    target_size=(IMG_SIZE, IMG_SIZE),
    clahe_clip_limit=2.0,
    brightness_target=0.5,
    enable_stabilization=True
)
cctv_preprocessor = CCTVPreprocessor(preprocessing_config)

# ============================================================================
# MODEL LOADING
# ============================================================================

model = None

def load_model():
    """Load the trained model - EXACT COPY from working realTEST.PY"""
    global model
    if model is not None:
        return model
    
    print(f"Loading model from {MODEL_PATH}...")
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    # Load model - EXACT COPY from testingconvnext.py
    model = ConvNext_LSTM(len(CLASS_NAMES)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # Verify model is on correct device
    actual_device = next(model.parameters()).device
    print(f"Model loaded successfully on {actual_device}")
    return model


# ============================================================================
# VIDEO PROCESSING - Enhanced with CCTV Preprocessing
# ============================================================================

def extract_frames(video_path, seq_len=SEQ_LEN, use_preprocessing=True):
    """
    Extract and preprocess frames from video
    
    Args:
        video_path: Path to video file
        seq_len: Number of frames to extract
        use_preprocessing: Whether to use CCTV preprocessing (default: True)
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError("⚠️ No frames found in video")

    indices = np.linspace(0, total_frames - 1, seq_len, dtype=int)
    raw_frames = []  # Store raw BGR frames for preprocessing

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            raw_frames.append(frame.copy())
    cap.release()

    # Pad if video has fewer frames
    while len(raw_frames) < seq_len:
        raw_frames.append(raw_frames[-1] if raw_frames else np.zeros((480, 640, 3), dtype=np.uint8))

    # Apply CCTV preprocessing if enabled
    if use_preprocessing:
        # Reset stabilization state for new video
        cctv_preprocessor.reset_stabilization()
        
        # Preprocess all frames
        preprocessed_frames = cctv_preprocessor.preprocess_sequence(raw_frames, reset_stabilization=True)
        
        # Convert to tensor format
        frames_tensor = cctv_preprocessor.frames_to_tensor(preprocessed_frames, normalize=True)
    else:
        # Original processing (for backward compatibility)
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor()
        ])
        frames = []
        for frame in raw_frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = transform(img)
            frames.append(img)
        frames_tensor = torch.stack(frames)
    
    return frames_tensor


def predict_on_video(video_path, use_intelligent_logic=False, use_preprocessing=True):
    """
    Run inference on a video file with optional intelligent logic and preprocessing
    
    Args:
        video_path: Path to video file
        use_intelligent_logic: Whether to use SuspiciousLogicEngine (default: False - raw model only)
        use_preprocessing: Whether to use CCTV preprocessing (default: True)
    """
    model = load_model()
    
    # Ensure model is on the correct device and in eval mode
    device = next(model.parameters()).device
    model = model.to(device)
    model.eval()
    
    # Verify video file exists and can be opened
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    print(f"\n🎞️ Processing video: {video_path}")
    print(f"📋 Settings: Intelligent Logic={use_intelligent_logic}, Preprocessing={use_preprocessing}")
    
    # Extract and preprocess frames
    frames = extract_frames(video_path, use_preprocessing=use_preprocessing)
    frames = frames.unsqueeze(0).to(device)  # [1, T, C, H, W]
    
    # Debug: Print tensor stats
    print(f"Input tensor shape: {frames.shape}")
    print(f"Input tensor range: [{frames.min().item():.4f}, {frames.max().item():.4f}]")
    print(f"Input tensor mean: {frames.mean().item():.4f}")
    
    # Predict
    with torch.no_grad():
        outputs = model(frames)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
        confidence = float(probs[pred_idx])
    
    # Get all class probabilities
    class_probs = {}
    for i, class_name in enumerate(CLASS_NAMES):
        class_probs[class_name] = float(probs[i])
    
    # Debug: print raw probabilities
    print(f"\n📊 Raw Model Prediction: {CLASS_NAMES[pred_idx]} ({confidence*100:.2f}%)")
    for i, p in enumerate(probs):
        print(f"  {CLASS_NAMES[i]}: {p*100:.2f}%")
    
    # Apply intelligent suspicious logic
    if use_intelligent_logic:
        import time
        stabilized_result = suspicious_logic_engine.process_prediction(
            predicted_class=CLASS_NAMES[pred_idx],
            confidence=confidence,
            all_probabilities=class_probs,
            predicted_index=int(pred_idx),
            timestamp=time.time()
        )
        
        print(f"\n🧠 Intelligent Logic Result:")
        print(f"  Alert Level: {stabilized_result.alert_level.value}")
        print(f"  Stabilized Class: {stabilized_result.stabilized_class}")
        print(f"  Stabilized Confidence: {stabilized_result.stabilized_confidence*100:.2f}%")
        print(f"  Is Stable: {stabilized_result.is_stable}")
        print(f"  Vote Count: {stabilized_result.vote_count}/{stabilized_result.buffer_size}")
        
        # Return stabilized result
        return {
            'predicted_class': stabilized_result.stabilized_class,
            'predicted_index': CLASS_NAMES.index(stabilized_result.stabilized_class) if stabilized_result.stabilized_class in CLASS_NAMES else -1,
            'confidence': stabilized_result.stabilized_confidence,
            'all_probabilities': stabilized_result.all_probabilities,
            'alert_level': stabilized_result.alert_level.value,
            'is_stable': stabilized_result.is_stable,
            'vote_count': stabilized_result.vote_count,
            'raw_prediction': {
                'predicted_class': CLASS_NAMES[pred_idx],
                'confidence': confidence,
                'all_probabilities': class_probs
            }
        }
    else:
        # Return raw prediction (backward compatibility)
        # IMPORTANT: Only Emergency class has threshold filtering
        # All other classes (Robbery, Violence, Weaponized, Trespassing) are shown regardless of confidence
        sorted_indices = np.argsort(probs)[::-1]
        top_conf = probs[sorted_indices[0]]
        second_conf = probs[sorted_indices[1]] if len(sorted_indices) > 1 else 0
        diff = top_conf - second_conf
        
        print(f"Top confidence: {top_conf:.4f}, Second: {second_conf:.4f}, Difference: {diff:.4f}")
        
        # Safety check: ONLY filter low-confidence Emergency predictions
        # Emergency requires at least 0.95 (95%) confidence and 25% difference from second-best
        # All other classes pass through without threshold filtering
        if CLASS_NAMES[pred_idx] == 'Emergency':
            if confidence < 0.95 or diff < 0.25:
                print(f"⚠️  Low-confidence Emergency prediction filtered (conf={confidence:.2f}, diff={diff:.2f})")
                # Return as "Normal" (safe) - note: this is a fallback since there's no Normal class
                return {
                    'predicted_class': 'Normal',  # Fallback - indicates no threat detected
                    'predicted_index': -1,
                    'confidence': 1.0 - confidence,  # Inverted confidence
                    'all_probabilities': class_probs,
                    'warning': 'Low-confidence Emergency prediction filtered. Other classes are shown regardless of confidence.'
                }
        
        # For all other classes: return as-is, no threshold filtering
        return {
            'predicted_class': CLASS_NAMES[pred_idx],
            'predicted_index': int(pred_idx),
            'confidence': confidence,
            'all_probabilities': class_probs
        }


# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__)
# Enable CORS for frontend - allow all origins in development
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'device': str(DEVICE)})


@app.route('/predict', methods=['POST'])
def predict():
    """Handle video upload and return prediction"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get optional parameters (default: raw model only, with preprocessing)
        use_intelligent_logic = request.form.get('use_intelligent_logic', 'false').lower() == 'true'
        use_preprocessing = request.form.get('use_preprocessing', 'true').lower() == 'true'
        
        # Save uploaded file temporarily
        # Use Flask's save method which handles file uploads correctly
        file_ext = os.path.splitext(file.filename)[1] or '.mp4'
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            file.save(tmp_file.name)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
            temp_path = tmp_file.name
        
        # Verify file was saved and has content
        if not os.path.exists(temp_path):
            return jsonify({'error': 'Failed to save uploaded file'}), 500
        
        file_size = os.path.getsize(temp_path)
        if file_size == 0:
            os.unlink(temp_path)
            return jsonify({'error': 'Uploaded file is empty'}), 400
        
        print(f"Saved uploaded video: {temp_path} ({file_size} bytes)")
        
        try:
            # Run prediction with user's settings
            result = predict_on_video(temp_path, use_intelligent_logic=use_intelligent_logic, use_preprocessing=use_preprocessing)
            return jsonify(result)
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except Exception as e:
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500


@app.route('/classes', methods=['GET'])
def get_classes():
    """Get list of class names"""
    return jsonify({'classes': CLASS_NAMES})


@app.route('/test-file', methods=['POST'])
def test_file():
    """Test endpoint that accepts a file path instead of upload - for debugging"""
    try:
        data = request.get_json()
        if not data or 'file_path' not in data:
            return jsonify({'error': 'No file_path provided'}), 400
        
        file_path = data['file_path']
        if not os.path.exists(file_path):
            return jsonify({'error': f'File not found: {file_path}'}), 400
        
        # Get optional parameters (default: raw model only, with preprocessing)
        use_intelligent_logic = data.get('use_intelligent_logic', False)
        use_preprocessing = data.get('use_preprocessing', True)
        
        # Run prediction with user's settings
        result = predict_on_video(file_path, use_intelligent_logic=use_intelligent_logic, use_preprocessing=use_preprocessing)
        return jsonify(result)
    except Exception as e:
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500


@app.route('/predict-frames', methods=['POST'])
def predict_frames():
    """Handle frame sequence upload for real-time camera predictions"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get optional parameters (default: raw model only, with preprocessing)
        use_intelligent_logic = request.form.get('use_intelligent_logic', 'false').lower() == 'true'
        use_preprocessing = request.form.get('use_preprocessing', 'true').lower() == 'true'
        
        # Save uploaded file temporarily
        file_ext = os.path.splitext(file.filename)[1] or '.mp4'
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            file.save(tmp_file.name)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
            temp_path = tmp_file.name
        
        # Verify file was saved and has content
        if not os.path.exists(temp_path):
            return jsonify({'error': 'Failed to save uploaded file'}), 500
        
        file_size = os.path.getsize(temp_path)
        if file_size == 0:
            os.unlink(temp_path)
            return jsonify({'error': 'Uploaded file is empty'}), 400
        
        try:
            # Run prediction with user's settings
            result = predict_on_video(temp_path, use_intelligent_logic=use_intelligent_logic, use_preprocessing=use_preprocessing)
            return jsonify(result)
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except Exception as e:
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500


if __name__ == '__main__':
    print("=" * 80)
    print("🚀 Starting CCTV Detection API Server")
    print("=" * 80)
    print(f"Model path: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Classes: {CLASS_NAMES}")
    print("=" * 80)
    
    # Load model on startup
    try:
        load_model()
    except Exception as e:
        print(f"⚠️  Warning: Could not load model on startup: {e}")
        print("Model will be loaded on first request")
    
    # Run server
    app.run(host='0.0.0.0', port=5000, debug=False)

