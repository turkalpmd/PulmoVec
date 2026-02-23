"""Event Detector integration for detecting events in raw audio.

Uses TensorFlow HeAR event detector to find event segments, then PyTorch Model 1
to classify event types (Normal, Crackles, Rhonchi).
"""

import numpy as np
import torch
import librosa
from pathlib import Path
from typing import List, Tuple, Dict, Union
import sys

# Add paths - IMPORTANT: src/ must be first so models.py imports src/config.py
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT / "src"))
# EventDetect after src so EventDetect/config doesn't shadow src/config
sys.path.insert(1, str(PROJECT_ROOT / "EventDetect"))

# Import model configs and classes (this will import src/config.py via models.py)
import config_model1  # Event type classification model
from models import HeARClassifier

# Import EventDetect config (after models.py to avoid conflicts)
# Use absolute import to avoid shadowing
import importlib.util
spec = importlib.util.spec_from_file_location("eventdetect_config", PROJECT_ROOT / "EventDetect" / "config.py")
eventdetect_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eventdetect_config)

# Use EventDetect config values
SAMPLE_RATE = eventdetect_config.SAMPLE_RATE
CLIP_DURATION = eventdetect_config.CLIP_DURATION
CLIP_OVERLAP_PERCENT = eventdetect_config.CLIP_OVERLAP_PERCENT
DETECTION_THRESHOLD = eventdetect_config.DETECTION_THRESHOLD

# Model 1 class names
MODEL1_CLASS_NAMES = ["Normal", "Crackles", "Rhonchi"]


def load_tensorflow_event_detector():
    """Load TensorFlow HeAR event detector from HuggingFace.
    
    Uses tf.saved_model.load() for Keras 3 compatibility with SavedModel format.
    Forces CPU execution to avoid GPU memory conflicts with PyTorch.
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError(
            "TensorFlow is required for HeAR event detector. "
            "Please install: pip install tensorflow"
        )
    
    from huggingface_hub import snapshot_download
    import os
    
    # Disable XLA JIT compilation to avoid JIT errors
    # Use CPU for TensorFlow to avoid conflicts with PyTorch
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
    tf.config.set_visible_devices([], 'GPU')  # Force CPU
    print("  TensorFlow will use CPU (XLA disabled)")
    
    # Try local snapshot first
    EVENT_DETECTOR_PATH = PROJECT_ROOT / "hear" / "event_detector" / "event_detector_small"
    
    if EVENT_DETECTOR_PATH.exists():
        print(f"Loading TensorFlow HeAR event detector from local snapshot: {EVENT_DETECTOR_PATH}")
        event_detector_path = str(EVENT_DETECTOR_PATH)
    else:
        # Download from HuggingFace if not found locally
        print("Local snapshot not found. Downloading from HuggingFace...")
        print("This may take a few minutes on first run...")
        
        # Download snapshot
        hugging_face_repo = "google/hear"
        local_snapshot_path = snapshot_download(repo_id=hugging_face_repo)
        print(f"Downloaded HeAR snapshot to: {local_snapshot_path}")
        
        # Load from snapshot - path should be relative to snapshot root
        event_detector_path = os.path.join(local_snapshot_path, "event_detector", "event_detector_small")
        print(f"Loading event detector from: {event_detector_path}")
    
    # Load SavedModel using tf.saved_model.load() for Keras 3 compatibility
    print("Loading SavedModel on CPU...")
    event_detector = tf.saved_model.load(event_detector_path)
    
    # The model should have a 'serving_default' signature
    # Wrap it to make it callable like a Keras model
    class EventDetectorWrapper:
        def __init__(self, saved_model):
            self.model = saved_model
            # Get signature name (string key)
            self.signature_name = None
            if 'serving_default' in saved_model.signatures:
                self.signature_name = 'serving_default'
            elif len(saved_model.signatures) > 0:
                # Use first available signature
                self.signature_name = list(saved_model.signatures.keys())[0]
            print(f"  Using signature: {self.signature_name}")
        
        def __call__(self, inputs):
            """Call the model with inputs.
            
            Args:
                inputs: TensorFlow tensor of shape [batch_size, 32000]
            
            Returns:
                Dictionary with 'scores' key containing detection scores [batch_size, 8]
            """
            if self.signature_name:
                # Call using signature name (string key)
                signature_func = self.model.signatures[self.signature_name]
                # The signature expects 'audio_wav' as input key
                result = signature_func(audio_wav=inputs)
                return result
            else:
                # Fallback: try direct call
                return self.model(inputs)
        
        def predict(self, inputs):
            """Alias for __call__ for compatibility."""
            return self.__call__(inputs)
    
    wrapped_detector = EventDetectorWrapper(event_detector)
    print("✓ TensorFlow event detector loaded (CPU mode)")
    
    return wrapped_detector


def load_pytorch_classifier():
    """Load PyTorch Model 1 for event type classification."""
    print(f"Loading PyTorch Model 1 (Event Type) from: {config_model1.BEST_MODEL_PATH}")
    
    if not config_model1.BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {config_model1.BEST_MODEL_PATH}. "
            f"Please train Model 1 first using: python scripts/train_ensemble_models.py --model model1"
        )
    
    # Clear GPU cache before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("  Cleared GPU cache")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint to CPU first, then move to GPU
    print(f"  Loading checkpoint to CPU first...")
    checkpoint = torch.load(config_model1.BEST_MODEL_PATH, map_location='cpu', weights_only=False)
    
    # Create model
    model = HeARClassifier(num_classes=config_model1.NUM_CLASSES)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device
    if device.type == 'cuda':
        print(f"  Moving model to GPU...")
        model = model.to(device)
        torch.cuda.empty_cache()  # Clear cache after moving
    
    model.eval()
    
    print("✓ PyTorch Model 1 (Event Type Classifier) loaded")
    print(f"  Device: {device}")
    print(f"  Classes: {MODEL1_CLASS_NAMES}")
    
    return model, device


def preprocess_audio_clip(audio: np.ndarray, start_sec: float, end_sec: float) -> torch.Tensor:
    """
    Preprocess audio clip for PyTorch model (same as training).
    
    Args:
        audio: Full audio array
        start_sec: Start time in seconds
        end_sec: End time in seconds
        
    Returns:
        audio_tensor: (32000,) tensor ready for model
    """
    start_sample = int(start_sec * SAMPLE_RATE)
    end_sample = int(end_sec * SAMPLE_RATE)
    
    # Extract clip
    clip = audio[start_sample:end_sample]
    
    # Pad or trim to exactly 2 seconds (32000 samples)
    target_length = int(CLIP_DURATION * SAMPLE_RATE)
    if len(clip) < target_length:
        clip = np.pad(clip, (0, target_length - len(clip)), mode='constant')
    elif len(clip) > target_length:
        clip = clip[:target_length]
    
    # Convert to tensor
    audio_tensor = torch.from_numpy(clip).float()
    
    return audio_tensor


def detect_event_segments_pytorch_model1(
    audio_path: str,
    pytorch_model=None,
    device=None,
    threshold: float = None
) -> Tuple[List[Tuple[float, float]], np.ndarray, Dict]:
    """
    Detect event segments using PyTorch Model 1 (Event Type: Normal, Crackles, Rhonchi).
    Normal olmayan her şey event olarak kabul edilir.
    
    Args:
        audio_path: Path to WAV file
        pytorch_model: Pre-loaded PyTorch Model 1 (optional)
        device: PyTorch device (optional)
        threshold: Detection threshold for non-Normal class probability (default: 0.3)
        
    Returns:
        segments: List of (start_sec, end_sec) tuples for detected events
        detection_scores: Array of detection scores [num_clips, 3] (Normal, Crackles, Rhonchi)
        detection_results: Detailed detection results dictionary
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import config_model1
    from models import HeARClassifier
    
    if threshold is None:
        threshold = DETECTION_THRESHOLD
    
    # Load PyTorch Model 1 if not provided
    if pytorch_model is None:
        print(f"Loading PyTorch Model 1 (Event Type) from: {config_model1.BEST_MODEL_PATH}")
        
        if not config_model1.BEST_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {config_model1.BEST_MODEL_PATH}. "
                f"Please train Model 1 first using: python scripts/train_ensemble_models.py --model model1"
            )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pytorch_model = HeARClassifier(num_classes=config_model1.NUM_CLASSES)
        
        checkpoint = torch.load(config_model1.BEST_MODEL_PATH, map_location='cpu', weights_only=False)
        pytorch_model.load_state_dict(checkpoint['model_state_dict'])
        pytorch_model.to(device)
        pytorch_model.eval()
        print("✓ PyTorch Model 1 loaded")
        print(f"  Classes: {MODEL1_CLASS_NAMES}")
    elif device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    duration_sec = len(audio) / sr
    
    print(f"Loaded audio: {duration_sec:.2f}s, {len(audio)} samples")
    
    # Create overlapping clips
    frame_length = int(CLIP_DURATION * SAMPLE_RATE)  # 32000 samples
    frame_step = int(frame_length * (1 - CLIP_OVERLAP_PERCENT / 100))  # 28800 samples (10% overlap)
    
    # Generate clip start times
    clip_start_times = []
    i = 0
    while i * frame_step < len(audio):
        start_sample = i * frame_step
        start_sec = start_sample / SAMPLE_RATE
        end_sec = min(start_sec + CLIP_DURATION, duration_sec)
        
        if end_sec - start_sec >= CLIP_DURATION * 0.5:  # Only include clips with at least 50% duration
            clip_start_times.append(start_sec)
        i += 1
    
    num_clips = len(clip_start_times)
    print(f"Created {num_clips} overlapping clips (overlap: {CLIP_OVERLAP_PERCENT}%)")
    
    # Run PyTorch Model 2 on each clip
    print("Running PyTorch Model 2 (Binary) inference...")
    detection_scores = []
    
    with torch.no_grad():
        for clip_idx, start_sec in enumerate(clip_start_times):
            end_sec = min(start_sec + CLIP_DURATION, duration_sec)
            
            # Preprocess clip
            audio_tensor = preprocess_audio_clip(audio, start_sec, end_sec)
            audio_tensor = audio_tensor.unsqueeze(0).to(device)  # [1, 32000]
            
            # Get prediction
            logits = pytorch_model(audio_tensor)  # [1, 2]
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # [2] (Normal, Abnormal)
            detection_scores.append(probs)
    
    detection_scores = np.array(detection_scores)  # [num_clips, 3]
    print(f"Detection scores shape: {detection_scores.shape}")
    
    # Use non-Normal class probabilities (Crackles + Rhonchi) for event detection
    normal_probs = detection_scores[:, 0]  # [num_clips] - Normal
    crackles_probs = detection_scores[:, 1]  # [num_clips] - Crackles
    rhonchi_probs = detection_scores[:, 2]  # [num_clips] - Rhonchi
    
    # Event probability = max of (Crackles, Rhonchi) or sum
    event_probs = np.maximum(crackles_probs, rhonchi_probs)  # [num_clips]
    # Alternative: event_probs = crackles_probs + rhonchi_probs
    
    # Apply threshold to identify event clips (non-Normal)
    event_clips = event_probs >= threshold
    
    # Store detailed clip information for JSON export
    clip_details = []
    for i, (clip_start, is_event, normal_prob, crackles_prob, rhonchi_prob, event_prob) in enumerate(
        zip(clip_start_times, event_clips, normal_probs, crackles_probs, rhonchi_probs, event_probs)
    ):
        clip_end = min(clip_start + CLIP_DURATION, duration_sec)
        
        # Get predicted class
        clip_scores = detection_scores[i]
        pred_class_idx = int(np.argmax(clip_scores))
        pred_class = MODEL1_CLASS_NAMES[pred_class_idx]
        
        clip_details.append({
            'clip_index': i,
            'start_sec': float(clip_start),
            'end_sec': float(clip_end),
            'duration_sec': float(clip_end - clip_start),
            'is_event': bool(is_event),
            'predicted_class': pred_class,
            'event_probability': float(event_prob),
            'detection_scores': {
                'Normal': float(normal_prob),
                'Crackles': float(crackles_prob),
                'Rhonchi': float(rhonchi_prob)
            },
            'max_probability': float(np.max(clip_scores))
        })
    
    # Very conservative segment merging: each event clip is a separate segment
    # Only merge if clips are DIRECTLY adjacent (no gap) and have same predicted class
    segments = []
    in_segment = False
    segment_start = None
    segment_predicted_class = None
    
    MAX_GAP_SECONDS = 0.1  # Very small gap - only merge directly adjacent clips
    MIN_CONFIDENCE_FOR_MERGE = 0.6  # Higher confidence required for merging
    
    for i, is_event in enumerate(event_clips):
        clip_start = clip_start_times[i]
        clip_end = min(clip_start + CLIP_DURATION, duration_sec)
        event_prob = event_probs[i]
        
        # Get predicted class for this clip
        clip_scores = detection_scores[i]
        pred_class_idx = int(np.argmax(clip_scores))
        pred_class = MODEL1_CLASS_NAMES[pred_class_idx]
        
        if is_event:
            if not in_segment:
                # Start new segment
                segment_start = clip_start
                segment_predicted_class = pred_class
                in_segment = True
            else:
                # Already in segment - check if we should continue or split
                # Only continue if:
                # 1. Same predicted class
                # 2. High confidence
                # 3. Directly adjacent (no gap)
                prev_clip_end = min(clip_start_times[i-1] + CLIP_DURATION, duration_sec)
                gap = clip_start - prev_clip_end
                
                if (pred_class == segment_predicted_class and 
                    event_prob >= MIN_CONFIDENCE_FOR_MERGE and 
                    gap <= MAX_GAP_SECONDS):
                    # Continue segment - same class, high confidence, no gap
                    pass
                else:
                    # Different class, low confidence, or gap - end previous segment
                    segments.append((segment_start, prev_clip_end))
                    segment_start = clip_start
                    segment_predicted_class = pred_class
        else:
            # No event in this clip - end segment if we're in one
            if in_segment:
                prev_clip_end = min(clip_start_times[i-1] + CLIP_DURATION, duration_sec)
                segments.append((segment_start, prev_clip_end))
                in_segment = False
    
    # Handle segment that extends to end of audio
    if in_segment:
        last_clip_end = min(clip_start_times[-1] + CLIP_DURATION, duration_sec)
        segments.append((segment_start, last_clip_end))
    
    print(f"Found {len(segments)} event segments")
    for i, (start, end) in enumerate(segments):
        print(f"  Segment {i+1}: {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
    
    # Create detailed detection results for JSON export
    detection_results = {
        'audio_path': audio_path,
        'audio_duration_sec': float(duration_sec),
        'num_clips': int(num_clips),
        'clip_duration_sec': float(CLIP_DURATION),
        'clip_overlap_percent': float(CLIP_OVERLAP_PERCENT),
        'detection_threshold': float(threshold),
        'num_segments_found': len(segments),
        'segments': [
            {
                'segment_index': idx,
                'start_sec': float(start),
                'end_sec': float(end),
                'duration_sec': float(end - start)
            }
            for idx, (start, end) in enumerate(segments)
        ],
        'clip_details': clip_details,
        'detection_statistics': {
            'total_clips': int(num_clips),
            'event_clips': int(np.sum(event_clips)),
            'normal_clips': int(np.sum(~event_clips)),
            'mean_event_prob': float(np.mean(event_probs)),
            'max_event_prob': float(np.max(event_probs)),
            'min_event_prob': float(np.min(event_probs)),
            'mean_normal_prob': float(np.mean(normal_probs)),
            'mean_crackles_prob': float(np.mean(crackles_probs)),
            'mean_rhonchi_prob': float(np.mean(rhonchi_probs)),
            'class_distribution': {
                'Normal': int(np.sum([cd['predicted_class'] == 'Normal' for cd in clip_details])),
                'Crackles': int(np.sum([cd['predicted_class'] == 'Crackles' for cd in clip_details])),
                'Rhonchi': int(np.sum([cd['predicted_class'] == 'Rhonchi' for cd in clip_details]))
            }
        }
    }
    
    return segments, detection_scores, detection_results


def detect_event_segments_only(
    audio_path: str,
    tf_event_detector=None,
    threshold: float = None
) -> Tuple[List[Tuple[float, float]], np.ndarray, Dict]:
    """
    Detect event segments using TensorFlow HeAR event detector ONLY.
    No classification - just segmentation.
    
    Args:
        audio_path: Path to WAV file
        tf_event_detector: Pre-loaded TensorFlow event detector (optional)
        threshold: Detection threshold for event presence (default: DETECTION_THRESHOLD)
        
    Returns:
        segments: List of (start_sec, end_sec) tuples for detected events
        detection_scores: Array of TensorFlow detection scores [num_clips, 8]
    """
    if threshold is None:
        threshold = DETECTION_THRESHOLD
    
    # Load TensorFlow detector if not provided
    if tf_event_detector is None:
        tf_event_detector = load_tensorflow_event_detector()
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    duration_sec = len(audio) / sr
    
    print(f"Loaded audio: {duration_sec:.2f}s, {len(audio)} samples")
    
    # Use TensorFlow event detector to find event segments
    import tensorflow as tf
    
    frame_length = int(CLIP_DURATION * SAMPLE_RATE)  # 32000 samples
    frame_step = int(frame_length * (1 - CLIP_OVERLAP_PERCENT / 100))  # 28800 samples (10% overlap)
    
    # Pad if necessary
    if len(audio) < frame_length:
        audio = np.pad(audio, (0, frame_length - len(audio)), mode='constant')
    
    # Create overlapping clips using TensorFlow
    audio_clip_batch = tf.signal.frame(audio, frame_length, frame_step)
    num_clips = len(audio_clip_batch)
    
    print(f"Created {num_clips} overlapping clips (overlap: {CLIP_OVERLAP_PERCENT}%)")
    
    # Run TensorFlow event detector
    # Model expects input shape [1, 32000] for each clip, so process one at a time
    print("Running TensorFlow event detector inference...")
    detection_scores_list = []
    
    for clip_idx in range(num_clips):
        # Get single clip: shape [32000]
        clip = audio_clip_batch[clip_idx]  # [32000]
        # Add batch dimension: [1, 32000]
        clip_batch = tf.expand_dims(tf.cast(clip, tf.float32), axis=0)
        
        # Call the wrapped detector with single clip
        detector_output = tf_event_detector(clip_batch)
        
        # Handle different output formats
        if isinstance(detector_output, dict):
            # Output format: {'mobilenetv3_small_model': tensor}
            scores = list(detector_output.values())[0].numpy()  # [1, 8]
            detection_scores_list.append(scores[0])  # [8]
        else:
            # Direct tensor output
            scores = detector_output.numpy()  # [1, 8]
            detection_scores_list.append(scores[0])  # [8]
    
    # Stack all scores: [num_clips, 8]
    detection_scores_batch = np.array(detection_scores_list)
    print(f"TensorFlow detection scores shape: {detection_scores_batch.shape}")
    
    # Aggregate scores: max across all 8 labels to get binary "event present" signal
    max_scores = np.max(detection_scores_batch, axis=1)  # [num_clips]
    
    # Apply threshold to identify event clips
    event_clips = max_scores >= threshold
    
    # Convert clip indices to time segments
    clip_start_times = []
    for i in range(num_clips):
        clip_start_sec = i * (frame_step / SAMPLE_RATE)
        clip_start_times.append(clip_start_sec)
    
    # Merge adjacent positive clips into continuous segments
    raw_segments = []
    in_segment = False
    segment_start = None
    
    for i, is_event in enumerate(event_clips):
        clip_start = clip_start_times[i]
        clip_end = clip_start + CLIP_DURATION
        
        if is_event and not in_segment:
            # Start new segment
            segment_start = clip_start
            in_segment = True
        elif not is_event and in_segment:
            # End segment
            raw_segments.append((segment_start, clip_end))
            in_segment = False
        elif is_event and in_segment:
            # Continue segment
            pass
    
    # Handle segment that extends to end of audio
    if in_segment:
        raw_segments.append((segment_start, min(segment_start + CLIP_DURATION, duration_sec)))
    
    print(f"Found {len(raw_segments)} event segments")
    for i, (start, end) in enumerate(raw_segments):
        print(f"  Segment {i+1}: {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
    
    return raw_segments, detection_scores_batch


def detect_events_with_classification(
    audio_path: str, 
    tf_event_detector=None,
    pytorch_classifier=None,
    device=None,
    threshold: float = None
) -> Tuple[List[Dict], np.ndarray]:
    """
    Detect event segments using TensorFlow HeAR event detector, then classify
    each segment using PyTorch Model 1.
    
    Args:
        audio_path: Path to WAV file
        tf_event_detector: Pre-loaded TensorFlow event detector (optional)
        pytorch_classifier: Pre-loaded PyTorch Model 1 (optional)
        device: PyTorch device (optional)
        threshold: Detection threshold for event presence (default: DETECTION_THRESHOLD)
        
    Returns:
        segments: List of dicts with keys: 'start', 'end', 'event_type', 'confidence'
        detection_scores: Array of TensorFlow detection scores [num_clips, 8]
    """
    if threshold is None:
        threshold = DETECTION_THRESHOLD
    
    # Load models if not provided
    if tf_event_detector is None:
        tf_event_detector = load_tensorflow_event_detector()
    
    if pytorch_classifier is None:
        pytorch_classifier, device = load_pytorch_classifier()
    elif device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    duration_sec = len(audio) / sr
    
    print(f"Loaded audio: {duration_sec:.2f}s, {len(audio)} samples")
    
    # Step 1: Use TensorFlow event detector to find event segments
    import tensorflow as tf
    
    frame_length = int(CLIP_DURATION * SAMPLE_RATE)  # 32000 samples
    frame_step = int(frame_length * (1 - CLIP_OVERLAP_PERCENT / 100))  # 28800 samples (10% overlap)
    
    # Pad if necessary
    if len(audio) < frame_length:
        audio = np.pad(audio, (0, frame_length - len(audio)), mode='constant')
    
    # Create overlapping clips using TensorFlow
    audio_clip_batch = tf.signal.frame(audio, frame_length, frame_step)
    num_clips = len(audio_clip_batch)
    
    print(f"Created {num_clips} overlapping clips (overlap: {CLIP_OVERLAP_PERCENT}%)")
    
    # Run TensorFlow event detector
    # Model expects input shape [1, 32000] for each clip, so process one at a time
    print("Running TensorFlow event detector inference...")
    detection_scores_list = []
    
    for clip_idx in range(num_clips):
        # Get single clip: shape [32000]
        clip = audio_clip_batch[clip_idx]  # [32000]
        # Add batch dimension: [1, 32000]
        clip_batch = tf.expand_dims(tf.cast(clip, tf.float32), axis=0)
        
        # Call the wrapped detector with single clip
        detector_output = tf_event_detector(clip_batch)
        
        # Handle different output formats
        if isinstance(detector_output, dict):
            # Output format: {'mobilenetv3_small_model': tensor}
            scores = list(detector_output.values())[0].numpy()  # [1, 8]
            detection_scores_list.append(scores[0])  # [8]
        else:
            # Direct tensor output
            scores = detector_output.numpy()  # [1, 8]
            detection_scores_list.append(scores[0])  # [8]
    
    # Stack all scores: [num_clips, 8]
    detection_scores_batch = np.array(detection_scores_list)
    print(f"TensorFlow detection scores shape: {detection_scores_batch.shape}")
    
    # Aggregate scores: max across all 8 labels to get binary "event present" signal
    max_scores = np.max(detection_scores_batch, axis=1)  # [num_clips]
    
    # Apply threshold to identify event clips
    event_clips = max_scores >= threshold
    
    # Convert clip indices to time segments
    clip_start_times = []
    for i in range(num_clips):
        clip_start_sec = i * (frame_step / SAMPLE_RATE)
        clip_start_times.append(clip_start_sec)
    
    # Merge adjacent positive clips into continuous segments
    raw_segments = []
    in_segment = False
    segment_start = None
    
    for i, is_event in enumerate(event_clips):
        clip_start = clip_start_times[i]
        clip_end = clip_start + CLIP_DURATION
        
        if is_event and not in_segment:
            # Start new segment
            segment_start = clip_start
            in_segment = True
        elif not is_event and in_segment:
            # End segment
            raw_segments.append((segment_start, clip_end))
            in_segment = False
        elif is_event and in_segment:
            # Continue segment
            pass
    
    # Handle segment that extends to end of audio
    if in_segment:
        raw_segments.append((segment_start, min(segment_start + CLIP_DURATION, duration_sec)))
    
    print(f"Found {len(raw_segments)} event segments using TensorFlow detector")
    
    # Step 2: Classify each segment using PyTorch Model 1
    print("Classifying segments using PyTorch Model 1...")
    classified_segments = []
    
    with torch.no_grad():
        for seg_idx, (start_sec, end_sec) in enumerate(raw_segments):
            # Preprocess segment for PyTorch model
            audio_tensor = preprocess_audio_clip(audio, start_sec, end_sec)
            audio_tensor = audio_tensor.unsqueeze(0).to(device)  # [1, 32000]
            
            # Get event type prediction
            logits = pytorch_classifier(audio_tensor)  # [1, 3]
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # [3] (Normal, Crackles, Rhonchi)
            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class])
            event_type = MODEL1_CLASS_NAMES[pred_class]
            
            classified_segments.append({
                'start': start_sec,
                'end': end_sec,
                'event_type': event_type,
                'event_type_idx': pred_class,
                'confidence': confidence,
                'probabilities': {
                    'Normal': float(probs[0]),
                    'Crackles': float(probs[1]),
                    'Rhonchi': float(probs[2])
                }
            })
            
            print(f"  Segment {seg_idx+1}: {start_sec:.2f}s - {end_sec:.2f}s → {event_type} (conf: {confidence:.3f})")
    
    return classified_segments, detection_scores_batch


# Backward compatibility: keep old function name
def detect_events(audio_path: str, tf_event_detector=None, pytorch_classifier=None, device=None, threshold: float = None):
    """Alias for detect_events_with_classification for backward compatibility."""
    return detect_events_with_classification(audio_path, tf_event_detector, pytorch_classifier, device, threshold)


if __name__ == "__main__":
    # Test detection
    import json
    SELECTED_SAMPLES_JSON = PROJECT_ROOT / "EventDetect" / "samples" / "selected_samples.json"
    
    if not SELECTED_SAMPLES_JSON.exists():
        print(f"Selected samples file not found: {SELECTED_SAMPLES_JSON}")
        print("Please run: python EventDetect/scripts/select_samples.py")
    else:
        with open(SELECTED_SAMPLES_JSON, 'r') as f:
            samples = json.load(f)
        
        # Load models once
        tf_detector = load_tensorflow_event_detector()
        pytorch_model, device = load_pytorch_classifier()
        
        for event_group, sample_info in samples.items():
            print(f"\n{'='*60}")
            print(f"Processing: {event_group}")
            print(f"{'='*60}")
            segments, scores = detect_events_with_classification(
                sample_info['wav_path'], 
                tf_event_detector=tf_detector,
                pytorch_classifier=pytorch_model,
                device=device
            )
