"""
BrutPred Visualization Script

Loads model1 (event type classifier) and visualizes predictions on 100 unique WAV files.
Splits audio into 2-second chunks with 10% overlap, runs inference, and visualizes
both model predictions and ground truth annotations on the waveform.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import json
import torch
from typing import List, Tuple, Optional
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import model and config
from models import HeARClassifier
import config_model1

# Paths
CSV_PATH = PROJECT_ROOT / "data" / "SPRSound_Event_Level_Ensemble_Dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model1_event_type" / "best.pth"
OUTPUT_DIR = PROJECT_ROOT / "EventDetect" / "BrutPred" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Audio parameters
SAMPLE_RATE = 16000
CHUNK_DURATION = 2.0  # seconds
CHUNK_LENGTH_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)  # 32000
STRIDE_DURATION = 0.5  # seconds (daha sık prediction için küçük stride)
STRIDE_SAMPLES = int(STRIDE_DURATION * SAMPLE_RATE)  # 8000

# Class names and colors for predictions
CLASS_NAMES = config_model1.CLASS_NAMES  # ["Normal", "Crackles", "Wheeze/Rhonchi"]
PREDICTION_COLORS = {
    0: 'lightgray',  # Normal
    1: 'orange',     # Crackles
    2: 'red'         # Wheeze/Rhonchi
}

# Ground truth colors (different from predictions)
GT_COLORS = {
    'Normal': 'lightblue',
    'Fine Crackle': 'green',
    'Coarse Crackle': 'darkgreen',
    'Wheeze': 'purple',
    'Rhonchi': 'blue',
    'Wheeze+Crackle': 'magenta',
    'Stridor': 'cyan'
}


def load_model(device: torch.device) -> HeARClassifier:
    """Load model1 checkpoint."""
    print(f"Loading model from {MODEL_PATH}")
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    
    # Create model
    model = HeARClassifier(num_classes=config_model1.NUM_CLASSES)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Device: {device}")
    print(f"  Classes: {CLASS_NAMES}")
    
    return model


def find_json_path(wav_path: str, json_path_from_csv: Optional[str] = None) -> Optional[Path]:
    """Find corresponding JSON annotation file for a WAV file."""
    wav_path_obj = Path(wav_path)
    
    # First try path from CSV if provided
    if json_path_from_csv and Path(json_path_from_csv).exists():
        return Path(json_path_from_csv)
    
    # Try different patterns
    patterns = [
        wav_path_obj.with_suffix('.json'),
        wav_path_obj.parent.parent / (wav_path_obj.parent.name.replace('_wav', '_json')) / wav_path_obj.name.replace('.wav', '.json'),
        wav_path_obj.parent / wav_path_obj.name.replace('.wav', '.json'),
        Path(str(wav_path_obj).replace('train_classification_wav', 'train_classification_json').replace('.wav', '.json')),
        Path(str(wav_path_obj).replace('valid_classification_wav', 'valid_classification_json').replace('.wav', '.json')),
        Path(str(wav_path_obj).replace('test_classification_wav', 'test_classification_json').replace('.wav', '.json')),
    ]
    
    for pattern in patterns:
        if pattern.exists():
            return pattern
    
    return None


def load_ground_truth(json_path: Path) -> List[Tuple[float, float, str]]:
    """Load ground truth events from JSON file.
    
    Returns:
        List of (start_sec, end_sec, event_type) tuples
    """
    events = []
    
    if not json_path or not json_path.exists():
        return events
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        event_annotations = data.get('event_annotation', [])
        for event in event_annotations:
            start_ms = float(event.get('start', 0))
            end_ms = float(event.get('end', 0))
            event_type = event.get('type', 'Normal')
            
            # Convert to seconds
            start_sec = start_ms / 1000.0
            end_sec = end_ms / 1000.0
            
            events.append((start_sec, end_sec, event_type))
    except Exception as e:
        print(f"Warning: Could not load JSON {json_path}: {e}")
    
    return events


def predict_chunks(model: HeARClassifier, audio: np.ndarray, device: torch.device) -> List[Tuple[float, float, int]]:
    """Split audio into chunks and run inference.
    
    Returns:
        List of (start_sec, end_sec, predicted_class) tuples
    """
    predictions = []
    duration_sec = len(audio) / SAMPLE_RATE
    
    # Process chunks with 10% overlap
    for start_sample in range(0, len(audio) - CHUNK_LENGTH_SAMPLES + 1, STRIDE_SAMPLES):
        # Extract chunk
        chunk = audio[start_sample:start_sample + CHUNK_LENGTH_SAMPLES]
        
        # Pad if needed (shouldn't happen, but just in case)
        if len(chunk) < CHUNK_LENGTH_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_LENGTH_SAMPLES - len(chunk)), mode='constant')
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(chunk).float().unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            logits = model(audio_tensor)
            predicted_class = torch.argmax(logits, dim=1).item()
        
        # Calculate time range
        start_sec = start_sample / SAMPLE_RATE
        end_sec = (start_sample + CHUNK_LENGTH_SAMPLES) / SAMPLE_RATE
        
        predictions.append((start_sec, end_sec, predicted_class))
    
    # Handle last chunk if audio doesn't align perfectly
    remaining_samples = len(audio) % STRIDE_SAMPLES
    if remaining_samples > CHUNK_LENGTH_SAMPLES * 0.5:  # Only if at least 50% of chunk
        start_sample = len(audio) - CHUNK_LENGTH_SAMPLES
        chunk = audio[start_sample:start_sample + CHUNK_LENGTH_SAMPLES]
        
        if len(chunk) < CHUNK_LENGTH_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_LENGTH_SAMPLES - len(chunk)), mode='constant')
        
        audio_tensor = torch.from_numpy(chunk).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(audio_tensor)
            predicted_class = torch.argmax(logits, dim=1).item()
        
        start_sec = start_sample / SAMPLE_RATE
        end_sec = duration_sec
        
        predictions.append((start_sec, end_sec, predicted_class))
    
    return predictions


def visualize_predictions(
    audio: np.ndarray,
    predictions: List[Tuple[float, float, int]],
    ground_truth: List[Tuple[float, float, str]],
    filename: str,
    output_path: Path
):
    """Create visualization with waveform, predictions, and ground truth."""
    duration_sec = len(audio) / SAMPLE_RATE
    time_axis = np.linspace(0, duration_sec, len(audio))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Plot waveform
    ax.plot(time_axis, audio, color='black', linewidth=0.5, alpha=0.7, label='Waveform')
    
    # Add prediction regions (her 2 saniyelik chunk'ın kendi prediction'ı)
    # Overlap olduğu için alpha'yı düşük tutuyoruz
    for start_sec, end_sec, pred_class in predictions:
        color = PREDICTION_COLORS.get(pred_class, 'gray')
        ax.axvspan(start_sec, end_sec, facecolor=color, alpha=0.2, zorder=0)
    
    # Add ground truth annotations (overlay with different style)
    for start_sec, end_sec, event_type in ground_truth:
        color = GT_COLORS.get(event_type, 'yellow')
        ax.axvspan(start_sec, end_sec, facecolor=color, alpha=0.2, edgecolor=color, linewidth=2, zorder=1)
        
        # Add text label at center
        center_sec = (start_sec + end_sec) / 2.0
        center_ms = int(center_sec * 1000)
        duration_ms = int((end_sec - start_sec) * 1000)
        label_text = f"{event_type}\n{center_ms}ms ({duration_ms}ms)"
        
        # Position text above waveform
        y_pos = np.max(audio) * 0.9
        ax.text(center_sec, y_pos, label_text, 
                ha='center', va='bottom', fontsize=8, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                rotation=0, zorder=2)
    
    # Create custom legend
    from matplotlib.patches import Patch
    
    legend_elements = []
    
    # Prediction legend
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor='lightgray', alpha=0.3, label='Predictions:'))
    for class_idx, class_name in enumerate(CLASS_NAMES):
        color = PREDICTION_COLORS.get(class_idx, 'gray')
        legend_elements.append(Patch(facecolor=color, alpha=0.3, label=f'  {class_name}'))
    
    # Ground truth legend
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor='white', alpha=0, label='Ground Truth:'))
    unique_gt_types = set([gt[2] for gt in ground_truth])
    for event_type in sorted(unique_gt_types):
        color = GT_COLORS.get(event_type, 'yellow')
        legend_elements.append(Patch(facecolor=color, alpha=0.2, edgecolor=color, linewidth=2, label=f'  {event_type}'))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)
    
    # Labels and title
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title(f'{filename} - Model Predictions vs Ground Truth', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, duration_sec)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path.name}")


def main():
    """Main function."""
    print("="*80)
    print("BrutPred Visualization Script")
    print("="*80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load model
    print("\n" + "-"*80)
    print("Loading Model")
    print("-"*80)
    model = load_model(device)
    
    # Load CSV
    print("\n" + "-"*80)
    print("Loading Data")
    print("-"*80)
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")
    
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from CSV")
    
    # Get first 100 unique WAV files
    unique_wavs = df['wav_path'].drop_duplicates().head(100)
    print(f"Processing {len(unique_wavs)} unique WAV files")
    
    # Process each WAV file
    print("\n" + "-"*80)
    print("Processing WAV Files")
    print("-"*80)
    
    success_count = 0
    error_count = 0
    
    for idx, wav_path in enumerate(tqdm(unique_wavs, desc="Processing")):
        try:
            wav_path_obj = Path(wav_path)
            
            # Skip if WAV doesn't exist
            if not wav_path_obj.exists():
                print(f"\n[{idx+1}/{len(unique_wavs)}] Skipping (file not found): {wav_path}")
                error_count += 1
                continue
            
            filename = wav_path_obj.stem
            print(f"\n[{idx+1}/{len(unique_wavs)}] Processing: {filename}")
            
            # Load audio
            try:
                audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
            except Exception as e:
                print(f"  ✗ Error loading audio: {e}")
                error_count += 1
                continue
            
            duration_sec = len(audio) / sr
            print(f"  Audio duration: {duration_sec:.2f}s ({len(audio)} samples)")
            
            # Skip if too short
            if duration_sec < CHUNK_DURATION:
                print(f"  ⚠ Skipping (too short, need at least {CHUNK_DURATION}s)")
                error_count += 1
                continue
            
            # Find JSON path
            json_path_from_csv = df[df['wav_path'] == wav_path]['file_path'].iloc[0] if len(df[df['wav_path'] == wav_path]) > 0 else None
            json_path = find_json_path(wav_path, json_path_from_csv)
            
            # Load ground truth
            ground_truth = []
            if json_path:
                ground_truth = load_ground_truth(json_path)
                print(f"  Ground truth events: {len(ground_truth)}")
            else:
                print(f"  ⚠ No JSON file found (will show predictions only)")
            
            # Run inference
            print(f"  Running inference on {len(audio) // STRIDE_SAMPLES + 1} chunks...")
            predictions = predict_chunks(model, audio, device)
            print(f"  Predictions: {len(predictions)} chunks")
            
            # Count predictions by class
            pred_counts = {}
            for _, _, pred_class in predictions:
                pred_counts[pred_class] = pred_counts.get(pred_class, 0) + 1
            
            print(f"  Prediction distribution:")
            for class_idx, class_name in enumerate(CLASS_NAMES):
                count = pred_counts.get(class_idx, 0)
                percentage = 100 * count / len(predictions) if predictions else 0
                print(f"    {class_name}: {count} ({percentage:.1f}%)")
            
            # Visualize
            output_path = OUTPUT_DIR / f"{filename}_brutpred.png"
            visualize_predictions(audio, predictions, ground_truth, filename, output_path)
            
            success_count += 1
            
        except Exception as e:
            print(f"\n[{idx+1}/{len(unique_wavs)}] ✗ Error processing {wav_path}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
            continue
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Successfully processed: {success_count} files")
    print(f"Errors/Skipped: {error_count} files")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
