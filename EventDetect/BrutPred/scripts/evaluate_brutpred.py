"""
BrutPred Evaluation Script

Evaluates model1 predictions against ground truth annotations.
Calculates metrics: accuracy, precision, recall, F1-score per class, and confusion matrix.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import librosa
import json
import torch
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

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
STRIDE_DURATION = 0.5  # seconds
STRIDE_SAMPLES = int(STRIDE_DURATION * SAMPLE_RATE)  # 8000

# Class names
CLASS_NAMES = config_model1.CLASS_NAMES  # ["Normal", "Crackles", "Wheeze/Rhonchi"]

# Map original event types to model1 classes
def map_event_type_to_model1_class(event_type: str) -> int:
    """Map original event type to model1 class label."""
    if event_type == "Normal":
        return 0
    elif event_type in ["Fine Crackle", "Coarse Crackle", "Wheeze+Crackle"]:
        return 1
    elif event_type in ["Wheeze", "Rhonchi"]:
        return 2
    else:
        # Stridor, No Event -> None (skip)
        return None


def load_model(device: torch.device) -> HeARClassifier:
    """Load model1 checkpoint."""
    print(f"Loading model from {MODEL_PATH}")
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = HeARClassifier(num_classes=config_model1.NUM_CLASSES)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    return model


def find_json_path(wav_path: str, json_path_from_csv: Optional[str] = None) -> Optional[Path]:
    """Find corresponding JSON annotation file for a WAV file."""
    wav_path_obj = Path(wav_path)
    
    if json_path_from_csv and Path(json_path_from_csv).exists():
        return Path(json_path_from_csv)
    
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


def load_ground_truth_events(json_path: Path) -> List[Tuple[float, float, int]]:
    """Load ground truth events from JSON and map to model1 classes.
    
    Returns:
        List of (start_sec, end_sec, model1_class) tuples
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
            
            # Map to model1 class
            model1_class = map_event_type_to_model1_class(event_type)
            if model1_class is None:
                continue  # Skip Stridor, No Event, etc.
            
            start_sec = start_ms / 1000.0
            end_sec = end_ms / 1000.0
            
            events.append((start_sec, end_sec, model1_class))
    except Exception as e:
        print(f"Warning: Could not load JSON {json_path}: {e}")
    
    return events


def get_chunk_ground_truth(chunk_start_sec: float, chunk_end_sec: float, 
                           gt_events: List[Tuple[float, float, int]]) -> Optional[int]:
    """Determine ground truth label for a chunk based on overlapping events.
    
    Strategy:
    - If chunk center is within an event, use that event's label
    - Otherwise, if chunk overlaps with any event >50%, use that event's label
    - Otherwise, use majority label of overlapping events
    - If no overlap, label as Normal (0)
    """
    chunk_center = (chunk_start_sec + chunk_end_sec) / 2.0
    chunk_duration = chunk_end_sec - chunk_start_sec
    
    overlapping_events = []
    
    for event_start, event_end, event_class in gt_events:
        # Check overlap
        overlap_start = max(chunk_start_sec, event_start)
        overlap_end = min(chunk_end_sec, event_end)
        
        if overlap_start < overlap_end:
            overlap_duration = overlap_end - overlap_start
            overlap_ratio = overlap_duration / chunk_duration
            
            # Check if chunk center is within event
            if event_start <= chunk_center <= event_end:
                return event_class  # Direct match
            
            overlapping_events.append((overlap_ratio, event_class))
    
    if not overlapping_events:
        return 0  # Normal (no events)
    
    # Sort by overlap ratio (descending)
    overlapping_events.sort(reverse=True)
    
    # If largest overlap > 50%, use that
    if overlapping_events[0][0] > 0.5:
        return overlapping_events[0][1]
    
    # Otherwise, use majority class
    class_counts = {}
    for overlap_ratio, event_class in overlapping_events:
        class_counts[event_class] = class_counts.get(event_class, 0) + 1
    
    # Return most frequent class
    return max(class_counts.items(), key=lambda x: x[1])[0]


def predict_chunks(model: HeARClassifier, audio: np.ndarray, device: torch.device) -> List[Tuple[float, float, int]]:
    """Split audio into chunks and run inference.
    
    Returns:
        List of (start_sec, end_sec, predicted_class) tuples
    """
    predictions = []
    duration_sec = len(audio) / SAMPLE_RATE
    
    for start_sample in range(0, len(audio) - CHUNK_LENGTH_SAMPLES + 1, STRIDE_SAMPLES):
        chunk = audio[start_sample:start_sample + CHUNK_LENGTH_SAMPLES]
        
        if len(chunk) < CHUNK_LENGTH_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_LENGTH_SAMPLES - len(chunk)), mode='constant')
        
        audio_tensor = torch.from_numpy(chunk).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(audio_tensor)
            predicted_class = torch.argmax(logits, dim=1).item()
        
        start_sec = start_sample / SAMPLE_RATE
        end_sec = (start_sample + CHUNK_LENGTH_SAMPLES) / SAMPLE_RATE
        
        predictions.append((start_sec, end_sec, predicted_class))
    
    # Handle last chunk
    remaining_samples = len(audio) % STRIDE_SAMPLES
    if remaining_samples > CHUNK_LENGTH_SAMPLES * 0.5:
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


def evaluate_file(
    model: HeARClassifier,
    wav_path: str,
    json_path: Optional[Path],
    device: torch.device
) -> Tuple[List[int], List[int], Dict]:
    """Evaluate predictions for a single file.
    
    Returns:
        y_true: List of ground truth labels
        y_pred: List of predicted labels
        file_metrics: Dictionary with file-level metrics
    """
    # Load audio
    audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    
    # Load ground truth
    gt_events = []
    if json_path:
        gt_events = load_ground_truth_events(json_path)
    
    # Get predictions
    predictions = predict_chunks(model, audio, device)
    
    # Get ground truth labels for each chunk
    y_true = []
    y_pred = []
    
    for chunk_start, chunk_end, pred_class in predictions:
        # Get ground truth label for this chunk
        gt_label = get_chunk_ground_truth(chunk_start, chunk_end, gt_events)
        
        y_true.append(gt_label)
        y_pred.append(pred_class)
    
    # Calculate file-level metrics
    if len(y_true) > 0 and len(y_pred) > 0:
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2], zero_division=0
        )
        
        file_metrics = {
            'accuracy': accuracy,
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist(),
            'num_chunks': len(y_true)
        }
    else:
        file_metrics = {
            'accuracy': 0.0,
            'precision': [0.0, 0.0, 0.0],
            'recall': [0.0, 0.0, 0.0],
            'f1': [0.0, 0.0, 0.0],
            'support': [0, 0, 0],
            'num_chunks': 0
        }
    
    return y_true, y_pred, file_metrics


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], save_path: Path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main evaluation function."""
    print("="*80)
    print("BrutPred Evaluation Script")
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
    print(f"Evaluating {len(unique_wavs)} unique WAV files")
    
    # Evaluate each file
    print("\n" + "-"*80)
    print("Evaluating Files")
    print("-"*80)
    
    all_y_true = []
    all_y_pred = []
    file_results = []
    
    success_count = 0
    error_count = 0
    
    for idx, wav_path in enumerate(tqdm(unique_wavs, desc="Evaluating")):
        try:
            wav_path_obj = Path(wav_path)
            
            if not wav_path_obj.exists():
                print(f"\n[{idx+1}/{len(unique_wavs)}] Skipping (file not found): {wav_path}")
                error_count += 1
                continue
            
            filename = wav_path_obj.stem
            
            # Find JSON path
            json_path_from_csv = df[df['wav_path'] == wav_path]['file_path'].iloc[0] if len(df[df['wav_path'] == wav_path]) > 0 else None
            json_path = find_json_path(wav_path, json_path_from_csv)
            
            if not json_path:
                print(f"\n[{idx+1}/{len(unique_wavs)}] Skipping {filename} (no JSON found)")
                error_count += 1
                continue
            
            # Evaluate
            y_true, y_pred, file_metrics = evaluate_file(model, wav_path, json_path, device)
            
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            
            file_results.append({
                'filename': filename,
                'wav_path': wav_path,
                **file_metrics
            })
            
            success_count += 1
            
        except Exception as e:
            print(f"\n[{idx+1}/{len(unique_wavs)}] ✗ Error evaluating {wav_path}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
            continue
    
    # Calculate overall metrics
    print("\n" + "="*80)
    print("Overall Evaluation Results")
    print("="*80)
    
    if len(all_y_true) == 0 or len(all_y_pred) == 0:
        print("No predictions to evaluate!")
        return
    
    # Overall metrics
    accuracy = accuracy_score(all_y_true, all_y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_y_true, all_y_pred, labels=[0, 1, 2], zero_division=0
    )
    
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name:<20} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
    
    # Macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    print(f"\nMacro Averages:")
    print(f"  Precision: {macro_precision:.4f}")
    print(f"  Recall: {macro_recall:.4f}")
    print(f"  F1-Score: {macro_f1:.4f}")
    
    # Weighted averages
    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)
    
    print(f"\nWeighted Averages:")
    print(f"  Precision: {weighted_precision:.4f}")
    print(f"  Recall: {weighted_recall:.4f}")
    print(f"  F1-Score: {weighted_f1:.4f}")
    
    # Classification report
    print(f"\n{'='*80}")
    print("Detailed Classification Report")
    print("="*80)
    print(classification_report(all_y_true, all_y_pred, target_names=CLASS_NAMES, labels=[0, 1, 2]))
    
    # Confusion matrix
    cm_path = OUTPUT_DIR / "confusion_matrix.png"
    plot_confusion_matrix(all_y_true, all_y_pred, cm_path)
    print(f"\n✓ Confusion matrix saved to: {cm_path}")
    
    # Save results to CSV
    results_df = pd.DataFrame(file_results)
    results_csv_path = OUTPUT_DIR / "evaluation_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"✓ Per-file results saved to: {results_csv_path}")
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Successfully evaluated: {success_count} files")
    print(f"Errors/Skipped: {error_count} files")
    print(f"Total chunks evaluated: {len(all_y_true)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\n✓ Evaluation completed!")


if __name__ == "__main__":
    main()
