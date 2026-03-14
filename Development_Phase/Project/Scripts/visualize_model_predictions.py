#!/usr/bin/env python3
"""
Visualize predictions from 3 ensemble models on sample audio clips.

This script:
1. Selects one sample each of Crackles, Rhonchi, and Normal events
2. Loads the audio clips
3. Runs predictions through Model 1, Model 2, and Model 3
4. Creates separate plots for each model showing waveform, spectrogram, and predictions

Usage:
    python Project/Scripts/visualize_model_predictions.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import librosa
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI issues
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import blended_transform_factory
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import configs and models
from src import config
from src import config_model1
from src import config_model2
from src import config_model3
from src.models import HeARClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "Project" / "Results" / "model_visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Class names
MODEL1_CLASS_NAMES = ['Normal', 'Crackles', 'Rhonchi']
MODEL2_CLASS_NAMES = ['Normal', 'Abnormal']
MODEL3_CLASS_NAMES = ['Pneumonia', 'Bronchitis-Asthma-Bronchiolitis', 'Normal/Other']


def load_models():
    """Load all 3 ensemble models."""
    print("Loading ensemble models...")
    
    # Model 1: Event Type
    model1 = HeARClassifier(num_classes=config_model1.NUM_CLASSES).to(DEVICE)
    checkpoint1 = torch.load(config_model1.BEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
    model1.load_state_dict(checkpoint1['model_state_dict'])
    model1.eval()
    print(f"✓ Model 1 loaded: {config_model1.MODEL_NAME}")
    
    # Model 2: Binary
    model2 = HeARClassifier(num_classes=config_model2.NUM_CLASSES).to(DEVICE)
    checkpoint2 = torch.load(config_model2.BEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
    model2.load_state_dict(checkpoint2['model_state_dict'])
    model2.eval()
    print(f"✓ Model 2 loaded: {config_model2.MODEL_NAME}")
    
    # Model 3: Disease
    model3 = HeARClassifier(num_classes=config_model3.NUM_CLASSES).to(DEVICE)
    checkpoint3 = torch.load(config_model3.BEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
    model3.load_state_dict(checkpoint3['model_state_dict'])
    model3.eval()
    print(f"✓ Model 3 loaded: {config_model3.MODEL_NAME}")
    
    return model1, model2, model3


def select_samples(df):
    """Select one sample each of Crackles, Rhonchi, and Normal."""
    samples = {}
    
    # Find Crackles sample (Fine Crackle or Coarse Crackle)
    crackles_df = df[df['event_type'].isin(['Fine Crackle', 'Coarse Crackle'])]
    if len(crackles_df) > 0:
        samples['Crackles'] = crackles_df.iloc[0]
    
    # Find Rhonchi sample
    rhonchi_df = df[df['event_type'] == 'Rhonchi']
    if len(rhonchi_df) > 0:
        samples['Rhonchi'] = rhonchi_df.iloc[0]
    
    # Find Normal sample
    normal_df = df[df['event_type'] == 'Normal']
    if len(normal_df) > 0:
        samples['Normal'] = normal_df.iloc[0]
    
    return samples


def load_audio_clip(audio_path, event_start_ms, event_end_ms):
    """
    Load and preprocess audio event clip.
    
    Returns:
        audio_clip: Audio array [32000 samples]
        duration_sec: Clip duration in seconds
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    
    # Resample to 16kHz if needed
    if sr != config.SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=config.SAMPLE_RATE)
        sr = config.SAMPLE_RATE
    
    # Calculate clip range
    event_duration_ms = event_end_ms - event_start_ms
    overlap_ms = event_duration_ms * (config.OVERLAP_PERCENT / 100.0)
    clip_start_ms = max(0, event_start_ms - overlap_ms)
    clip_end_ms = event_end_ms + overlap_ms
    
    # Convert to samples
    clip_start_sample = int(clip_start_ms * sr / 1000)
    clip_end_sample = int(clip_end_ms * sr / 1000)
    clip_end_sample = min(clip_end_sample, len(audio))
    clip_start_sample = min(clip_start_sample, len(audio))
    
    # Extract clip
    event_clip = audio[clip_start_sample:clip_end_sample]
    
    # Pad or trim to exactly 2 seconds (32000 samples)
    if len(event_clip) < config.CLIP_LENGTH:
        pad_length = config.CLIP_LENGTH - len(event_clip)
        event_clip = np.pad(event_clip, (0, pad_length), mode='constant')
    elif len(event_clip) > config.CLIP_LENGTH:
        excess = len(event_clip) - config.CLIP_LENGTH
        start_trim = excess // 2
        event_clip = event_clip[start_trim:start_trim + config.CLIP_LENGTH]
    
    duration_sec = len(event_clip) / sr
    
    return event_clip, duration_sec


def predict_with_models(audio_clip, model1, model2, model3):
    """Run predictions through all 3 models."""
    # Convert to tensor
    audio_tensor = torch.from_numpy(audio_clip).float().unsqueeze(0).to(DEVICE)  # [1, 32000]
    
    predictions = {}
    
    # Model 1: Event Type
    with torch.no_grad():
        logits1 = model1(audio_tensor)
        probs1 = F.softmax(logits1, dim=1).cpu().numpy()[0]
        pred1 = np.argmax(probs1)
        predictions['model1'] = {
            'class': MODEL1_CLASS_NAMES[pred1],
            'probabilities': {name: float(prob) for name, prob in zip(MODEL1_CLASS_NAMES, probs1)},
            'predicted_class_idx': int(pred1)
        }
    
    # Model 2: Binary
    with torch.no_grad():
        logits2 = model2(audio_tensor)
        probs2 = F.softmax(logits2, dim=1).cpu().numpy()[0]
        pred2 = np.argmax(probs2)
        predictions['model2'] = {
            'class': MODEL2_CLASS_NAMES[pred2],
            'probabilities': {name: float(prob) for name, prob in zip(MODEL2_CLASS_NAMES, probs2)},
            'predicted_class_idx': int(pred2)
        }
    
    # Model 3: Disease
    with torch.no_grad():
        logits3 = model3(audio_tensor)
        probs3 = F.softmax(logits3, dim=1).cpu().numpy()[0]
        pred3 = np.argmax(probs3)
        predictions['model3'] = {
            'class': MODEL3_CLASS_NAMES[pred3],
            'probabilities': {name: float(prob) for name, prob in zip(MODEL3_CLASS_NAMES, probs3)},
            'predicted_class_idx': int(pred3)
        }
    
    return predictions


def compute_spectrogram(audio_clip, sr=16000):
    """Compute Mel-spectrogram from audio clip."""
    mel_spec = librosa.feature.melspectrogram(
        y=audio_clip,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def plot_model_predictions(audio_clip, predictions, event_type, model_name, model_num, save_path):
    """
    Create a plot showing waveform, spectrogram, and model predictions.
    
    Args:
        audio_clip: Audio array [32000 samples]
        predictions: Dictionary with model predictions
        event_type: Ground truth event type (Crackles, Rhonchi, Normal)
        model_name: Model name (Model 1, Model 2, Model 3)
        model_num: Model number (1, 2, or 3)
        save_path: Path to save the plot
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 1, height_ratios=[1, 1.5, 1], hspace=0.3, left=0.08, right=0.92)
    
    sr = config.SAMPLE_RATE
    duration_sec = len(audio_clip) / sr
    time_axis = np.linspace(0, duration_sec, len(audio_clip))
    
    # 1. Waveform
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time_axis, audio_clip, color='#2E86AB', linewidth=0.5)
    ax1.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
    ax1.set_title(f'Audio Waveform - {event_type}', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, duration_sec)  # Explicitly set xlim
    ax1.set_xticklabels([])  # Remove x-axis labels from waveform
    ax1.tick_params(axis='x', which='both', bottom=False, top=False)
    
    # 2. Spectrogram (share x-axis with waveform for perfect alignment)
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    mel_spec_db = compute_spectrogram(audio_clip, sr)
    im = ax2.imshow(
        mel_spec_db,
        aspect='auto',
        origin='lower',
        cmap='Blues',
        extent=[0, duration_sec, 0, 128]
    )
    ax2.set_ylabel('Mel Frequency Bin', fontsize=11, fontweight='bold')
    ax2.set_title('Mel-Spectrogram', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax2.set_xlim(0, duration_sec)  # Explicitly set xlim to match ax1
    # Colorbar with slightly more padding
    cbar = plt.colorbar(im, ax=ax2, label='dB', pad=0.05)
    cbar.ax.tick_params(labelsize=9)
    
    # After colorbar is added, align waveform x-axis with spectrogram
    pos2 = ax2.get_position()
    pos1 = ax1.get_position()
    # Align waveform x-axis with spectrogram (same x0 and width)
    ax1.set_position([pos2.x0, pos1.y0, pos2.width, pos1.height])
    
    # 3. Predictions
    ax3 = fig.add_subplot(gs[2])
    model_key = f'model{model_num}'
    pred_data = predictions[model_key]
    
    if model_num == 1:
        class_names = MODEL1_CLASS_NAMES
    elif model_num == 2:
        class_names = MODEL2_CLASS_NAMES
    else:
        class_names = MODEL3_CLASS_NAMES
    
    probs = [pred_data['probabilities'][name] for name in class_names]
    colors = ['#2E86AB', '#A23B72', '#F18F01'][:len(class_names)]
    
    bars = ax3.bar(class_names, probs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    pred_idx = pred_data['predicted_class_idx']
    bars[pred_idx].set_alpha(1.0)
    bars[pred_idx].set_edgecolor('red')
    bars[pred_idx].set_linewidth(3)
    
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{prob:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax3.set_ylabel('Probability', fontsize=11, fontweight='bold')
    ax3.set_title(f'{model_name} Predictions - Ground Truth: {event_type}', 
                  fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 1.35)
    ax3.grid(True, alpha=0.3, axis='y')
    
    pred_class = pred_data['class']
    pred_prob = pred_data['probabilities'][pred_class]
    max_prob = max(probs)
    trans = blended_transform_factory(ax3.transAxes, ax3.transData)
    ax3.text(0.5, max_prob + 0.20, f'Predicted: {pred_class} ({pred_prob:.3f})',
             transform=trans, ha='center', va='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=11, fontweight='bold')
    
    # Align predictions x-axis with spectrogram (same x0 and width)
    pos3 = ax3.get_position()
    ax3.set_position([pos2.x0, pos3.y0, pos2.width, pos3.height])
    
    plt.suptitle(f'{model_name} - Event Type: {event_type}', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Improved layout: Use tight_layout with padding for better alignment
    fig.tight_layout(pad=2.0, h_pad=1.0, rect=[0, 0, 1, 0.95])  # Adjust rect to make room for suptitle
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {save_path.name}")


def main():
    print("=" * 80)
    print("ENSEMBLE MODEL PREDICTION VISUALIZATION")
    print("=" * 80)
    
    # Load CSV
    csv_path = PROJECT_ROOT / "data" / "SPRSound_Event_Level_Ensemble_Dataset.csv"
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    print(f"\nLoading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total events: {len(df)}")
    
    # Select samples
    print("\nSelecting samples...")
    samples = select_samples(df)
    
    if len(samples) < 3:
        print(f"❌ Could not find all 3 event types. Found: {list(samples.keys())}")
        return
    
    print(f"✓ Selected samples:")
    for event_type, sample in samples.items():
        print(f"  - {event_type}: {sample['wav_path']} ({sample['event_start_ms']:.0f}-{sample['event_end_ms']:.0f}ms)")
    
    # Load models
    print("\n" + "-" * 80)
    model1, model2, model3 = load_models()
    
    # Process each sample
    print("\n" + "=" * 80)
    print("PROCESSING SAMPLES")
    print("=" * 80)
    
    for event_type, sample in samples.items():
        print(f"\n{'='*80}")
        print(f"Processing: {event_type}")
        print(f"{'='*80}")
        
        audio_path = PROJECT_ROOT / sample['wav_path']
        if not audio_path.exists():
            print(f"⚠ Audio file not found: {audio_path}")
            continue
        
        # Load audio clip
        print(f"Loading audio clip...")
        audio_clip, duration_sec = load_audio_clip(
            str(audio_path),
            sample['event_start_ms'],
            sample['event_end_ms']
        )
        print(f"  Clip duration: {duration_sec:.2f}s")
        
        # Run predictions
        print(f"Running predictions...")
        predictions = predict_with_models(audio_clip, model1, model2, model3)
        
        # Print predictions
        print(f"\nPredictions:")
        for model_key, pred_data in predictions.items():
            print(f"  {model_key.upper()}: {pred_data['class']} "
                  f"({pred_data['probabilities'][pred_data['class']]:.3f})")
        
        # Create plots for each model
        print(f"\nCreating plots...")
        
        # Model 1 plot
        plot_model_predictions(
            audio_clip, predictions, event_type,
            "Model 1: Event Type Classification", 1,
            OUTPUT_DIR / f"{event_type}_Model1.png"
        )
        
        # Model 2 plot
        plot_model_predictions(
            audio_clip, predictions, event_type,
            "Model 2: Binary Classification", 2,
            OUTPUT_DIR / f"{event_type}_Model2.png"
        )
        
        # Model 3 plot
        plot_model_predictions(
            audio_clip, predictions, event_type,
            "Model 3: Disease Classification", 3,
            OUTPUT_DIR / f"{event_type}_Model3.png"
        )
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE!")
    print("=" * 80)
    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print(f"\nGenerated files:")
    for event_type in samples.keys():
        for model_num in [1, 2, 3]:
            print(f"  - {event_type}_Model{model_num}.png")


if __name__ == "__main__":
    main()