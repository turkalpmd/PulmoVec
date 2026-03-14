#!/usr/bin/env python3
"""
Visualize predictions from 3 ensemble models on full audio files with multiple events.

This script:
1. Selects one sample each of Severe Pneumonia, Asthma, Bronchiolitis, and Normal patients
2. Loads the full audio files
3. Shows all labeled events in each file
4. Runs predictions through Model 1, Model 2, and Model 3 for each event
5. Creates timeline visualizations showing ground truth vs predictions

Usage:
    python Project/Scripts/visualize_full_audio_predictions.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
from pydub import AudioSegment
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
OUTPUT_DIR = PROJECT_ROOT / "Project" / "Results" / "full_audio_visualizations"
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


def evaluate_patient_accuracy(patient_events, model1, model2, model3):
    """
    Evaluate prediction accuracy for a patient's events.
    
    Returns:
        accuracies: Dict with accuracies for each model
        total_events: Number of events evaluated
    """
    accuracies = {'model1': 0, 'model2': 0, 'model3': 0}
    correct_counts = {'model1': 0, 'model2': 0, 'model3': 0}
    total_events = 0
    
    # Get audio path from first event
    first_event = patient_events.iloc[0]
    audio_path = PROJECT_ROOT / first_event['wav_path']
    
    if not audio_path.exists():
        return accuracies, 0
    
    for _, event in patient_events.iterrows():
        # Load audio clip
        try:
            audio_clip, _ = load_audio_clip(
                str(audio_path),
                event['event_start_ms'],
                event['event_end_ms']
            )
            
            # Run predictions
            predictions = predict_with_models(audio_clip, model1, model2, model3)
            
            # Check Model 1 accuracy
            gt_type = event['event_type']
            pred1_class = predictions['model1']['class']
            if gt_type == 'Normal' and pred1_class == 'Normal':
                correct_counts['model1'] += 1
            elif gt_type in ['Fine Crackle', 'Coarse Crackle', 'Wheeze+Crackle'] and pred1_class == 'Crackles':
                correct_counts['model1'] += 1
            elif gt_type in ['Wheeze', 'Rhonchi'] and pred1_class == 'Rhonchi':
                correct_counts['model1'] += 1
            
            # Check Model 2 accuracy
            pred2_class = predictions['model2']['class']
            if gt_type == 'Normal' and pred2_class == 'Normal':
                correct_counts['model2'] += 1
            elif gt_type != 'Normal' and pred2_class == 'Abnormal':
                correct_counts['model2'] += 1
            
            # Check Model 3 accuracy
            gt_disease = event['disease']
            pred3_class = predictions['model3']['class']
            if gt_disease in ['Pneumonia (severe)', 'Pneumonia (non-severe)'] and pred3_class == 'Pneumonia':
                correct_counts['model3'] += 1
            elif gt_disease in ['Bronchitis', 'Asthma', 'Bronchiolitis'] and pred3_class == 'Bronchitis-Asthma-Bronchiolitis':
                correct_counts['model3'] += 1
            elif gt_disease == 'Control Group' and pred3_class == 'Normal/Other':
                correct_counts['model3'] += 1
            
            total_events += 1
        except Exception as e:
            continue
    
    if total_events > 0:
        accuracies['model1'] = correct_counts['model1'] / total_events
        accuracies['model2'] = correct_counts['model2'] / total_events
        accuracies['model3'] = correct_counts['model3'] / total_events
    
    return accuracies, total_events


def select_diverse_patients(df, model1, model2, model3, min_events=3, max_events=20, num_patients_per_type=5):
    """
    Select diverse patients for each disease type (high, medium, low accuracy).
    Includes both correct and incorrect predictions.
    
    Args:
        df: Full dataset
        model1, model2, model3: Loaded models
        min_events: Minimum number of events per patient
        max_events: Maximum number of events per patient (to avoid too complex cases)
        num_patients_per_type: Number of patients to select per disease type
    
    Returns:
        samples: List of patient dicts with diverse accuracy levels
    """
    samples = []
    
    # Disease types to find
    disease_types = {
        'Severe Pneumonia': 'Pneumonia (severe)',
        'Asthma': 'Asthma',
        'Bronchiolitis': 'Bronchiolitis',
        'Normal': 'Control Group'
    }
    
    print("\nEvaluating patients to find diverse examples (correct and incorrect)...")
    
    for patient_type, disease_name in disease_types.items():
        print(f"\n  Evaluating {patient_type} patients...")
        disease_df = df[df['disease'] == disease_name]
        
        if len(disease_df) == 0:
            continue
        
        # Get unique patients
        patient_nums = disease_df['patient_number'].unique()
        print(f"    Found {len(patient_nums)} patients")
        
        patient_scores = []
        
        for patient_num in patient_nums[:100]:  # Check more patients
            patient_events = disease_df[disease_df['patient_number'] == patient_num]
            
            # Filter by event count
            if len(patient_events) < min_events or len(patient_events) > max_events:
                continue
            
            # Evaluate accuracy
            accuracies, total_events = evaluate_patient_accuracy(patient_events, model1, model2, model3)
            
            if total_events == 0:
                continue
            
            # Calculate average accuracy across all 3 models
            avg_accuracy = (accuracies['model1'] + accuracies['model2'] + accuracies['model3']) / 3.0
            
            patient_scores.append({
                'patient_number': patient_num,
                'events': patient_events,
                'accuracy': avg_accuracy,
                'model1_acc': accuracies['model1'],
                'model2_acc': accuracies['model2'],
                'model3_acc': accuracies['model3'],
                'num_events': total_events,
                'patient_type': patient_type,
                'disease': disease_name
            })
        
        # Sort by accuracy
        patient_scores.sort(key=lambda x: x['accuracy'], reverse=True)
        
        # Select diverse patients: best, good, medium, poor, worst
        if len(patient_scores) > 0:
            num_to_select = min(num_patients_per_type, len(patient_scores))
            if num_to_select == 1:
                selected_indices = [0]
            elif num_to_select == 2:
                selected_indices = [0, len(patient_scores) - 1]  # Best and worst
            elif num_to_select == 3:
                selected_indices = [0, len(patient_scores) // 2, len(patient_scores) - 1]  # Best, middle, worst
            elif num_to_select == 4:
                selected_indices = [0, len(patient_scores) // 3, 2 * len(patient_scores) // 3, len(patient_scores) - 1]
            else:  # 5 or more
                selected_indices = [
                    0,  # Best
                    len(patient_scores) // 4,  # Top 25%
                    len(patient_scores) // 2,  # Middle
                    3 * len(patient_scores) // 4,  # Bottom 25%
                    len(patient_scores) - 1  # Worst
                ][:num_to_select]
            
            for idx in selected_indices:
                patient = patient_scores[idx]
                samples.append({
                    'patient_number': patient['patient_number'],
                    'disease': patient['disease'],
                    'events': patient['events'],
                    'accuracy': patient['accuracy'],
                    'model1_acc': patient['model1_acc'],
                    'model2_acc': patient['model2_acc'],
                    'model3_acc': patient['model3_acc'],
                    'num_events': patient['num_events'],
                    'patient_type': patient['patient_type']
                })
                print(f"    ✓ Selected Patient {patient['patient_number']}: "
                      f"Avg Acc={patient['accuracy']:.3f} "
                      f"(M1={patient['model1_acc']:.3f}, "
                      f"M2={patient['model2_acc']:.3f}, "
                      f"M3={patient['model3_acc']:.3f}), "
                      f"{patient['num_events']} events")
        else:
            print(f"    ⚠ No suitable patients found for {patient_type}")
    
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


def load_full_audio(audio_path):
    """Load full audio file."""
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    
    # Resample to 16kHz if needed
    if sr != config.SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=config.SAMPLE_RATE)
        sr = config.SAMPLE_RATE
    
    duration_sec = len(audio) / sr
    return audio, duration_sec, sr


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
            'predicted_class_idx': int(pred1),
            'max_prob': float(probs1[pred1])
        }
    
    # Model 2: Binary
    with torch.no_grad():
        logits2 = model2(audio_tensor)
        probs2 = F.softmax(logits2, dim=1).cpu().numpy()[0]
        pred2 = np.argmax(probs2)
        predictions['model2'] = {
            'class': MODEL2_CLASS_NAMES[pred2],
            'probabilities': {name: float(prob) for name, prob in zip(MODEL2_CLASS_NAMES, probs2)},
            'predicted_class_idx': int(pred2),
            'max_prob': float(probs2[pred2])
        }
    
    # Model 3: Disease
    with torch.no_grad():
        logits3 = model3(audio_tensor)
        probs3 = F.softmax(logits3, dim=1).cpu().numpy()[0]
        pred3 = np.argmax(probs3)
        predictions['model3'] = {
            'class': MODEL3_CLASS_NAMES[pred3],
            'probabilities': {name: float(prob) for name, prob in zip(MODEL3_CLASS_NAMES, probs3)},
            'predicted_class_idx': int(pred3),
            'max_prob': float(probs3[pred3])
        }
    
    return predictions


def plot_full_audio_predictions(audio, sr, events_df, all_predictions, patient_info, model_num, save_path_base, audio_path, max_events_per_plot=3):
    """
    Create plots showing full audio waveform with GT and prediction timelines.
    Each plot shows maximum 3 events.
    
    Args:
        audio: Full audio array
        sr: Sample rate
        events_df: DataFrame with all events for this patient
        all_predictions: List of prediction dicts for each event
        patient_info: Dictionary with patient information
        model_num: Model number (1, 2, or 3)
        save_path_base: Base path to save plots (will add _partN if multiple plots)
        max_events_per_plot: Maximum number of events per plot (default: 3)
    """
    duration_sec = len(audio) / sr
    time_axis = np.linspace(0, duration_sec, len(audio))
    
    num_events = len(events_df)
    num_plots = (num_events + max_events_per_plot - 1) // max_events_per_plot  # Ceiling division
    
    # Get model-specific class names and colors
    if model_num == 1:
        class_names = MODEL1_CLASS_NAMES
        pred_colors = {'Normal': '#4CAF50', 'Crackles': '#FF9800', 'Rhonchi': '#E91E63'}
    elif model_num == 2:
        class_names = MODEL2_CLASS_NAMES
        pred_colors = {'Normal': '#4CAF50', 'Abnormal': '#F44336'}
    else:
        class_names = MODEL3_CLASS_NAMES
        pred_colors = {'Pneumonia': '#FF5722', 'Bronchitis-Asthma-Bronchiolitis': '#9C27B0', 'Normal/Other': '#4CAF50'}
    
    # Color map for event types (GT)
    event_colors = {
        'Normal': '#4CAF50',
        'Fine Crackle': '#FF9800',
        'Coarse Crackle': '#FF5722',
        'Wheeze': '#9C27B0',
        'Rhonchi': '#E91E63',
        'Wheeze+Crackle': '#F44336',
        'Stridor': '#795548'
    }
    
    # Create plots in batches of max_events_per_plot
    for plot_idx in range(num_plots):
        start_idx = plot_idx * max_events_per_plot
        end_idx = min(start_idx + max_events_per_plot, num_events)
        events_subset = events_df.iloc[start_idx:end_idx]
        predictions_subset = all_predictions[start_idx:end_idx]
        num_subset_events = len(events_subset)
        
        # Create figure: 3 rows (GT Timeline, Waveform, Prediction Timeline)
        # All events shown on the same waveform
        fig = plt.figure(figsize=(20, 8))
        gs = fig.add_gridspec(3, 1, height_ratios=[0.8, 2, 0.8], hspace=0.2)
        
        # Calculate x-axis range dynamically
        # Find first event start time
        first_event_start = events_subset.iloc[0]['event_start_ms'] / 1000.0
        last_event_end = events_subset.iloc[-1]['event_end_ms'] / 1000.0
        
        # If first event starts after 8 seconds, start x-axis 1 second before first event
        # Always show exactly 8 seconds
        if first_event_start > 8.0:
            x_start = max(0, first_event_start - 1.0)
            x_end = x_start + 8.0
            # Don't exceed audio duration
            if x_end > duration_sec:
                x_end = duration_sec
                x_start = max(0, x_end - 8.0)
        else:
            # Normal case: start from 0, exactly 8 seconds (or less if audio is shorter)
            x_start = 0
            x_end = min(8.0, duration_sec)
        
        x_limit = x_end
        
        # 1. Ground Truth Timeline (Top) - Show GT labels matching what the model predicts
        ax_gt = fig.add_subplot(gs[0])
        ax_gt.set_xlim(x_start, x_limit)
        ax_gt.set_ylim(-0.5, num_subset_events - 0.5)
        ax_gt.set_yticks(range(num_subset_events))
        ax_gt.set_yticklabels([f"Event {start_idx + i + 1}" for i in range(num_subset_events)])
        ax_gt.set_ylabel('Ground Truth', fontsize=11, fontweight='bold')
        ax_gt.set_title('Ground Truth Events', fontsize=12, fontweight='bold')
        ax_gt.grid(True, alpha=0.3, axis='x')
        ax_gt.set_xticklabels([])
        
        for local_idx, (_, event) in enumerate(events_subset.iterrows()):
            start_sec = event['event_start_ms'] / 1000.0
            end_sec = event['event_end_ms'] / 1000.0
            event_type = event['event_type']
            disease = event.get('disease', 'Unknown')
            
            # Map GT to model's prediction space
            if model_num == 1:
                # Model 1: Event Type (Normal, Crackles, Rhonchi)
                if event_type == 'Normal':
                    gt_label = 'Normal'
                    color = pred_colors.get('Normal', '#757575')
                elif event_type in ['Fine Crackle', 'Coarse Crackle', 'Wheeze+Crackle']:
                    gt_label = 'Crackles'
                    color = pred_colors.get('Crackles', '#757575')
                elif event_type in ['Wheeze', 'Rhonchi']:
                    gt_label = 'Rhonchi'
                    color = pred_colors.get('Rhonchi', '#757575')
                else:
                    gt_label = event_type
                    color = '#757575'
            elif model_num == 2:
                # Model 2: Binary (Normal, Abnormal)
                if event_type == 'Normal':
                    gt_label = 'Normal'
                    color = pred_colors.get('Normal', '#757575')
                else:
                    gt_label = 'Abnormal'
                    color = pred_colors.get('Abnormal', '#757575')
            else:  # model_num == 3
                # Model 3: Disease (Pneumonia, Bronchitis-Asthma-Bronchiolitis, Normal/Other)
                if disease in ['Pneumonia (severe)', 'Pneumonia (non-severe)']:
                    gt_label = 'Pneumonia'
                    color = pred_colors.get('Pneumonia', '#757575')
                elif disease in ['Bronchitis', 'Asthma', 'Bronchiolitis']:
                    gt_label = 'Bronchitis-Asthma-Bronchiolitis'
                    color = pred_colors.get('Bronchitis-Asthma-Bronchiolitis', '#757575')
                else:
                    gt_label = 'Normal/Other'
                    color = pred_colors.get('Normal/Other', '#757575')
            
            # Shorten label if too long
            if len(gt_label) > 25:
                gt_label_display = gt_label[:22] + "..."
            else:
                gt_label_display = gt_label
            
            # Vertical rectangle with light color
            rect_gt = Rectangle((start_sec, local_idx - 0.4), end_sec - start_sec, 0.8,
                               facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
            ax_gt.add_patch(rect_gt)
            ax_gt.text((start_sec + end_sec) / 2, local_idx, gt_label_display, 
                      ha='center', va='center', fontsize=9, fontweight='bold')
        
        # 2. Full Audio Waveform (Middle)
        ax_wave = fig.add_subplot(gs[1], sharex=ax_gt)
        ax_wave.plot(time_axis, audio, color='#2E86AB', linewidth=0.5, alpha=0.8)
        ax_wave.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
        ax_wave.set_title('Audio Waveform', fontsize=12, fontweight='bold')
        ax_wave.grid(True, alpha=0.3)
        ax_wave.set_xlim(x_start, x_limit)
        ax_wave.set_xticklabels([])
        
        # Highlight event regions with vertical lines (light color) - use model's class colors
        for local_idx, (_, event) in enumerate(events_subset.iterrows()):
            start_sec = event['event_start_ms'] / 1000.0
            end_sec = event['event_end_ms'] / 1000.0
            event_type = event['event_type']
            disease = event.get('disease', 'Unknown')
            
            # Map to model's prediction space for consistent coloring
            if model_num == 1:
                # Model 1: Event Type
                if event_type == 'Normal':
                    color = pred_colors.get('Normal', '#757575')
                elif event_type in ['Fine Crackle', 'Coarse Crackle', 'Wheeze+Crackle']:
                    color = pred_colors.get('Crackles', '#757575')
                elif event_type in ['Wheeze', 'Rhonchi']:
                    color = pred_colors.get('Rhonchi', '#757575')
                else:
                    color = '#757575'
            elif model_num == 2:
                # Model 2: Binary
                if event_type == 'Normal':
                    color = pred_colors.get('Normal', '#757575')
                else:
                    color = pred_colors.get('Abnormal', '#757575')
            else:  # model_num == 3
                # Model 3: Disease
                if disease in ['Pneumonia (severe)', 'Pneumonia (non-severe)']:
                    color = pred_colors.get('Pneumonia', '#757575')
                elif disease in ['Bronchitis', 'Asthma', 'Bronchiolitis']:
                    color = pred_colors.get('Bronchitis-Asthma-Bronchiolitis', '#757575')
                else:
                    color = pred_colors.get('Normal/Other', '#757575')
            
            # Vertical span with light color
            ax_wave.axvspan(start_sec, end_sec, alpha=0.15, color=color)
            # Vertical lines at boundaries
            ax_wave.axvline(start_sec, color=color, linestyle='--', linewidth=1.5, alpha=0.6)
            ax_wave.axvline(end_sec, color=color, linestyle='--', linewidth=1.5, alpha=0.6)
        
        # 3. Model Predictions Timeline (Bottom)
        ax_pred = fig.add_subplot(gs[2], sharex=ax_gt)
        ax_pred.set_xlim(x_start, x_limit)
        ax_pred.set_ylim(-0.5, num_subset_events - 0.5)
        ax_pred.set_yticks(range(num_subset_events))
        ax_pred.set_yticklabels([f"Event {start_idx + i + 1}" for i in range(num_subset_events)])
        ax_pred.set_ylabel('Prediction', fontsize=11, fontweight='bold')
        ax_pred.set_title(f'Model {model_num} Predictions', fontsize=12, fontweight='bold')
        ax_pred.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
        ax_pred.grid(True, alpha=0.3, axis='x')
        
        for local_idx, (_, event) in enumerate(events_subset.iterrows()):
            start_sec = event['event_start_ms'] / 1000.0
            end_sec = event['event_end_ms'] / 1000.0
            pred = predictions_subset[local_idx]
            model_key = f'model{model_num}'
            pred_class = pred[model_key]['class']
            
            # Show only what the model predicts (no disease prediction for Model 1-2)
            # Shorten label if too long
            if len(pred_class) > 25:
                pred_label_display = pred_class[:22] + "..."
            else:
                pred_label_display = pred_class
            
            color = pred_colors.get(pred_class, '#757575')
            # Vertical rectangle with light color
            rect_pred = Rectangle((start_sec, local_idx - 0.4), end_sec - start_sec, 0.8,
                                 facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
            ax_pred.add_patch(rect_pred)
            ax_pred.text((start_sec + end_sec) / 2, local_idx, pred_label_display, 
                       ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Overall title
        model_names = {
            1: "Model 1: Event Type Classification",
            2: "Model 2: Binary Classification",
            3: "Model 3: Disease Classification"
        }
        
        title_suffix = f" - Part {plot_idx+1}/{num_plots}" if num_plots > 1 else ""
        plt.suptitle(f'{model_names[model_num]} - Patient: {patient_info["patient_number"]} ({patient_info["disease"]}){title_suffix}',
                    fontsize=14, fontweight='bold', y=0.998)
        
        # Save path
        if num_plots > 1:
            save_path = save_path_base.parent / f"{save_path_base.stem}_part{plot_idx+1}{save_path_base.suffix}"
        else:
            save_path = save_path_base
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {save_path.name}")
        
        # Save WAV clip for this entire part (all events concatenated)
        patient_name = patient_info['disease'].replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        patient_num = patient_info['patient_number']
        
        try:
            # Load full audio
            full_audio, _, sr = load_full_audio(str(audio_path))
            
            # Find the time range for this part (from first event start to last event end)
            first_event = events_subset.iloc[0]
            last_event = events_subset.iloc[-1]
            
            start_sec = first_event['event_start_ms'] / 1000.0
            end_sec = last_event['event_end_ms'] / 1000.0
            
            # Extract the part audio segment
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            end_sample = min(end_sample, len(full_audio))
            start_sample = min(start_sample, len(full_audio))
            
            part_audio = full_audio[start_sample:end_sample]
            
            # Save WAV and MP3 files for this part
            if num_plots > 1:
                base_filename = f"{patient_name}_Patient{patient_num}_part{plot_idx+1}"
            else:
                base_filename = f"{patient_name}_Patient{patient_num}"
            
            wav_filename = f"{base_filename}.wav"
            mp3_filename = f"{base_filename}.mp3"
            
            wav_path = save_path.parent / wav_filename
            mp3_path = save_path.parent / mp3_filename
            
            # Save as WAV
            sf.write(str(wav_path), part_audio, sr)
            print(f"  ✓ Saved WAV: {wav_filename}")
            
            # Save as MP3 using pydub
            try:
                # Convert numpy array to AudioSegment
                # Normalize to int16 range
                audio_int16 = (part_audio * 32767).astype(np.int16)
                audio_segment = AudioSegment(
                    audio_int16.tobytes(),
                    frame_rate=sr,
                    channels=1,
                    sample_width=2  # 16-bit = 2 bytes
                )
                audio_segment.export(str(mp3_path), format="mp3", bitrate="192k")
                print(f"  ✓ Saved MP3: {mp3_filename}")
            except Exception as mp3_error:
                print(f"    ⚠ Could not save MP3: {mp3_error}")
            
        except Exception as e:
            print(f"    ⚠ Could not save audio for part {plot_idx+1}: {e}")


def main():
    print("=" * 80)
    print("FULL AUDIO PREDICTION VISUALIZATION")
    print("=" * 80)
    
    # Load CSV
    csv_path = PROJECT_ROOT / "data" / "SPRSound_Event_Level_Ensemble_Dataset.csv"
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    print(f"\nLoading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total events: {len(df)}")
    
    # Load models first (needed for patient evaluation)
    print("\n" + "-" * 80)
    model1, model2, model3 = load_models()
    
    # Select diverse patient samples (high, medium, low accuracy)
    print("\n" + "=" * 80)
    print("SELECTING DIVERSE PATIENTS (CORRECT AND INCORRECT PREDICTIONS)")
    print("=" * 80)
    samples = select_diverse_patients(df, model1, model2, model3, min_events=3, max_events=20, num_patients_per_type=5)
    
    if len(samples) == 0:
        print("⚠ No suitable patients found!")
        return
    
    print(f"\n✓ Selected {len(samples)} diverse patients (including correct and incorrect predictions):")
    for patient_info in samples:
        num_events = patient_info.get('num_events', len(patient_info['events']))
        avg_acc = patient_info.get('accuracy', 0)
        patient_type = patient_info.get('patient_type', 'Unknown')
        print(f"  - {patient_type}: Patient {patient_info['patient_number']} "
              f"({num_events} events, Avg Accuracy: {avg_acc:.3f})")
    
    # Process each patient
    print("\n" + "=" * 80)
    print("PROCESSING PATIENTS")
    print("=" * 80)
    
    for patient_info in samples:
        patient_type = patient_info.get('patient_type', 'Unknown')
        patient_num = patient_info['patient_number']
        avg_acc = patient_info.get('accuracy', 0)
        
        print(f"\n{'='*80}")
        print(f"Processing: {patient_type} (Patient {patient_num}, Accuracy: {avg_acc:.3f})")
        print(f"{'='*80}")
        
        events_df = patient_info['events'].copy()
        
        # Get audio path from first event (all events from same patient should have same audio)
        first_event = events_df.iloc[0]
        audio_path = PROJECT_ROOT / first_event['wav_path']
        
        if not audio_path.exists():
            print(f"⚠ Audio file not found: {audio_path}")
            continue
        
        print(f"Loading full audio: {audio_path.name}")
        audio, duration_sec, sr = load_full_audio(str(audio_path))
        print(f"  Audio duration: {duration_sec:.2f}s")
        print(f"  Number of events: {len(events_df)}")
        
        # Process each event
        print(f"\nProcessing {len(events_df)} events...")
        all_predictions = []
        
        for idx, (_, event) in enumerate(events_df.iterrows()):
            print(f"  Event {idx+1}/{len(events_df)}: {event['event_type']} "
                  f"({event['event_start_ms']:.0f}-{event['event_end_ms']:.0f}ms)")
            
            # Load audio clip for this event
            audio_clip, clip_duration = load_audio_clip(
                str(audio_path),
                event['event_start_ms'],
                event['event_end_ms']
            )
            
            # Run predictions
            predictions = predict_with_models(audio_clip, model1, model2, model3)
            all_predictions.append(predictions)
            
            # Print predictions
            print(f"    Model 1: {predictions['model1']['class']} ({predictions['model1']['max_prob']:.3f})")
            print(f"    Model 2: {predictions['model2']['class']} ({predictions['model2']['max_prob']:.3f})")
            print(f"    Model 3: {predictions['model3']['class']} ({predictions['model3']['max_prob']:.3f})")
        
        # Create plots for each model
        print(f"\nCreating plots...")
        
        # Add accuracy to filename to distinguish correct/incorrect predictions
        acc_str = f"Acc{avg_acc:.2f}".replace('.', '_')
        patient_type_clean = patient_type.replace(' ', '_')
        
        # Model 1 plot
        plot_full_audio_predictions(
            audio, sr, events_df, all_predictions, patient_info, 1,
            OUTPUT_DIR / f"{patient_type_clean}_Patient{patient_num}_{acc_str}_Model1.png",
            audio_path
        )
        
        # Model 2 plot
        plot_full_audio_predictions(
            audio, sr, events_df, all_predictions, patient_info, 2,
            OUTPUT_DIR / f"{patient_type_clean}_Patient{patient_num}_{acc_str}_Model2.png",
            audio_path
        )
        
        # Model 3 plot
        plot_full_audio_predictions(
            audio, sr, events_df, all_predictions, patient_info, 3,
            OUTPUT_DIR / f"{patient_type_clean}_Patient{patient_num}_{acc_str}_Model3.png",
            audio_path
        )
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE!")
    print("=" * 80)
    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print(f"\nGenerated {len(samples) * 3} plot files:")
    print(f"  - {len(samples)} patients × 3 models = {len(samples) * 3} plots")
    print(f"  - Includes both correct (high accuracy) and incorrect (low accuracy) predictions")


if __name__ == "__main__":
    main()
