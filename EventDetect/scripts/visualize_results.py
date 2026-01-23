"""Visualization functions for event detection results."""

import numpy as np
import matplotlib.pyplot as plt
import librosa
from pathlib import Path
from typing import List, Tuple, Dict
import seaborn as sns

sns.set_style("whitegrid")


def plot_timeline(audio_path: str, 
                 ground_truth: List[Tuple[float, float]],
                 predictions: List[Tuple[float, float]],
                 gt_event_types: List[str] = None,
                 pred_event_types: List[str] = None,
                 save_path: str = None,
                 title: str = "Event Detection Timeline"):
    """
    Plot timeline showing ground truth vs predicted segments overlaid on audio waveform.
    
    Args:
        audio_path: Path to WAV file
        ground_truth: List of (start_sec, end_sec) tuples
        predictions: List of (start_sec, end_sec) tuples
        gt_event_types: List of event type strings for ground truth (optional)
        pred_event_types: List of event type strings for predictions (optional)
        save_path: Path to save plot
        title: Plot title
    """
    # Load audio for waveform
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(audio) / sr
    time_axis = np.linspace(0, duration, len(audio))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Plot waveform
    ax1.plot(time_axis, audio, color='gray', alpha=0.6, linewidth=0.5)
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.set_title('Audio Waveform', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot segments
    y_gt = 0.8
    y_pred = 0.2
    
    # Color mapping for event types
    type_colors = {
        'Normal': 'green',
        'Crackles': 'orange',
        'Rhonchi': 'purple',
        'Wheeze/Rhonchi': 'purple'
    }
    
    # Ground truth segments
    for i, (start, end) in enumerate(ground_truth):
        if gt_event_types and i < len(gt_event_types):
            event_type = gt_event_types[i]
            color = type_colors.get(event_type, 'green')
        else:
            color = 'green'
        ax2.barh(y_gt, end - start, left=start, height=0.3, 
                color=color, alpha=0.7, edgecolor='black', linewidth=1.5,
                label='GT' if i == 0 else '')
        # Add event type label
        if gt_event_types and i < len(gt_event_types):
            ax2.text(start + (end - start) / 2, y_gt, gt_event_types[i], 
                   ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Predicted segments with event types
    for i, (start, end) in enumerate(predictions):
        if pred_event_types and i < len(pred_event_types):
            event_type = pred_event_types[i]
            color = type_colors.get(event_type, 'red')
        else:
            color = 'red'
        ax2.barh(y_pred, end - start, left=start, height=0.3,
                color=color, alpha=0.7, edgecolor='black', linewidth=1.5,
                label='Predicted' if i == 0 else '')
        # Add event type label
        if pred_event_types and i < len(pred_event_types):
            ax2.text(start + (end - start) / 2, y_pred, pred_event_types[i], 
                   ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Segments', fontsize=12)
    ax2.set_title(title, fontsize=14, fontweight='bold')
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_yticks([y_gt, y_pred])
    ax2.set_yticklabels(['Ground Truth', 'Predicted'])
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved timeline plot: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_detection_scores(detection_scores: np.ndarray,
                         clip_start_times: List[float],
                         label_list: List[str],
                         threshold: float = 0.5,
                         save_path: str = None,
                         title: str = "HeAR Detection Scores"):
    """
    Plot detection scores over time as heatmap.
    
    Args:
        detection_scores: Array of shape [num_clips, num_labels]
        clip_start_times: List of start times for each clip
        label_list: List of label names
        threshold: Detection threshold
        save_path: Path to save plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Transpose for heatmap: time on x-axis, labels on y-axis
    scores_t = detection_scores.T  # [num_labels, num_clips]
    
    im = ax.imshow(scores_t, aspect='auto', cmap='YlOrRd', 
                   interpolation='nearest', vmin=0, vmax=1)
    
    # Set ticks
    num_clips = len(clip_start_times)
    if num_clips > 20:
        # Show fewer ticks for readability
        tick_indices = np.linspace(0, num_clips - 1, 10, dtype=int)
    else:
        tick_indices = np.arange(num_clips)
    
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([f'{clip_start_times[i]:.1f}' for i in tick_indices], rotation=45)
    ax.set_yticks(np.arange(len(label_list)))
    ax.set_yticklabels(label_list)
    
    # Add threshold line
    ax.axhline(y=-0.5, color='black', linewidth=2, linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Event Labels', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Detection Score', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved detection scores plot: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_metrics_summary(metrics: Dict,
                        save_path: str = None,
                        title: str = "Evaluation Metrics Summary"):
    """
    Plot bar charts for key metrics.
    
    Args:
        metrics: Dictionary from evaluate_segmentation()
        save_path: Path to save plot
        title: Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. IoU metrics
    ax = axes[0, 0]
    iou_data = {
        'Mean IoU': metrics.get('mean_iou', 0),
        'Median IoU': metrics.get('median_iou', 0)
    }
    bars = ax.bar(iou_data.keys(), iou_data.values(), color=['#3498db', '#2ecc71'])
    ax.set_ylabel('IoU', fontsize=11)
    ax.set_title('IoU Metrics', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 2. Precision/Recall/F1 at IoU=0.5
    ax = axes[0, 1]
    if 'precision_recall_f1' in metrics and '0.5' in metrics['precision_recall_f1']:
        prf1 = metrics['precision_recall_f1']['0.5']
        prf1_data = {
            'Precision': prf1.get('precision', 0),
            'Recall': prf1.get('recall', 0),
            'F1': prf1.get('f1', 0)
        }
        bars = ax.bar(prf1_data.keys(), prf1_data.values(), color=['#e74c3c', '#f39c12', '#9b59b6'])
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title('Precision/Recall/F1 @ IoU=0.5', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 3. Boundary errors
    ax = axes[1, 0]
    error_data = {
        'Onset Error': metrics.get('median_onset_error', 0),
        'Offset Error': metrics.get('median_offset_error', 0),
        'Duration Error': metrics.get('median_duration_error', 0)
    }
    bars = ax.bar(error_data.keys(), error_data.values(), color=['#34495e', '#7f8c8d', '#95a5a6'])
    ax.set_ylabel('Error (seconds)', fontsize=11)
    ax.set_title('Median Boundary Errors', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}s', ha='center', va='bottom', fontsize=10)
    
    # 4. Counts
    ax = axes[1, 1]
    count_data = {
        'Ground Truth': metrics.get('num_ground_truth', 0),
        'Predictions': metrics.get('num_predictions', 0),
        'Matched': metrics.get('num_matched', 0)
    }
    bars = ax.bar(count_data.keys(), count_data.values(), color=['#16a085', '#27ae60', '#2ecc71'])
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Segment Counts', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved metrics summary: {save_path}")
    else:
        plt.show()
    
    plt.close()
