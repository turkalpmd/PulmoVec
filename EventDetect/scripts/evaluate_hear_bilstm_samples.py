"""
Evaluate HeAR + BiLSTM temporal segmentation on sample audio files.
Compares predictions with ground truth from JSON annotations.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import librosa
from typing import Dict, List, Tuple
import argparse

# Add EventDetect to path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import config_hear_bilstm as config
from models.hear_encoder import HeAREncoder, EmbeddingCache
from models.bilstm_event_detector import BiLSTMEventDetector
from scripts.inference_hear_bilstm import load_model, predict_events
from scripts.load_ground_truth import load_ground_truth
from utils.metrics_seq import compute_segment_f1


def load_sample_info() -> Dict:
    """Load sample information from selected_samples.json."""
    samples_json = config.EVENT_DETECT_DIR / "samples" / "selected_samples.json"
    
    if not samples_json.exists():
        print(f"Warning: Sample info not found at {samples_json}")
        print("Using default samples if available")
        return {}
    
    with open(samples_json, 'r') as f:
        samples = json.load(f)
    
    return samples


def plot_timeline(
    audio_path: str,
    gt_segments: List[Tuple[float, float]],
    pred_segments: List[Tuple[float, float]],
    window_probs: np.ndarray,
    output_path: Path,
    event_group: str = "unknown"
):
    """
    Create timeline visualization with waveform, GT segments, and predictions.
    
    Args:
        audio_path: Path to audio file
        gt_segments: List of (start, end) ground truth segments
        pred_segments: List of (start, end) predicted segments
        window_probs: (T_windows,) window probabilities
        output_path: Path to save plot
        event_group: Event group name for title
    """
    # Load audio for waveform
    audio, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE, mono=True)
    duration_sec = len(audio) / sr
    time_axis = np.linspace(0, duration_sec, len(audio))
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f'Event Detection Timeline: {event_group}', fontsize=16, fontweight='bold')
    
    # Plot 1: Waveform
    ax1 = axes[0]
    ax1.plot(time_axis, audio, color='gray', alpha=0.7, linewidth=0.5)
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.set_title('Audio Waveform', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, duration_sec)
    
    # Plot 2: Ground Truth and Predictions
    ax2 = axes[1]
    
    # Plot GT segments
    for i, (start, end) in enumerate(gt_segments):
        label = 'Ground Truth' if i == 0 else ''
        ax2.axvspan(start, end, alpha=0.5, color='green', label=label)
    
    # Plot predicted segments
    for i, (start, end) in enumerate(pred_segments):
        label = 'Predicted' if i == 0 else ''
        ax2.axvspan(start, end, alpha=0.5, color='red', label=label)
    
    ax2.set_ylabel('Segments', fontsize=12)
    ax2.set_title('Event Segments', fontsize=14)
    ax2.set_xlim(0, duration_sec)
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Window probabilities
    ax3 = axes[2]
    window_times = np.arange(len(window_probs)) * config.HOP_SEC + config.WINDOW_SEC / 2
    ax3.plot(window_times, window_probs, color='blue', linewidth=2, label='Event Probability')
    ax3.axhline(y=config.EVAL_THRESHOLD, color='red', linestyle='--', linewidth=1, label=f'Threshold ({config.EVAL_THRESHOLD})')
    ax3.fill_between(window_times, 0, window_probs, alpha=0.3, color='blue')
    ax3.set_xlabel('Time (seconds)', fontsize=12)
    ax3.set_ylabel('Probability', fontsize=12)
    ax3.set_title('Window-level Event Probabilities', fontsize=14)
    ax3.set_xlim(0, duration_sec)
    ax3.set_ylim(0, 1.05)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Timeline plot saved: {output_path}")


def plot_probabilities(
    window_probs: np.ndarray,
    output_path: Path,
    event_group: str = "unknown"
):
    """
    Create probability plot over time.
    
    Args:
        window_probs: (T_windows,) window probabilities
        output_path: Path to save plot
        event_group: Event group name for title
    """
    window_times = np.arange(len(window_probs)) * config.HOP_SEC + config.WINDOW_SEC / 2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(window_times, window_probs, color='blue', linewidth=2, label='Event Probability')
    ax.axhline(y=config.EVAL_THRESHOLD, color='red', linestyle='--', linewidth=1, label=f'Threshold ({config.EVAL_THRESHOLD})')
    ax.fill_between(window_times, 0, window_probs, alpha=0.3, color='blue')
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(f'Window-level Event Probabilities: {event_group}', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Probability plot saved: {output_path}")


def evaluate_sample(
    sample_info: Dict,
    model: BiLSTMEventDetector,
    hear_encoder: HeAREncoder,
    embedding_cache: EmbeddingCache,
    device: torch.device,
    output_dir: Path
) -> Dict:
    """
    Evaluate HeAR+BiLSTM predictions for a single sample.
    
    Args:
        sample_info: Dictionary with wav_path, json_path, event_group
        model: Trained BiLSTM model
        hear_encoder: HeAR encoder instance
        embedding_cache: Embedding cache instance
        device: PyTorch device
        output_dir: Directory to save results
    
    Returns:
        results: Dictionary with metrics and predictions
    """
    wav_path = sample_info['wav_path']
    json_path = sample_info.get('json_path', None)
    event_group = sample_info.get('event_group', 'unknown')
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {event_group}")
    print(f"  Audio: {wav_path}")
    print(f"  JSON: {json_path}")
    print(f"{'='*80}")
    
    # Predict events
    print("\n1. Running HeAR+BiLSTM inference...")
    pred_results = predict_events(
        wav_path,
        model,
        hear_encoder,
        embedding_cache,
        device,
        threshold=config.EVAL_THRESHOLD,
        min_duration_sec=config.MIN_DURATION_SEC
    )
    
    predicted_segments = [(s, e) for s, e in pred_results['segments']]
    window_probs = np.array(pred_results['window_probs'])
    print(f"   Found {len(predicted_segments)} event segments")
    
    # Load ground truth from JSON
    print("\n2. Loading ground truth...")
    if json_path and Path(json_path).exists():
        gt_segments_ms, gt_event_types = load_ground_truth(json_path)
        # Convert to seconds
        gt_segments = [(s/1000.0, e/1000.0) for s, e in gt_segments_ms]
    else:
        print("   Warning: JSON not found, using empty ground truth")
        gt_segments = []
        gt_event_types = []
    
    print(f"   Ground truth: {len(gt_segments)} event segments")
    
    # Compute segment-level metrics
    print("\n3. Computing metrics...")
    segment_metrics = compute_segment_f1(
        window_probs,
        gt_segments,
        config.WINDOW_SEC,
        config.HOP_SEC,
        threshold=config.EVAL_THRESHOLD,
        min_duration_sec=config.MIN_DURATION_SEC
    )
    
    # Create visualizations
    print("\n4. Creating visualizations...")
    sample_output_dir = output_dir / event_group
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    
    timeline_path = sample_output_dir / "timeline.png"
    plot_timeline(
        wav_path,
        gt_segments,
        predicted_segments,
        window_probs,
        timeline_path,
        event_group
    )
    
    prob_plot_path = sample_output_dir / "prob_plot.png"
    plot_probabilities(window_probs, prob_plot_path, event_group)
    
    # Prepare results
    results = {
        'event_group': event_group,
        'audio_path': wav_path,
        'json_path': json_path,
        'duration_sec': pred_results['duration_sec'],
        'num_windows': pred_results['num_windows'],
        'num_pred_segments': len(predicted_segments),
        'num_gt_segments': len(gt_segments),
        'predicted_segments': predicted_segments,
        'gt_segments': gt_segments,
        'window_probs': window_probs.tolist(),
        'mean_event_prob': float(np.mean(window_probs)),
        'metrics': segment_metrics
    }
    
    return results


def generate_summary_report(all_results: List[Dict], output_path: Path):
    """Generate summary report in Markdown format."""
    with open(output_path, 'w') as f:
        f.write("# HeAR + BiLSTM Event Detection Evaluation Report\n\n")
        f.write(f"Generated on: {Path(output_path).stat().st_mtime}\n\n")
        f.write("## Summary\n\n")
        
        # Overall statistics
        total_samples = len(all_results)
        total_pred_segments = sum(r['num_pred_segments'] for r in all_results)
        total_gt_segments = sum(r['num_gt_segments'] for r in all_results)
        
        f.write(f"- **Total samples evaluated**: {total_samples}\n")
        f.write(f"- **Total predicted segments**: {total_pred_segments}\n")
        f.write(f"- **Total ground truth segments**: {total_gt_segments}\n\n")
        
        # Per-sample results
        f.write("## Per-Sample Results\n\n")
        for result in all_results:
            f.write(f"### {result['event_group']}\n\n")
            f.write(f"- **Audio**: `{result['audio_path']}`\n")
            f.write(f"- **Duration**: {result['duration_sec']:.2f}s\n")
            f.write(f"- **Windows**: {result['num_windows']}\n")
            f.write(f"- **Predicted segments**: {result['num_pred_segments']}\n")
            f.write(f"- **Ground truth segments**: {result['num_gt_segments']}\n")
            f.write(f"- **Mean event probability**: {result['mean_event_prob']:.4f}\n\n")
            
            # Metrics
            metrics = result['metrics']
            f.write("**Metrics:**\n")
            f.write(f"- Segment F1: {metrics['segment_f1']:.4f}\n")
            f.write(f"- Segment Precision: {metrics['segment_precision']:.4f}\n")
            f.write(f"- Segment Recall: {metrics['segment_recall']:.4f}\n")
            f.write(f"- TP: {metrics['segment_tp']}, FP: {metrics['segment_fp']}, FN: {metrics['segment_fn']}\n\n")
            
            # Segments
            if result['predicted_segments']:
                f.write("**Predicted segments:**\n")
                for i, (start, end) in enumerate(result['predicted_segments'], 1):
                    f.write(f"  {i}. {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)\n")
            f.write("\n")
        
        # Average metrics
        avg_f1 = np.mean([r['metrics']['segment_f1'] for r in all_results])
        avg_precision = np.mean([r['metrics']['segment_precision'] for r in all_results])
        avg_recall = np.mean([r['metrics']['segment_recall'] for r in all_results])
        
        f.write("## Average Metrics\n\n")
        f.write(f"- **Average Segment F1**: {avg_f1:.4f}\n")
        f.write(f"- **Average Segment Precision**: {avg_precision:.4f}\n")
        f.write(f"- **Average Segment Recall**: {avg_recall:.4f}\n\n")
    
    print(f"  ✓ Summary report saved: {output_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate HeAR + BiLSTM on sample audio files')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint (default: best.pth)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: results/hear_bilstm_evaluation)')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load HeAR encoder
    print("\nLoading HeAR encoder...")
    hear_encoder = HeAREncoder(
        model_name=config.HEAR_MODEL_NAME,
        frozen=True,
        device=device
    )
    
    # Create embedding cache
    embedding_cache = EmbeddingCache(config.EMBEDDING_CACHE_DIR)
    
    # Load model
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else config.BEST_MODEL_PATH
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    model = load_model(checkpoint_path, device)
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = config.EVENT_DETECT_DIR / "results" / "hear_bilstm_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load sample info
    print("\nLoading sample information...")
    samples = load_sample_info()
    
    if not samples:
        print("No samples found. Please create selected_samples.json or provide sample files.")
        return
    
    # Evaluate each sample
    all_results = []
    for sample_key, sample_info in samples.items():
        try:
            results = evaluate_sample(
                sample_info,
                model,
                hear_encoder,
                embedding_cache,
                device,
                output_dir
            )
            all_results.append(results)
        except Exception as e:
            print(f"Error evaluating {sample_key}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results JSON
    results_json_path = output_dir / "results.json"
    with open(results_json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results JSON saved: {results_json_path}")
    
    # Generate summary report
    summary_path = output_dir / "summary_report.md"
    generate_summary_report(all_results, summary_path)
    
    print(f"\n{'='*80}")
    print("Evaluation completed!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
