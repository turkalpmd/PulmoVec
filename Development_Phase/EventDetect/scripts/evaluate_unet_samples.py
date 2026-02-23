"""
Evaluate U-Net temporal segmentation on 3 sample audio files.
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

# Add EventDetect to path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from models.unet_segmentation import UNetTemporalSegmentation
from scripts.inference_unet import load_model, predict_events
# Import from EventDetect/scripts
sys.path.insert(0, str(Path(__file__).parent.absolute()))
from load_ground_truth import load_ground_truth, get_recording_duration
from evaluation.temporal_metrics import evaluate_segmentation
import config_unet as config


def load_sample_info() -> Dict:
    """Load sample information from selected_samples.json."""
    samples_json = config.EVENT_DETECT_DIR / "samples" / "selected_samples.json"
    
    if not samples_json.exists():
        print(f"Error: Sample info not found at {samples_json}")
        print("Please run EventDetect/scripts/select_samples.py first")
        return {}
    
    with open(samples_json, 'r') as f:
        samples = json.load(f)
    
    return samples


def evaluate_sample(
    sample_info: Dict,
    model: UNetTemporalSegmentation,
    device: torch.device,
    output_dir: Path
) -> Dict:
    """
    Evaluate U-Net predictions for a single sample.
    
    Args:
        sample_info: Dictionary with wav_path, json_path, event_group
        model: Trained U-Net model
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
    
    # Predict events using U-Net
    print("\n1. Running U-Net inference...")
    pred_results = predict_events(
        wav_path,
        model,
        device,
        threshold=0.5,
        min_segment_duration=0.1
    )
    
    predicted_segments = pred_results['segments']
    print(f"   Found {len(predicted_segments)} event segments")
    
    # Load ground truth from JSON
    print("\n2. Loading ground truth...")
    if json_path and Path(json_path).exists():
        gt_segments, gt_event_types = load_ground_truth(json_path)
        # Convert to seconds (ground truth is in milliseconds)
        gt_segments_sec = [(s/1000.0, e/1000.0) for s, e in gt_segments]
    else:
        print("   Warning: JSON not found, using empty ground truth")
        gt_segments_sec = []
        gt_event_types = []
    
    print(f"   Ground truth: {len(gt_segments_sec)} event segments")
    
    # Get recording duration
    duration_sec = pred_results['duration_sec']
    
    # Evaluate segmentation
    print("\n3. Computing metrics...")
    metrics = evaluate_segmentation(
        gt_segments_sec,
        predicted_segments,
        tolerances=[0.05, 0.1, 0.25],
        iou_thresholds=[0.3, 0.5, 0.7]
    )
    
    # Prepare results
    results = {
        'event_group': event_group,
        'audio_path': wav_path,
        'json_path': json_path,
        'duration_sec': duration_sec,
        'ground_truth_segments': gt_segments_sec,
        'predicted_segments': predicted_segments,
        'metrics': metrics,
        'prediction_stats': {
            'num_segments': len(predicted_segments),
            'mean_event_prob': pred_results['mean_event_prob'],
            'max_event_prob': pred_results['max_event_prob'],
            'min_event_prob': pred_results['min_event_prob']
        }
    }
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_json = output_dir / "results.json"
    with open(results_json, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_results = convert_to_serializable(results)
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_json}")
    
    # Print summary
    print(f"\n📊 Metrics Summary:")
    print(f"   Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"   Median IoU: {metrics['median_iou']:.4f}")
    print(f"   Precision @ IoU 0.5: {metrics['precision_at_iou']['0.5']:.4f}")
    print(f"   Recall @ IoU 0.5: {metrics['recall_at_iou']['0.5']:.4f}")
    print(f"   F1 @ IoU 0.5: {metrics['f1_at_iou']['0.5']:.4f}")
    print(f"   False Positives/Hour: {metrics['false_positives_per_hour']:.2f}")
    
    return results


def plot_timeline_comparison(
    results: Dict,
    output_path: Path
):
    """Plot timeline comparison of ground truth vs predictions with audio waveform."""
    gt_segments = results['ground_truth_segments']
    pred_segments = results['predicted_segments']
    duration = results['duration_sec']
    event_group = results['event_group']
    audio_path = results['audio_path']
    
    # Load audio waveform
    try:
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        time_axis = np.linspace(0, duration, len(audio))
    except Exception as e:
        print(f"   Warning: Could not load audio for waveform: {e}")
        audio = None
        time_axis = None
    
    # Create figure with two subplots: waveform on top, segments below
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1])
    
    # Top plot: Audio waveform
    if audio is not None:
        ax1.plot(time_axis, audio, color='gray', linewidth=0.5, alpha=0.7)
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'Audio Waveform: {event_group}')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, duration)
    
    # Bottom plot: Event segments
    # Plot ground truth
    for i, (start, end) in enumerate(gt_segments):
        ax2.barh(0, end - start, left=start, height=0.4, color='green', alpha=0.7, label='GT' if i == 0 else '')
    
    # Plot predictions
    for i, (start, end) in enumerate(pred_segments):
        ax2.barh(1, end - start, left=start, height=0.4, color='orange', alpha=0.7, label='Predicted' if i == 0 else '')
    
    ax2.set_xlim(0, duration)
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Ground Truth', 'Predicted'])
    ax2.set_xlabel('Time (seconds)')
    ax2.set_title('Event Detection & Classification')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Align x-axes
    if audio is not None:
        ax1.set_xticks(ax2.get_xticks())
        ax1.set_xticklabels([])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Timeline plot saved to: {output_path}")


def plot_temporal_probabilities(
    results: Dict,
    output_path: Path
):
    """Plot temporal event probabilities."""
    pred_results = results.get('prediction_stats', {})
    duration = results['duration_sec']
    
    # Load temporal probabilities from prediction results if available
    # For now, we'll create a simple visualization
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # This would require loading the full temporal mask from inference
    # For now, just show segment boundaries
    pred_segments = results['predicted_segments']
    
    for start, end in pred_segments:
        ax.axvspan(start, end, alpha=0.3, color='orange')
    
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Event Probability')
    ax.set_title(f'Event Detection Probabilities: {results["event_group"]}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Probability plot saved to: {output_path}")


def main():
    """Main evaluation function."""
    print("=" * 80)
    print("U-Net Temporal Segmentation Evaluation")
    print("=" * 80)
    
    # Device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    checkpoint_path = config.BEST_MODEL_PATH
    if not checkpoint_path.exists():
        print(f"\n⚠️  Warning: Model checkpoint not found at {checkpoint_path}")
        print("   Please train the model first using: python EventDetect/scripts/train_unet.py")
        return
    
    print(f"\nLoading model from: {checkpoint_path}")
    model = load_model(checkpoint_path, device)
    
    # Load sample information
    samples = load_sample_info()
    if not samples:
        return
    
    # Results directory
    results_dir = config.EVENT_DETECT_DIR / "results" / "unet_evaluation"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate each sample
    all_results = {}
    all_metrics = []
    
    for event_group, sample_info in samples.items():
        sample_info['event_group'] = event_group
        output_dir = results_dir / f"sample_{event_group}"
        
        try:
            result = evaluate_sample(sample_info, model, device, output_dir)
            all_results[event_group] = result
            all_metrics.append(result['metrics'])
            
            # Generate plots
            plot_timeline_comparison(result, output_dir / "timeline.png")
            plot_temporal_probabilities(result, output_dir / "probabilities.png")
            
        except Exception as e:
            print(f"\n❌ Error evaluating {event_group}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate summary report
    if all_metrics:
        print("\n" + "=" * 80)
        print("Summary Report")
        print("=" * 80)
        
        mean_iou = np.mean([m['mean_iou'] for m in all_metrics])
        mean_precision = np.mean([m['precision_at_iou']['0.5'] for m in all_metrics])
        mean_recall = np.mean([m['recall_at_iou']['0.5'] for m in all_metrics])
        mean_f1 = np.mean([m['f1_at_iou']['0.5'] for m in all_metrics])
        
        print(f"\nAverage Metrics (across {len(all_metrics)} samples):")
        print(f"  Mean IoU: {mean_iou:.4f}")
        print(f"  Precision @ IoU 0.5: {mean_precision:.4f}")
        print(f"  Recall @ IoU 0.5: {mean_recall:.4f}")
        print(f"  F1 @ IoU 0.5: {mean_f1:.4f}")
        
        # Save summary
        summary_path = results_dir / "summary_report.md"
        with open(summary_path, 'w') as f:
            f.write("# U-Net Temporal Segmentation Evaluation Summary\n\n")
            f.write(f"## Average Metrics\n\n")
            f.write(f"- **Mean IoU**: {mean_iou:.4f}\n")
            f.write(f"- **Precision @ IoU 0.5**: {mean_precision:.4f}\n")
            f.write(f"- **Recall @ IoU 0.5**: {mean_recall:.4f}\n")
            f.write(f"- **F1 @ IoU 0.5**: {mean_f1:.4f}\n\n")
            f.write(f"## Per-Sample Results\n\n")
            for event_group, result in all_results.items():
                metrics = result['metrics']
                f.write(f"### {event_group}\n\n")
                f.write(f"- Mean IoU: {metrics['mean_iou']:.4f}\n")
                f.write(f"- Precision @ IoU 0.5: {metrics['precision_at_iou']['0.5']:.4f}\n")
                f.write(f"- Recall @ IoU 0.5: {metrics['recall_at_iou']['0.5']:.4f}\n")
                f.write(f"- F1 @ IoU 0.5: {metrics['f1_at_iou']['0.5']:.4f}\n\n")
        
        print(f"\n✓ Summary report saved to: {summary_path}")
    
    print("\n" + "=" * 80)
    print("Evaluation completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
