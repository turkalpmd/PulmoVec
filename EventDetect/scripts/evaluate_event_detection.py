"""Main script for evaluating event detection and classification.

Uses TensorFlow HeAR event detector to find event segments, then PyTorch Model 1
to classify event types (Normal, Crackles, Rhonchi).
"""

import json
import numpy as np
from pathlib import Path
import sys

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT / "EventDetect"))

# Import EventDetect config using absolute path to avoid conflicts
import importlib.util
spec = importlib.util.spec_from_file_location("eventdetect_config", PROJECT_ROOT / "EventDetect" / "config.py")
eventdetect_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eventdetect_config)

SELECTED_SAMPLES_JSON = eventdetect_config.SELECTED_SAMPLES_JSON
RESULTS_DIR = eventdetect_config.RESULTS_DIR
BOUNDARY_TOLERANCES = eventdetect_config.BOUNDARY_TOLERANCES
IOU_THRESHOLDS = eventdetect_config.IOU_THRESHOLDS
CLIP_DURATION = eventdetect_config.CLIP_DURATION
CLIP_OVERLAP_PERCENT = eventdetect_config.CLIP_OVERLAP_PERCENT
SAMPLE_RATE = eventdetect_config.SAMPLE_RATE
ENSEMBLE_CSV_PATH = PROJECT_ROOT / "data" / "SPRSound_Event_Level_Ensemble_Dataset.csv"

# Model 1 classes: Normal, Crackles, Rhonchi
LABEL_LIST = ['Normal', 'Crackles', 'Rhonchi']

from scripts.detect_events import detect_event_segments_only, detect_event_segments_pytorch_model1, detect_events_with_classification, load_tensorflow_event_detector, load_pytorch_classifier
from scripts.load_ground_truth import load_ground_truth, get_recording_duration
from scripts.visualize_results import plot_timeline, plot_detection_scores, plot_metrics_summary
from evaluation.temporal_metrics import evaluate_segmentation


def evaluate_sample(sample_info: dict, tf_event_detector, pytorch_classifier, device, output_dir: Path, ensemble_csv_path: Path, use_pytorch_binary=False, pytorch_binary_model=None):
    """
    Evaluate event detection and classification for a single sample.
    
    Args:
        sample_info: Dictionary with wav_path, json_path, etc.
        tf_event_detector: Pre-loaded TensorFlow event detector
        pytorch_classifier: Pre-loaded PyTorch Model 1 classifier
        device: PyTorch device
        output_dir: Directory to save results
        ensemble_csv_path: Path to ensemble CSV with model1_label
        
    Returns:
        Dictionary with metrics and visualization data
    """
    event_group = sample_info.get('event_group', 'unknown')
    wav_path = sample_info['wav_path']
    json_path = sample_info['json_path']
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {event_group}")
    print(f"  WAV: {wav_path}")
    print(f"  JSON: {json_path}")
    print(f"{'='*60}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Detect events (segmentation only for now)
    if pytorch_classifier is not None:
        print("\n1. Running TensorFlow event detector + PyTorch Model 1 classification...")
        predicted_segments_dicts, detection_scores = detect_events_with_classification(
            wav_path, 
            tf_event_detector=tf_event_detector,
            pytorch_classifier=pytorch_classifier,
            device=device
        )
        # Convert to simple format for evaluation
        predicted_segments = [(seg['start'], seg['end']) for seg in predicted_segments_dicts]
        predicted_event_types = [seg['event_type'] for seg in predicted_segments_dicts]
    elif use_pytorch_binary:
        print("\n1. Running PyTorch Model 1 (Event Type) for event detection...")
        predicted_segments, detection_scores, detection_results = detect_event_segments_pytorch_model1(
            wav_path,
            pytorch_model=pytorch_binary_model,  # Actually Model 1
            device=device
        )
        # Extract predicted event types from detection results
        if detection_results and 'clip_details' in detection_results:
            predicted_event_types = [cd.get('predicted_class', 'Unknown') for cd in detection_results['clip_details'] 
                                   if cd.get('is_event', False)]
            # Map to segments (simplified - use most common class in segment)
            predicted_event_types = [detection_results['clip_details'][i].get('predicted_class', 'Unknown') 
                                   for i in range(len(detection_results['clip_details'])) 
                                   if detection_results['clip_details'][i].get('is_event', False)]
        else:
            predicted_event_types = None
        
        # Save detection results to JSON
        detection_json_path = output_dir / "detection_results.json"
        import json
        with open(detection_json_path, 'w') as f:
            json.dump(detection_results, f, indent=2)
        print(f"   ✓ Saved detection results: {detection_json_path}")
    else:
        print("\n1. Running TensorFlow event detector (segmentation only)...")
        result = detect_event_segments_only(
            wav_path,
            tf_event_detector=tf_event_detector
        )
        if len(result) == 3:
            predicted_segments, detection_scores, detection_results = result
            # Save detection results to JSON
            detection_json_path = output_dir / "detection_results.json"
            import json
            with open(detection_json_path, 'w') as f:
                json.dump(detection_results, f, indent=2)
            print(f"   ✓ Saved detection results: {detection_json_path}")
        else:
            predicted_segments, detection_scores = result
            detection_results = None
        predicted_event_types = None  # No classification
    
    # 2. Load ground truth from CSV using model1_label
    print("\n2. Loading ground truth from CSV (model1_label)...")
    if ensemble_csv_path and ensemble_csv_path.exists():
        gt_segments, gt_event_types = load_ground_truth(json_path, csv_path=str(ensemble_csv_path))
    else:
        print("   Warning: Using JSON event types (model1_label not available)")
        gt_segments, gt_event_types = load_ground_truth(json_path)
    print(f"   Found {len(gt_segments)} ground truth events")
    if len(gt_event_types) > 0:
        print(f"   Event types: {gt_event_types}")
    
    # 3. Get recording duration
    if ensemble_csv_path and ensemble_csv_path.exists():
        recording_duration = get_recording_duration(json_path, csv_path=str(ensemble_csv_path))
    else:
        recording_duration = get_recording_duration(json_path)
    
    # 4. Evaluate
    print("\n3. Computing evaluation metrics...")
    metrics = evaluate_segmentation(
        gt_segments,
        predicted_segments,
        tolerances=BOUNDARY_TOLERANCES,
        iou_thresholds=IOU_THRESHOLDS,
        recording_duration_seconds=recording_duration
    )
    
    # Print key metrics
    print(f"\n   Key Metrics:")
    print(f"   - Mean IoU: {metrics['mean_iou']:.3f}")
    print(f"   - Median IoU: {metrics['median_iou']:.3f}")
    if 'precision_recall_f1' in metrics and '0.5' in metrics['precision_recall_f1']:
        prf1 = metrics['precision_recall_f1']['0.5']
        print(f"   - Precision @ IoU=0.5: {prf1['precision']:.3f}")
        print(f"   - Recall @ IoU=0.5: {prf1['recall']:.3f}")
        print(f"   - F1 @ IoU=0.5: {prf1['f1']:.3f}")
    print(f"   - Median Onset Error: {metrics['median_onset_error']:.3f}s")
    print(f"   - Median Offset Error: {metrics['median_offset_error']:.3f}s")
    
    # 5. Generate visualizations
    print("\n4. Generating visualizations...")
    
    # Timeline plot
    timeline_path = output_dir / "timeline.png"
    title = f"Event Detection & Classification: {event_group}" if predicted_event_types else f"Event Detection (Segmentation): {event_group}"
    plot_timeline(
        wav_path,
        gt_segments,
        predicted_segments,
        gt_event_types=gt_event_types,
        pred_event_types=predicted_event_types,
        save_path=str(timeline_path),
        title=title
    )
    
    # Detection scores plot
    import librosa
    audio_viz, sr_viz = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    frame_length = int(CLIP_DURATION * SAMPLE_RATE)
    frame_step = int(frame_length * (1 - CLIP_OVERLAP_PERCENT / 100))
    num_clips = detection_scores.shape[0]
    
    # Generate clip start times
    clip_start_times = []
    i = 0
    while i * frame_step < len(audio_viz):
        start_sample = i * frame_step
        start_sec = start_sample / SAMPLE_RATE
        end_sec = min(start_sec + CLIP_DURATION, len(audio_viz) / sr_viz)
        if end_sec - start_sec >= CLIP_DURATION * 0.5:
            clip_start_times.append(start_sec)
        i += 1
    
    scores_path = output_dir / "detection_scores.png"
    
    if use_pytorch_binary:
        # PyTorch Model 1 outputs 3 classes: Normal, Crackles, Rhonchi
        label_list = ['Normal', 'Crackles', 'Rhonchi']
        detector_name = "PyTorch Model 1"
    else:
        # TensorFlow HeAR event detector outputs 8 labels
        label_list = ['Cough', 'Snore', 'Baby Cough', 'Breathe', 'Sneeze', 'Throat Clear', 'Laugh', 'Speech']
        detector_name = "TensorFlow HeAR"
    
    plot_detection_scores(
        detection_scores,
        clip_start_times[:len(detection_scores)],
        label_list,
        threshold=0.5,
        save_path=str(scores_path),
        title=f"{detector_name} Detection Scores: {event_group}"
    )
    
    # Metrics summary
    metrics_path = output_dir / "metrics_summary.png"
    plot_metrics_summary(
        metrics,
        save_path=str(metrics_path),
        title=f"Evaluation Metrics: {event_group}"
    )
    
    # 6. Save metrics to JSON
    metrics_path_json = output_dir / "metrics.json"
    # Convert numpy types to native Python types
    metrics_serializable = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            metrics_serializable[k] = {
                k2: float(v2) if isinstance(v2, (np.integer, np.floating)) else v2
                for k2, v2 in v.items()
            }
        elif isinstance(v, (np.integer, np.floating)):
            metrics_serializable[k] = float(v)
        else:
            metrics_serializable[k] = v
    
    with open(metrics_path_json, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    print(f"   ✓ Saved metrics: {metrics_path_json}")
    
    return {
        'metrics': metrics,
        'gt_segments': gt_segments,
        'predicted_segments': predicted_segments,
        'gt_event_types': gt_event_types,
        'predicted_event_types': predicted_event_types
    }


def main():
    """Main evaluation function."""
    print("="*60)
    print("Event Detection & Classification Evaluation")
    print("TensorFlow HeAR Event Detector + PyTorch Model 1 Classifier")
    print("="*60)
    
    # Load selected samples
    if not SELECTED_SAMPLES_JSON.exists():
        print(f"\n❌ Selected samples file not found: {SELECTED_SAMPLES_JSON}")
        print("   Please run: python EventDetect/scripts/select_samples.py")
        return
    
    with open(SELECTED_SAMPLES_JSON, 'r') as f:
        samples = json.load(f)
    
    print(f"\nLoaded {len(samples)} samples")
    
    # Use PyTorch Model 1 (Event Type) for event detection - provides better outputs
    USE_PYTORCH_MODEL1 = True  # Use PyTorch Model 1 for segmentation
    USE_CLASSIFICATION = False  # Set to True to enable additional PyTorch Model 1 classification (redundant if using Model 1 for detection)
    
    if USE_PYTORCH_MODEL1:
        print("\nUsing PyTorch Model 1 (Event Type: Normal, Crackles, Rhonchi) for event detection...")
        # Load PyTorch Model 1 once
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        import config_model1
        from models import HeARClassifier
        import torch
        
        print(f"Loading PyTorch Model 1 from: {config_model1.BEST_MODEL_PATH}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pytorch_model1 = HeARClassifier(num_classes=config_model1.NUM_CLASSES)
        checkpoint = torch.load(config_model1.BEST_MODEL_PATH, map_location='cpu', weights_only=False)
        pytorch_model1.load_state_dict(checkpoint['model_state_dict'])
        pytorch_model1.to(device)
        pytorch_model1.eval()
        print("✓ PyTorch Model 1 loaded")
        print(f"  Classes: {config_model1.CLASS_NAMES}")
        tf_event_detector = None
        pytorch_binary_model = None
    else:
        print("\nLoading TensorFlow event detector (CPU mode)...")
        tf_event_detector = load_tensorflow_event_detector()
        pytorch_model1 = None
        pytorch_binary_model = None
        device = None
    
    if USE_CLASSIFICATION:
        # Clear GPU cache before loading PyTorch model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("  GPU cache cleared before loading PyTorch model")
        
        print("\nLoading PyTorch Model 1 (Event Type Classifier)...")
        pytorch_classifier, device = load_pytorch_classifier()
    else:
        pytorch_classifier = None
        device = None
        print("\n⚠️  Classification disabled - testing segmentation only")
    
    # Load ensemble CSV for ground truth
    if not ENSEMBLE_CSV_PATH.exists():
        print(f"\n⚠️  Warning: Ensemble CSV not found at {ENSEMBLE_CSV_PATH}")
        print("   Will use JSON event types instead of model1_label")
        ensemble_csv_path = None
    else:
        ensemble_csv_path = ENSEMBLE_CSV_PATH
        print(f"\n✓ Using ensemble CSV for ground truth (model1_label): {ENSEMBLE_CSV_PATH}")
    
    # Evaluate each sample
    all_results = {}
    all_metrics = []
    
    for event_group, sample_info in samples.items():
        sample_info['event_group'] = event_group
        output_dir = RESULTS_DIR / f"sample_{event_group}"
        
        try:
            result = evaluate_sample(
                sample_info, 
                tf_event_detector, 
                pytorch_classifier if USE_CLASSIFICATION else None, 
                device if (USE_CLASSIFICATION or USE_PYTORCH_MODEL1) else None, 
                output_dir,
                ensemble_csv_path,
                use_pytorch_binary=USE_PYTORCH_MODEL1,
                pytorch_binary_model=pytorch_model1 if USE_PYTORCH_MODEL1 else None
            )
            all_results[event_group] = result
            all_metrics.append(result['metrics'])
        except Exception as e:
            print(f"\n❌ Error evaluating {event_group}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate summary report
    print("\n" + "="*60)
    print("Generating Summary Report")
    print("="*60)
    
    summary_path = RESULTS_DIR / "summary_report.md"
    with open(summary_path, 'w') as f:
        f.write("# Event Detection & Classification Evaluation Summary\n\n")
        f.write(f"Evaluated {len(all_results)} samples.\n\n")
        f.write("**Pipeline:**\n")
        f.write("- TensorFlow HeAR Event Detector: Finds event segments\n")
        f.write("- PyTorch Model 1: Classifies event types (Normal, Crackles, Rhonchi)\n")
        f.write("- Ground Truth: model1_label from ensemble CSV\n\n")
        
        f.write("## Overall Metrics (Averaged)\n\n")
        
        if len(all_metrics) > 0:
            # Average metrics
            mean_iou = np.mean([m['mean_iou'] for m in all_metrics])
            median_iou = np.mean([m['median_iou'] for m in all_metrics])
            
            f.write(f"- **Mean IoU**: {mean_iou:.3f}\n")
            f.write(f"- **Median IoU**: {median_iou:.3f}\n")
            
            # Precision/Recall/F1 at IoU=0.5
            if all('precision_recall_f1' in m and '0.5' in m['precision_recall_f1'] for m in all_metrics):
                precisions = [m['precision_recall_f1']['0.5']['precision'] for m in all_metrics]
                recalls = [m['precision_recall_f1']['0.5']['recall'] for m in all_metrics]
                f1s = [m['precision_recall_f1']['0.5']['f1'] for m in all_metrics]
                
                f.write(f"- **Precision @ IoU=0.5**: {np.mean(precisions):.3f}\n")
                f.write(f"- **Recall @ IoU=0.5**: {np.mean(recalls):.3f}\n")
                f.write(f"- **F1 @ IoU=0.5**: {np.mean(f1s):.3f}\n")
            
            # Boundary errors
            onset_errors = [m['median_onset_error'] for m in all_metrics]
            offset_errors = [m['median_offset_error'] for m in all_metrics]
            
            f.write(f"- **Median Onset Error**: {np.mean(onset_errors):.3f}s\n")
            f.write(f"- **Median Offset Error**: {np.mean(offset_errors):.3f}s\n")
        
        f.write("\n## Per-Sample Results\n\n")
        for event_group, result in all_results.items():
            f.write(f"### {event_group}\n\n")
            metrics = result['metrics']
            f.write(f"- Mean IoU: {metrics['mean_iou']:.3f}\n")
            f.write(f"- Ground Truth Events: {metrics['num_ground_truth']}\n")
            f.write(f"- Predicted Events: {metrics['num_predictions']}\n")
            f.write(f"- Matched Events: {metrics['num_matched']}\n")
            if 'predicted_event_types' in result:
                f.write(f"- Predicted Types: {result['predicted_event_types']}\n")
            if 'gt_event_types' in result:
                f.write(f"- Ground Truth Types: {result['gt_event_types']}\n")
            f.write(f"- Results saved to: `results/sample_{event_group}/`\n\n")
    
    print(f"✓ Summary report saved: {summary_path}")
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
