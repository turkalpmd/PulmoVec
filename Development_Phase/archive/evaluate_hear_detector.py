"""
Evaluate HeAR's pre-trained event detector on SPRSound dataset.

HeAR's event detector was trained to detect: 
    ['Cough', 'Snore', 'Baby Cough', 'Breathe', 'Sneeze', 'Throat Clear', 'Laugh', 'Speech']

This script evaluates whether these detections can identify respiratory sounds from SPRSound:
    ['Normal', 'Fine Crackle', 'Wheeze', 'Rhonchi', 'Coarse Crackle', 'Wheeze+Crackle', 'Stridor']

Usage:
    python evaluate_hear_detector.py
"""

import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import tensorflow as tf
from huggingface_hub import from_pretrained_keras
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

import config
from sprsound_dataset import SPRSoundDataset


def evaluate_hear_detector():
    """
    Evaluate HeAR's pre-trained event detector on SPRSound events.
    """
    
    print("="*70)
    print("HeAR Event Detector Evaluation on SPRSound Dataset")
    print("="*70)
    
    # Load HuggingFace token
    if config.ENV_FILE.exists():
        load_dotenv(config.ENV_FILE)
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token
            print("✓ HuggingFace token loaded")
    
    # Load HeAR event detector (TensorFlow/Keras model)
    print("\nLoading HeAR event detector model...")
    print("Note: This is the TensorFlow version with event detector")
    
    try:
        # Try to load the local snapshot if available
        local_snapshot_path = os.path.expanduser("~/.cache/huggingface/hub/models--google--hear/snapshots/")
        if os.path.exists(local_snapshot_path):
            # Find the latest snapshot
            snapshots = [d for d in os.listdir(local_snapshot_path) if os.path.isdir(os.path.join(local_snapshot_path, d))]
            if snapshots:
                latest_snapshot = os.path.join(local_snapshot_path, snapshots[0])
                print(f"Using local snapshot: {latest_snapshot}")
                
                # Load HeAR model
                hear_model = from_pretrained_keras(latest_snapshot)
                
                # Load event detector
                detector_path = os.path.join(latest_snapshot, "event_detector", "event_detector_small")
                if os.path.exists(detector_path):
                    event_detector = from_pretrained_keras(detector_path)
                    print("✓ Event detector loaded from local cache")
                else:
                    print("⚠ Event detector not found in local cache, downloading...")
                    # This will download if not available
                    event_detector = None
            else:
                event_detector = None
        else:
            event_detector = None
    except Exception as e:
        print(f"⚠ Could not load from local cache: {e}")
        event_detector = None
    
    if event_detector is None:
        print("\n⚠ HeAR event detector requires TensorFlow model with event_detector component")
        print("This is available via: huggingface.co/google/hear")
        print("\nNote: The PyTorch version (google/hear-pytorch) is for embeddings only.")
        print("For event detection evaluation, you need the full TensorFlow model.")
        print("\nTo get this:")
        print("  1. Download the full model from HuggingFace")
        print("  2. Or use the hear_event_detector_demo.ipynb notebook")
        
        # Fallback: Just analyze the dataset itself
        print("\n" + "="*70)
        print("Analyzing SPRSound Dataset Labels")
        print("="*70)
        
        dataset = SPRSoundDataset()
        
        print(f"\nTotal events: {len(dataset)}")
        print("\nEvent type distribution:")
        for class_name in config.CLASS_NAMES:
            count = len(dataset.df[dataset.df['event_type'] == class_name])
            percentage = 100 * count / len(dataset)
            print(f"  {class_name}: {count} ({percentage:.2f}%)")
        
        # Analyze by recording location
        print("\nEvents by recording location:")
        location_dist = dataset.df.groupby('recording_location_name')['event_type'].value_counts()
        print(location_dist)
        
        # Save report
        report_path = config.DETECTOR_EVAL_REPORT
        with open(report_path, 'w') as f:
            f.write("# HeAR Event Detector Evaluation Report\n\n")
            f.write("## Summary\n\n")
            f.write("HeAR's pre-trained event detector is available in the TensorFlow model.\n")
            f.write("This script attempted to evaluate it but requires the full model with event detector.\n\n")
            f.write("## SPRSound Dataset Analysis\n\n")
            f.write(f"Total events: {len(dataset)}\n\n")
            f.write("### Event Type Distribution\n\n")
            f.write("| Event Type | Count | Percentage |\n")
            f.write("|------------|-------|------------|\n")
            for class_name in config.CLASS_NAMES:
                count = len(dataset.df[dataset.df['event_type'] == class_name])
                percentage = 100 * count / len(dataset)
                f.write(f"| {class_name} | {count} | {percentage:.2f}% |\n")
            
            f.write("\n### Analysis\n\n")
            f.write("**HeAR Detector Labels**: Cough, Snore, Baby Cough, Breathe, Sneeze, Throat Clear, Laugh, Speech\n\n")
            f.write("**SPRSound Labels**: Normal, Fine Crackle, Wheeze, Rhonchi, Coarse Crackle, Wheeze+Crackle, Stridor\n\n")
            f.write("**Mapping Hypothesis**:\n")
            f.write("- SPRSound 'Normal' breathing might trigger HeAR's 'Breathe' detector\n")
            f.write("- SPRSound adventitious sounds (crackles, wheezes) are specialized respiratory events\n")
            f.write("- HeAR's detector may not have specific labels for these pathological sounds\n")
            f.write("- This is why we need to **fine-tune** HeAR for SPRSound classification\n\n")
            f.write("### Recommendation\n\n")
            f.write("1. Use the classification model (train_hear_classifier.py) for SPRSound\n")
            f.write("2. HeAR's embeddings are more valuable than its detector for this task\n")
            f.write("3. Fine-tuning allows the model to learn respiratory-specific patterns\n")
        
        print(f"\n✓ Report saved to: {report_path}")
        return
    
    # If we have the detector, run full evaluation
    print("\n✓ Event detector loaded successfully")
    print(f"Detector labels: {config.HEAR_DETECTOR_LABELS}")
    
    # Load SPRSound dataset
    print("\nLoading SPRSound dataset...")
    dataset = SPRSoundDataset()
    
    # Sample a subset for evaluation (to save time)
    sample_size = min(1000, len(dataset))
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    print(f"Evaluating on {sample_size} samples...")
    
    # Run detection on all samples
    results = []
    
    for idx in tqdm(indices, desc="Running HeAR detector"):
        audio, label, metadata = dataset[idx]
        
        # Prepare audio for detector (needs batch dimension)
        audio_np = audio.numpy().reshape(1, -1)
        
        # Run detector
        detection_scores = event_detector(audio_np)
        
        # Get scores for each label
        scores = detection_scores['scores'].numpy()[0]
        
        # Record results
        result = {
            'sprsound_label': dataset.idx_to_class[label],
            'sprsound_idx': label
        }
        
        # Add HeAR detector scores
        for i, hear_label in enumerate(config.HEAR_DETECTOR_LABELS):
            result[f'hear_{hear_label.lower().replace(" ", "_")}'] = scores[i]
        
        # Check which HeAR labels exceeded threshold
        detections = [
            config.HEAR_DETECTOR_LABELS[i] 
            for i in range(len(config.HEAR_DETECTOR_LABELS))
            if scores[i] > config.DETECTION_THRESHOLD
        ]
        result['hear_detections'] = ','.join(detections) if detections else 'none'
        
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Analyze results
    print("\n" + "="*70)
    print("Analysis Results")
    print("="*70)
    
    # Detection rate per SPRSound label
    print("\nDetection rates per SPRSound event type:")
    print("(Percentage of samples where HeAR detected 'Breathe' label)\n")
    
    for class_name in config.CLASS_NAMES:
        subset = results_df[results_df['sprsound_label'] == class_name]
        if len(subset) > 0:
            breathe_rate = (subset['hear_breathe'] > config.DETECTION_THRESHOLD).mean()
            avg_breathe_score = subset['hear_breathe'].mean()
            print(f"  {class_name:20s}: {breathe_rate*100:5.1f}% (avg score: {avg_breathe_score:.3f})")
    
    # Most common HeAR detections per SPRSound label
    print("\nMost common HeAR detections per SPRSound label:\n")
    for class_name in config.CLASS_NAMES:
        subset = results_df[results_df['sprsound_label'] == class_name]
        if len(subset) > 0:
            detections = subset['hear_detections'].value_counts().head(3)
            print(f"  {class_name}:")
            for detection, count in detections.items():
                percentage = 100 * count / len(subset)
                print(f"    {detection:30s}: {count:3d} ({percentage:5.1f}%)")
    
    # Save detailed report
    report_path = config.DETECTOR_EVAL_REPORT
    with open(report_path, 'w') as f:
        f.write("# HeAR Event Detector Evaluation Report\n\n")
        f.write("## Summary\n\n")
        f.write(f"Evaluated HeAR event detector on {sample_size} SPRSound samples.\n\n")
        f.write("### HeAR Detector Labels\n")
        f.write(", ".join(config.HEAR_DETECTOR_LABELS) + "\n\n")
        f.write("### SPRSound Labels\n")
        f.write(", ".join(config.CLASS_NAMES) + "\n\n")
        
        f.write("## Detection Results\n\n")
        f.write("### 'Breathe' Detection Rate by SPRSound Event Type\n\n")
        f.write("| SPRSound Label | Detection Rate | Avg Score |\n")
        f.write("|----------------|----------------|------------|\n")
        for class_name in config.CLASS_NAMES:
            subset = results_df[results_df['sprsound_label'] == class_name]
            if len(subset) > 0:
                breathe_rate = (subset['hear_breathe'] > config.DETECTION_THRESHOLD).mean()
                avg_score = subset['hear_breathe'].mean()
                f.write(f"| {class_name} | {breathe_rate*100:.1f}% | {avg_score:.3f} |\n")
        
        f.write("\n## Insights\n\n")
        f.write("1. HeAR's event detector was trained on general health acoustics\n")
        f.write("2. It may detect 'Breathe' for respiratory sounds but lacks specialized labels\n")
        f.write("3. For SPRSound classification, **fine-tuning the HeAR encoder** is more effective\n")
        f.write("4. The embeddings capture acoustic patterns that can be learned for specific tasks\n")
    
    print(f"\n✓ Detailed report saved to: {report_path}")
    
    # Save results CSV
    results_csv_path = config.MODEL_DIR / "hear_detector_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"✓ Results CSV saved to: {results_csv_path}")


if __name__ == "__main__":
    evaluate_hear_detector()
