"""
End-to-end inference using the ensemble system.

Predicts disease from audio events using all 4 models:
    - Model 1: Event Type Classification
    - Model 2: Binary Abnormality Detection
    - Model 3: Disease Group Classification
    - Meta-Model: Random Forest combining all predictions

Usage:
    python predict_ensemble.py --audio_path PATH --start_ms START --end_ms END
    python predict_ensemble.py --csv_path PATH --output_path OUTPUT
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import librosa
import pickle
import argparse
import json
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Make Projects/src/ importable regardless of working directory
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

# Import configs
import config_model1
import config_model2
import config_model3
from models import HeARClassifier
import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EnsemblePredictor:
    """Ensemble prediction system combining all 4 models."""
    
    def __init__(self):
        """Load all models."""
        print("Loading ensemble models...")
        
        # Load Model 1
        self.model1 = HeARClassifier(num_classes=config_model1.NUM_CLASSES).to(DEVICE)
        checkpoint1 = torch.load(config_model1.BEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
        self.model1.load_state_dict(checkpoint1['model_state_dict'])
        self.model1.eval()
        print(f"✓ Model 1 loaded: {config_model1.MODEL_NAME}")
        
        # Load Model 2
        self.model2 = HeARClassifier(num_classes=config_model2.NUM_CLASSES).to(DEVICE)
        checkpoint2 = torch.load(config_model2.BEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
        self.model2.load_state_dict(checkpoint2['model_state_dict'])
        self.model2.eval()
        print(f"✓ Model 2 loaded: {config_model2.MODEL_NAME}")
        
        # Load Model 3
        self.model3 = HeARClassifier(num_classes=config_model3.NUM_CLASSES).to(DEVICE)
        checkpoint3 = torch.load(config_model3.BEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
        self.model3.load_state_dict(checkpoint3['model_state_dict'])
        self.model3.eval()
        print(f"✓ Model 3 loaded: {config_model3.MODEL_NAME}")
        
        # Load Random Forest meta-model
        meta_model_path = Path("models/meta_model/random_forest.pkl")
        with open(meta_model_path, 'rb') as f:
            self.meta_model = pickle.load(f)
        print(f"✓ Meta-model loaded: Random Forest")
        
        print("\n✅ All models loaded successfully!")
    
    def preprocess_audio(
        self,
        audio_path: str,
        event_start_ms: float,
        event_end_ms: float
    ) -> torch.Tensor:
        """
        Load and preprocess audio event.
        
        Args:
            audio_path: Path to WAV file
            event_start_ms: Event start time in milliseconds
            event_end_ms: Event end time in milliseconds
            
        Returns:
            audio_tensor: Preprocessed audio [1, 32000]
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None, mono=False)
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        
        # Resample to 16kHz if needed
        if sr != config.SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=config.SAMPLE_RATE)
            sr = config.SAMPLE_RATE
        
        # Calculate clip range with overlap
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
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(event_clip).float().unsqueeze(0)  # [1, 32000]
        
        return audio_tensor
    
    def predict(
        self,
        audio_path: str,
        event_start_ms: float,
        event_end_ms: float,
        return_details: bool = True
    ) -> Dict:
        """
        Predict disease from audio event.
        
        Args:
            audio_path: Path to WAV file
            event_start_ms: Event start time in milliseconds
            event_end_ms: Event end time in milliseconds
            return_details: If True, return intermediate predictions
            
        Returns:
            results: Dictionary with predictions and probabilities
        """
        # Preprocess audio
        audio = self.preprocess_audio(audio_path, event_start_ms, event_end_ms)
        audio = audio.to(DEVICE)
        
        # Get predictions from all 3 models
        with torch.no_grad():
            # Model 1: Event Type
            logits1 = self.model1(audio)
            probs1 = F.softmax(logits1, dim=1).cpu().numpy()[0]  # [3]
            pred1 = int(np.argmax(probs1))
            
            # Model 2: Binary
            logits2 = self.model2(audio)
            probs2 = F.softmax(logits2, dim=1).cpu().numpy()[0]  # [2]
            pred2 = int(np.argmax(probs2))
            
            # Model 3: Disease
            logits3 = self.model3(audio)
            probs3 = F.softmax(logits3, dim=1).cpu().numpy()[0]  # [3]
            pred3 = int(np.argmax(probs3))
        
        # Concatenate features for meta-model
        ensemble_features = np.concatenate([probs1, probs2, probs3]).reshape(1, -1)  # [1, 8]
        
        # Meta-model prediction
        final_prediction = int(self.meta_model.predict(ensemble_features)[0])
        final_probabilities = self.meta_model.predict_proba(ensemble_features)[0]
        
        # Build results dictionary
        results = {
            'final_prediction': final_prediction,
            'final_prediction_name': config_model3.CLASS_NAMES[final_prediction],
            'final_probabilities': {
                config_model3.CLASS_NAMES[i]: float(final_probabilities[i])
                for i in range(len(final_probabilities))
            }
        }
        
        if return_details:
            results['model1'] = {
                'prediction': pred1,
                'prediction_name': config_model1.CLASS_NAMES[pred1],
                'probabilities': {
                    config_model1.CLASS_NAMES[i]: float(probs1[i])
                    for i in range(len(probs1))
                }
            }
            results['model2'] = {
                'prediction': pred2,
                'prediction_name': config_model2.CLASS_NAMES[pred2],
                'probabilities': {
                    config_model2.CLASS_NAMES[i]: float(probs2[i])
                    for i in range(len(probs2))
                }
            }
            results['model3'] = {
                'prediction': pred3,
                'prediction_name': config_model3.CLASS_NAMES[pred3],
                'probabilities': {
                    config_model3.CLASS_NAMES[i]: float(probs3[i])
                    for i in range(len(probs3))
                }
            }
        
        return results
    
    def predict_batch(
        self,
        csv_path: str,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Predict for all events in a CSV file.
        
        Args:
            csv_path: Path to CSV with columns: wav_path, event_start_ms, event_end_ms
            output_path: Optional path to save results CSV
            
        Returns:
            results_df: DataFrame with predictions
        """
        print(f"\nLoading events from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        required_cols = ['wav_path', 'event_start_ms', 'event_end_ms']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"CSV must contain column: {col}")
        
        print(f"Processing {len(df)} events...")
        
        predictions = []
        
        for idx, row in df.iterrows():
            try:
                result = self.predict(
                    row['wav_path'],
                    row['event_start_ms'],
                    row['event_end_ms'],
                    return_details=False
                )
                
                predictions.append({
                    'index': idx,
                    'final_prediction': result['final_prediction'],
                    'final_prediction_name': result['final_prediction_name'],
                    'prob_pneumonia': result['final_probabilities']['Pneumonia'],
                    'prob_bronchitis_asthma': result['final_probabilities']['Bronchitis-Asthma-Bronchiolitis'],
                    'prob_normal_other': result['final_probabilities']['Normal/Other']
                })
                
                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1}/{len(df)} events...")
                    
            except Exception as e:
                print(f"  Error processing row {idx}: {e}")
                predictions.append({
                    'index': idx,
                    'final_prediction': -1,
                    'final_prediction_name': 'ERROR',
                    'prob_pneumonia': 0.0,
                    'prob_bronchitis_asthma': 0.0,
                    'prob_normal_other': 0.0
                })
        
        # Create results DataFrame
        results_df = pd.DataFrame(predictions)
        results_df = pd.concat([df.reset_index(drop=True), results_df.drop('index', axis=1)], axis=1)
        
        # Save if output path provided
        if output_path:
            results_df.to_csv(output_path, index=False)
            print(f"\n✓ Results saved to: {output_path}")
        
        # Print summary
        print("\nPrediction Summary:")
        print(results_df['final_prediction_name'].value_counts())
        
        return results_df


def main():
    parser = argparse.ArgumentParser(description='Ensemble prediction for respiratory sounds')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'batch'],
                        help='Prediction mode: single event or batch processing')
    parser.add_argument('--audio_path', type=str, help='Path to audio file (single mode)')
    parser.add_argument('--start_ms', type=float, help='Event start time in ms (single mode)')
    parser.add_argument('--end_ms', type=float, help='Event end time in ms (single mode)')
    parser.add_argument('--csv_path', type=str, help='Path to CSV file (batch mode)')
    parser.add_argument('--output_path', type=str, help='Output CSV path (batch mode)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = EnsemblePredictor()
    
    if args.mode == 'single':
        if not all([args.audio_path, args.start_ms, args.end_ms]):
            print("Error: Single mode requires --audio_path, --start_ms, and --end_ms")
            return
        
        print(f"\nPredicting for:")
        print(f"  Audio: {args.audio_path}")
        print(f"  Event: {args.start_ms} - {args.end_ms} ms")
        
        result = predictor.predict(args.audio_path, args.start_ms, args.end_ms)
        
        print("\n" + "="*70)
        print("PREDICTION RESULTS")
        print("="*70)
        
        print(f"\n🎯 Final Prediction: {result['final_prediction_name']}")
        print("\nProbabilities:")
        for name, prob in result['final_probabilities'].items():
            print(f"  {name}: {prob:.4f}")
        
        print("\n" + "-"*70)
        print("Intermediate Model Predictions:")
        print("-"*70)
        
        print(f"\nModel 1 (Event Type): {result['model1']['prediction_name']}")
        for name, prob in result['model1']['probabilities'].items():
            print(f"  {name}: {prob:.4f}")
        
        print(f"\nModel 2 (Abnormality): {result['model2']['prediction_name']}")
        for name, prob in result['model2']['probabilities'].items():
            print(f"  {name}: {prob:.4f}")
        
        print(f"\nModel 3 (Disease): {result['model3']['prediction_name']}")
        for name, prob in result['model3']['probabilities'].items():
            print(f"  {name}: {prob:.4f}")
        
        # Save result to JSON
        output_json = Path(args.audio_path).stem + "_prediction.json"
        with open(output_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Detailed results saved to: {output_json}")
    
    elif args.mode == 'batch':
        if not args.csv_path:
            print("Error: Batch mode requires --csv_path")
            return
        
        output_path = args.output_path or "predictions.csv"
        results_df = predictor.predict_batch(args.csv_path, output_path)
        
        print("\n✅ Batch prediction completed!")


if __name__ == "__main__":
    main()
