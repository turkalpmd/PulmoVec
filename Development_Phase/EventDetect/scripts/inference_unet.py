"""
Inference script for U-Net temporal event segmentation.
Predicts event segments from audio files.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import librosa
import json
from typing import List, Tuple, Dict

# Add EventDetect to path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from models.unet_segmentation import UNetTemporalSegmentation
import config_unet as config


def load_model(checkpoint_path: Path, device: torch.device) -> UNetTemporalSegmentation:
    """Load trained U-Net model from checkpoint."""
    model = UNetTemporalSegmentation(
        n_channels=1,
        n_freq_bins=config.N_MELS,
        bilinear=True
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded from: {checkpoint_path}")
    return model


def preprocess_audio(audio_path: str) -> Tuple[torch.Tensor, float]:
    """
    Load audio and convert to Mel-Spectrogram.
    
    Returns:
        spectrogram: (1, 1, time_steps, freq_bins) - Ready for model input
        duration_sec: Audio duration in seconds
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE, mono=True)
    duration_sec = len(audio) / sr
    
    # Compute Mel-Spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS
    )
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [0, 1]
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    # Shape: (freq_bins, time_steps) -> (1, time_steps, freq_bins) -> (1, 1, time_steps, freq_bins)
    mel_spec_db = mel_spec_db.T  # (time_steps, freq_bins)
    mel_spec_db = mel_spec_db[np.newaxis, :, :]  # (1, time_steps, freq_bins)
    mel_spec_db = mel_spec_db[np.newaxis, :, :, :]  # (1, 1, time_steps, freq_bins)
    
    # Convert to tensor
    spectrogram = torch.FloatTensor(mel_spec_db)
    
    return spectrogram, duration_sec


def predict_temporal_mask(
    model: UNetTemporalSegmentation,
    spectrogram: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict temporal event mask from spectrogram.
    
    Args:
        model: Trained U-Net model
        spectrogram: (1, 1, time_steps, freq_bins)
        device: PyTorch device
        threshold: Probability threshold for event detection
    
    Returns:
        mask: (time_steps,) - Binary mask (0=Normal, 1=Event)
        probs: (time_steps,) - Event probabilities
    """
    model.eval()
    with torch.no_grad():
        spectrogram = spectrogram.to(device)
        output = model(spectrogram)  # (1, 1, time_steps, 1)
        
        # Get probabilities
        probs = torch.sigmoid(output.squeeze(1).squeeze(-1))  # (1, time_steps)
        probs = probs.cpu().numpy()[0]  # (time_steps,)
        
        # Get binary mask
        mask = (probs > threshold).astype(np.float32)
    
    return mask, probs


def mask_to_segments(
    mask: np.ndarray,
    probs: np.ndarray,
    duration_sec: float,
    min_segment_duration: float = 0.1
) -> List[Tuple[float, float]]:
    """
    Convert binary temporal mask to event segments.
    
    Args:
        mask: (time_steps,) - Binary mask
        probs: (time_steps,) - Event probabilities
        duration_sec: Audio duration in seconds
        min_segment_duration: Minimum segment duration in seconds
    
    Returns:
        segments: List of (start_sec, end_sec) tuples
    """
    num_time_steps = len(mask)
    time_step_duration = duration_sec / num_time_steps
    
    segments = []
    in_segment = False
    segment_start = None
    
    for i, is_event in enumerate(mask):
        time_sec = i * time_step_duration
        
        if is_event:
            if not in_segment:
                # Start new segment
                segment_start = time_sec
                in_segment = True
        else:
            if in_segment:
                # End segment
                segment_end = time_sec
                segment_duration = segment_end - segment_start
                
                # Only add if duration >= minimum
                if segment_duration >= min_segment_duration:
                    segments.append((segment_start, segment_end))
                
                in_segment = False
    
    # Handle segment that extends to end
    if in_segment:
        segments.append((segment_start, duration_sec))
    
    return segments


def predict_events(
    audio_path: str,
    model: UNetTemporalSegmentation,
    device: torch.device,
    threshold: float = 0.5,
    min_segment_duration: float = 0.1
) -> Dict:
    """
    Predict event segments from audio file.
    
    Args:
        audio_path: Path to audio file
        model: Trained U-Net model
        device: PyTorch device
        threshold: Probability threshold for event detection
        min_segment_duration: Minimum segment duration in seconds
    
    Returns:
        results: Dictionary with predictions and metadata
    """
    # Preprocess audio
    spectrogram, duration_sec = preprocess_audio(audio_path)
    
    # Predict temporal mask
    mask, probs = predict_temporal_mask(model, spectrogram, device, threshold)
    
    # Convert mask to segments
    segments = mask_to_segments(mask, probs, duration_sec, min_segment_duration)
    
    # Prepare results
    results = {
        'audio_path': audio_path,
        'duration_sec': float(duration_sec),
        'num_time_steps': int(len(mask)),
        'threshold': threshold,
        'num_segments': len(segments),
        'segments': [(float(s), float(e)) for s, e in segments],
        'temporal_mask': mask.tolist(),
        'temporal_probs': probs.tolist(),
        'mean_event_prob': float(np.mean(probs)),
        'max_event_prob': float(np.max(probs)),
        'min_event_prob': float(np.min(probs))
    }
    
    return results


def main():
    """Main inference function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='U-Net Temporal Event Segmentation Inference')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint (default: best.pth)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for event detection')
    parser.add_argument('--output', type=str, default=None, help='Path to save results JSON')
    parser.add_argument('--min-duration', type=float, default=0.1, help='Minimum segment duration in seconds')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else config.BEST_MODEL_PATH
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    model = load_model(checkpoint_path, device)
    
    # Predict events
    print(f"\nProcessing: {args.audio}")
    results = predict_events(
        args.audio,
        model,
        device,
        threshold=args.threshold,
        min_segment_duration=args.min_duration
    )
    
    # Print results
    print(f"\nResults:")
    print(f"  Duration: {results['duration_sec']:.2f}s")
    print(f"  Segments found: {results['num_segments']}")
    print(f"  Mean event probability: {results['mean_event_prob']:.4f}")
    
    if results['num_segments'] > 0:
        print(f"\n  Event segments:")
        for i, (start, end) in enumerate(results['segments'], 1):
            print(f"    {i}. {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
