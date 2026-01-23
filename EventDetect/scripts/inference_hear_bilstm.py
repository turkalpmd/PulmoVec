"""
Inference script for HeAR + BiLSTM temporal event segmentation.
Predicts event segments from audio files.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import librosa
import json
from typing import Dict
import argparse

# Add EventDetect to path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import config_hear_bilstm as config
from models.hear_encoder import HeAREncoder, EmbeddingCache
from models.bilstm_event_detector import BiLSTMEventDetector
from utils.postprocess_segments import postprocess_predictions


def load_model(checkpoint_path: Path, device: torch.device) -> BiLSTMEventDetector:
    """Load trained BiLSTM model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model config from checkpoint or use defaults
    model_config = checkpoint.get('config', {})
    hidden_dim = model_config.get('hidden_dim', config.BILSTM_HIDDEN_DIM)
    num_layers = model_config.get('num_layers', config.BILSTM_NUM_LAYERS)
    dropout = model_config.get('dropout', config.BILSTM_DROPOUT)
    input_dim = model_config.get('input_dim', config.HEAR_EMBEDDING_DIM)
    
    model = BiLSTMEventDetector(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded from: {checkpoint_path}")
    return model


def extract_embeddings(
    audio_path: str,
    hear_encoder: HeAREncoder,
    embedding_cache: EmbeddingCache
) -> np.ndarray:
    """
    Extract HeAR embeddings from audio file.
    
    Args:
        audio_path: Path to audio file
        hear_encoder: HeAR encoder instance
        embedding_cache: Embedding cache instance
    
    Returns:
        embeddings: (T_windows, 512) array
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE, mono=True)
    
    # Try to load from cache
    encoder_version = getattr(hear_encoder, 'model_version', 'unknown')
    cached_embeddings = embedding_cache.get(
        audio_path,
        encoder_version,
        config.WINDOW_SEC,
        config.HOP_SEC,
        config.SAMPLE_RATE
    )
    
    if cached_embeddings is not None:
        return cached_embeddings
    
    # Extract embeddings
    embeddings = hear_encoder.extract_embeddings_sliding_window(
        audio,
        sample_rate=config.SAMPLE_RATE,
        window_sec=config.WINDOW_SEC,
        hop_sec=config.HOP_SEC
    )
    
    # Save to cache
    embedding_cache.save(
        embeddings,
        audio_path,
        encoder_version,
        config.WINDOW_SEC,
        config.HOP_SEC,
        config.SAMPLE_RATE
    )
    
    return embeddings


def predict_events(
    audio_path: str,
    model: BiLSTMEventDetector,
    hear_encoder: HeAREncoder,
    embedding_cache: EmbeddingCache,
    device: torch.device,
    threshold: float = 0.5,
    min_duration_sec: float = 0.1
) -> Dict:
    """
    Predict event segments from audio file.
    
    Args:
        audio_path: Path to audio file
        model: Trained BiLSTM model
        hear_encoder: HeAR encoder instance
        embedding_cache: Embedding cache instance
        device: PyTorch device
        threshold: Probability threshold for event detection
        min_duration_sec: Minimum segment duration in seconds
    
    Returns:
        results: Dictionary with predictions and metadata
    """
    # Load audio to get duration
    audio, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE, mono=True)
    duration_sec = len(audio) / sr
    
    # Extract embeddings
    embeddings = extract_embeddings(audio_path, hear_encoder, embedding_cache)
    num_windows = len(embeddings)
    
    # Convert to tensor
    embeddings_tensor = torch.FloatTensor(embeddings).unsqueeze(0).to(device)  # (1, T, 512)
    lengths = torch.tensor([num_windows], device=device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        logits = model(embeddings_tensor, lengths)  # (1, T)
        probs = torch.sigmoid(logits).cpu().numpy()[0]  # (T,)
    
    # Post-process to get segments
    segments = postprocess_predictions(
        probs,
        config.WINDOW_SEC,
        config.HOP_SEC,
        threshold=threshold,
        min_duration_sec=min_duration_sec,
        smoothing_enabled=config.SMOOTHING_ENABLED,
        smoothing_window_size=config.SMOOTHING_WINDOW_SIZE,
        use_hysteresis=config.USE_HYSTERESIS,
        on_threshold=config.HYSTERESIS_ON_THRESHOLD,
        off_threshold=config.HYSTERESIS_OFF_THRESHOLD,
        max_gap_sec=config.MAX_GAP_SEC
    )
    
    # Prepare results
    results = {
        'audio_path': audio_path,
        'duration_sec': float(duration_sec),
        'window_sec': config.WINDOW_SEC,
        'hop_sec': config.HOP_SEC,
        'num_windows': int(num_windows),
        'threshold': threshold,
        'num_segments': len(segments),
        'segments': [[float(s), float(e)] for s, e in segments],
        'window_probs': probs.tolist(),
        'mean_event_prob': float(np.mean(probs)),
        'max_event_prob': float(np.max(probs)),
        'min_event_prob': float(np.min(probs))
    }
    
    return results


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='HeAR + BiLSTM Temporal Event Segmentation Inference')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint (default: best.pth)')
    parser.add_argument('--threshold', type=float, default=None, help='Probability threshold (default: from config)')
    parser.add_argument('--output', type=str, default=None, help='Path to save results JSON')
    parser.add_argument('--min-duration', type=float, default=None, help='Minimum segment duration in seconds (default: from config)')
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
        frozen=True,  # Inference mode: encoder is frozen
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
    
    # Get threshold and min_duration
    threshold = args.threshold if args.threshold is not None else config.EVAL_THRESHOLD
    min_duration = args.min_duration if args.min_duration is not None else config.MIN_DURATION_SEC
    
    # Predict events
    print(f"\nProcessing: {args.audio}")
    results = predict_events(
        args.audio,
        model,
        hear_encoder,
        embedding_cache,
        device,
        threshold=threshold,
        min_duration_sec=min_duration
    )
    
    # Print results
    print(f"\nResults:")
    print(f"  Duration: {results['duration_sec']:.2f}s")
    print(f"  Windows: {results['num_windows']}")
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
    else:
        # Print JSON to stdout
        print(f"\nJSON Output:")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
