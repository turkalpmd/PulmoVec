"""
Offline embedding extraction script for HeAR embeddings.

Pre-computes embeddings for all audio files in the dataset to speed up
training and experimentation.
"""

import sys
from pathlib import Path
import torch
import pandas as pd
import argparse
from tqdm import tqdm

# Add EventDetect to path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import config_hear_bilstm as config
from models.hear_encoder import HeAREncoder, EmbeddingCache
from data_processing.hear_embedding_dataset import HeAREmbeddingDataset


def main():
    """Main extraction function."""
    parser = argparse.ArgumentParser(description='Extract HeAR embeddings for all audio files')
    parser.add_argument('--recompute', action='store_true',
                        help='Recompute embeddings even if cached')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Number of files to process (for progress tracking)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("HeAR Embedding Extraction")
    print("=" * 80)
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create HeAR encoder
    print("\nLoading HeAR encoder...")
    hear_encoder = HeAREncoder(
        model_name=config.HEAR_MODEL_NAME,
        frozen=True,  # Extraction mode: encoder is frozen
        device=device
    )
    
    # Create embedding cache
    embedding_cache = EmbeddingCache(config.EMBEDDING_CACHE_DIR)
    
    # Load dataset to get all files
    print("\nLoading dataset...")
    dataset = HeAREmbeddingDataset(
        csv_path=str(config.CSV_PATH),
        hear_encoder=hear_encoder,
        embedding_cache=embedding_cache,
        window_sec=config.WINDOW_SEC,
        hop_sec=config.HOP_SEC,
        overlap_ratio_threshold=config.OVERLAP_RATIO_THRESHOLD,
        sample_rate=config.SAMPLE_RATE,
        recompute_embeddings=args.recompute
    )
    
    print(f"Total files: {len(dataset)}")
    
    # Extract embeddings for all files
    print("\nExtracting embeddings...")
    print("(Embeddings will be cached automatically)")
    
    cached_count = 0
    extracted_count = 0
    
    for i in tqdm(range(len(dataset)), desc="Processing files"):
        try:
            # Get item (this will extract or load from cache)
            item = dataset[i]
            
            # Check if it was cached or extracted
            wav_path = item['file_path']
            encoder_version = getattr(hear_encoder, 'model_version', 'unknown')
            
            cached_emb = embedding_cache.get(
                wav_path,
                encoder_version,
                config.WINDOW_SEC,
                config.HOP_SEC,
                config.SAMPLE_RATE
            )
            
            if cached_emb is not None and not args.recompute:
                cached_count += 1
            else:
                extracted_count += 1
                
        except Exception as e:
            print(f"\nError processing file {i}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("Extraction completed!")
    print(f"  Cached (skipped): {cached_count}")
    print(f"  Extracted: {extracted_count}")
    print(f"  Total: {len(dataset)}")
    print(f"  Cache directory: {config.EMBEDDING_CACHE_DIR}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
