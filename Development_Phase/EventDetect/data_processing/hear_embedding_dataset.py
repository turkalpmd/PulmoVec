"""
Dataset for HeAR + BiLSTM temporal segmentation training.

Extracts HeAR embeddings from audio files and creates window-level
binary labels from JSON annotations.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from typing import Optional, List, Tuple
import json
import sys

# Add EventDetect to path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from models.hear_encoder import HeAREncoder, EmbeddingCache


class HeAREmbeddingDataset(Dataset):
    """
    Dataset for HeAR embedding-based temporal event segmentation.
    
    Input: Audio file → HeAR embeddings (sliding window)
    Output: Window-level binary labels (event var/yok)
    
    Ground truth: Generated from JSON event annotations with overlap ratio
    """
    
    def __init__(
        self,
        csv_path: str,
        hear_encoder: HeAREncoder,
        embedding_cache: EmbeddingCache,
        window_sec: float = 2.0,
        hop_sec: float = 0.25,
        overlap_ratio_threshold: float = 0.2,
        sample_rate: int = 16000,
        recompute_embeddings: bool = False
    ):
        """
        Initialize HeAR embedding dataset.
        
        Args:
            csv_path: Path to CSV with event-level annotations
            hear_encoder: HeAR encoder instance
            embedding_cache: Embedding cache instance
            window_sec: Window duration (must be 2.0 for HeAR)
            hop_sec: Hop size in seconds
            overlap_ratio_threshold: Overlap ratio threshold for positive label
            sample_rate: Audio sample rate
            recompute_embeddings: If True, recompute even if cached
        """
        self.df = pd.read_csv(csv_path)
        self.hear_encoder = hear_encoder
        self.embedding_cache = embedding_cache
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.overlap_ratio_threshold = overlap_ratio_threshold
        self.sample_rate = sample_rate
        self.recompute_embeddings = recompute_embeddings
        
        # Get encoder version for cache key
        self.encoder_version = getattr(hear_encoder, 'model_version', 'unknown')
        
        # Group by audio file (wav_path) to get unique files
        self.file_groups = self.df.groupby('wav_path').groups
        
        # Filter files that exist
        self.valid_files = []
        for wav_path in self.file_groups.keys():
            if Path(wav_path).exists():
                self.valid_files.append(wav_path)
            else:
                print(f"Warning: File not found: {wav_path}")
        
        print(f"Loaded {len(self.valid_files)} unique audio files")
        print(f"Total events: {len(self.df)}")
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            embeddings: (T_windows, 512) - HeAR embeddings
            labels: (T_windows,) - Binary labels (0=Normal, 1=Event)
            length: int - Number of windows
            file_path: str - Path to audio file
            duration_sec: float - Audio duration
        """
        wav_path = self.valid_files[idx]
        
        # Load audio
        audio, sr = librosa.load(wav_path, sr=self.sample_rate, mono=True)
        duration_sec = len(audio) / sr
        
        # Extract or load cached embeddings
        embeddings = self._get_embeddings(wav_path, audio)
        
        # Get ground truth window-level labels from JSON
        json_path = self._get_json_path(wav_path)
        labels = self._create_window_labels(json_path, duration_sec, len(embeddings))
        
        # Convert to tensors
        embeddings_tensor = torch.FloatTensor(embeddings)  # (T_windows, 512)
        labels_tensor = torch.FloatTensor(labels)  # (T_windows,)
        
        return {
            'embeddings': embeddings_tensor,
            'labels': labels_tensor,
            'length': len(embeddings),
            'file_path': wav_path,
            'duration_sec': duration_sec
        }
    
    def _get_embeddings(self, wav_path: str, audio: np.ndarray) -> np.ndarray:
        """Get embeddings from cache or extract them."""
        # Try to load from cache
        if not self.recompute_embeddings:
            cached_embeddings = self.embedding_cache.get(
                wav_path,
                self.encoder_version,
                self.window_sec,
                self.hop_sec,
                self.sample_rate
            )
            if cached_embeddings is not None:
                return cached_embeddings
        
        # Extract embeddings
        print(f"Extracting embeddings for: {Path(wav_path).name}")
        embeddings = self.hear_encoder.extract_embeddings_sliding_window(
            audio,
            sample_rate=self.sample_rate,
            window_sec=self.window_sec,
            hop_sec=self.hop_sec
        )
        
        # Save to cache
        self.embedding_cache.save(
            embeddings,
            wav_path,
            self.encoder_version,
            self.window_sec,
            self.hop_sec,
            self.sample_rate
        )
        
        return embeddings
    
    def _get_json_path(self, wav_path: str) -> Optional[str]:
        """Get corresponding JSON annotation file path."""
        wav_path = Path(wav_path)
        
        # Try different JSON path patterns
        # Pattern 1: Replace .wav with .json in same directory
        json_path = wav_path.with_suffix('.json')
        if json_path.exists():
            return str(json_path)
        
        # Pattern 2: Look in corresponding _json directory
        json_dir = wav_path.parent.parent / (wav_path.parent.name.replace('_wav', '_json'))
        json_path = json_dir / wav_path.name.replace('.wav', '.json')
        if json_path.exists():
            return str(json_path)
        
        # Pattern 3: Look in same directory with different naming
        json_path = wav_path.parent / wav_path.name.replace('.wav', '.json')
        if json_path.exists():
            return str(json_path)
        
        return None
    
    def _create_window_labels(
        self,
        json_path: Optional[str],
        duration_sec: float,
        num_windows: int
    ) -> np.ndarray:
        """
        Create window-level binary labels from JSON annotations.
        
        For each window, check if it overlaps with any event segment
        by at least overlap_ratio_threshold. If yes, label=1, else 0.
        
        Args:
            json_path: Path to JSON annotation file
            duration_sec: Audio duration in seconds
            num_windows: Number of windows
        
        Returns:
            labels: (num_windows,) - Binary labels: 0=Normal, 1=Event
        """
        # Initialize labels: all Normal (0)
        labels = np.zeros(num_windows, dtype=np.float32)
        
        if json_path is None or not Path(json_path).exists():
            return labels
        
        # Load JSON annotations
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON {json_path}: {e}")
            return labels
        
        # Get event segments (non-Normal events)
        event_segments = []
        events = data.get('event_annotation', [])
        
        for event in events:
            start_ms = float(event.get('start', 0))
            end_ms = float(event.get('end', 0))
            event_type = event.get('type', 'Normal')
            
            # Skip Normal events
            if event_type == "Normal":
                continue
            
            # Convert ms to seconds
            start_sec = start_ms / 1000.0
            end_sec = end_ms / 1000.0
            
            event_segments.append((start_sec, end_sec))
        
        if len(event_segments) == 0:
            return labels
        
        # For each window, check if it contains significant event content
        for window_idx in range(num_windows):
            window_start_sec = window_idx * self.hop_sec
            window_end_sec = window_start_sec + self.window_sec
            window_center_sec = (window_start_sec + window_end_sec) / 2.0
            
            # Strategy 1: Check if window center is inside any event
            center_in_event = False
            for event_start, event_end in event_segments:
                if event_start <= window_center_sec <= event_end:
                    center_in_event = True
                    break
            
            if center_in_event:
                labels[window_idx] = 1.0
                continue
            
            # Strategy 2: Check overlap ratio (more strict)
            # Only label as positive if significant portion of window is event
            max_overlap_ratio = 0.0
            total_event_duration_in_window = 0.0
            
            for event_start, event_end in event_segments:
                # Calculate intersection
                intersection_start = max(window_start_sec, event_start)
                intersection_end = min(window_end_sec, event_end)
                intersection = max(0, intersection_end - intersection_start)
                
                if intersection > 0:
                    total_event_duration_in_window += intersection
                    overlap_ratio = intersection / self.window_sec
                    max_overlap_ratio = max(max_overlap_ratio, overlap_ratio)
            
            # Label as positive if:
            # 1. Overlap ratio is high enough (strict threshold)
            # 2. OR total event duration in window is significant (e.g., > 0.1s)
            if max_overlap_ratio >= self.overlap_ratio_threshold or total_event_duration_in_window >= 0.1:
                labels[window_idx] = 1.0
        
        return labels
    
    def _compute_overlap_ratio(
        self,
        window: Tuple[float, float],
        event: Tuple[float, float]
    ) -> float:
        """
        Compute overlap ratio between window and event segment.
        
        Overlap ratio = intersection / window_duration
        
        Args:
            window: (start_sec, end_sec) window boundaries
            event: (start_sec, end_sec) event boundaries
        
        Returns:
            overlap_ratio: Ratio of overlap (0.0 to 1.0)
        """
        window_start, window_end = window
        event_start, event_end = event
        
        # Intersection
        intersection_start = max(window_start, event_start)
        intersection_end = min(window_end, event_end)
        intersection = max(0, intersection_end - intersection_start)
        
        # Window duration
        window_duration = window_end - window_start
        
        if window_duration == 0:
            return 0.0
        
        return intersection / window_duration
