"""
Post-processing utilities for converting window predictions to event segments.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.ndimage import median_filter


def postprocess_predictions(
    window_probs: np.ndarray,
    window_sec: float,
    hop_sec: float,
    threshold: float = 0.5,
    min_duration_sec: float = 0.1,
    smoothing_enabled: bool = False,
    smoothing_window_size: int = 3,
    use_hysteresis: bool = False,
    on_threshold: float = 0.6,
    off_threshold: float = 0.4,
    max_gap_sec: float = 0.5
) -> List[Tuple[float, float]]:
    """
    Convert window probabilities to event segments.
    
    Args:
        window_probs: (T_windows,) array of event probabilities
        window_sec: Window duration in seconds
        hop_sec: Hop size in seconds
        threshold: Probability threshold for binary classification
        min_duration_sec: Minimum segment duration after merging
        smoothing_enabled: If True, apply median filter smoothing
        smoothing_window_size: Median filter window size
        use_hysteresis: If True, use hysteresis thresholding instead of simple threshold
        on_threshold: Threshold to turn on event (for hysteresis)
        off_threshold: Threshold to turn off event (for hysteresis)
        max_gap_sec: Maximum gap between windows to merge (in seconds)
    
    Returns:
        segments: List of (start_sec, end_sec) tuples
    """
    # Apply smoothing if enabled
    if smoothing_enabled:
        window_probs = median_filter(window_probs, size=smoothing_window_size)
    
    # Binary classification
    if use_hysteresis:
        binary_preds = apply_hysteresis(window_probs, on_threshold, off_threshold)
    else:
        binary_preds = (window_probs >= threshold).astype(int)
    
    # Merge consecutive positive windows with max gap
    segments = _merge_consecutive_windows(
        binary_preds,
        window_sec,
        hop_sec,
        max_gap_sec=max_gap_sec
    )
    
    # Filter by minimum duration
    segments = _filter_min_duration(segments, min_duration_sec)
    
    return segments


def _merge_consecutive_windows(
    binary_preds: np.ndarray,
    window_sec: float,
    hop_sec: float,
    max_gap_sec: float = 0.5
) -> List[Tuple[float, float]]:
    """
    Merge consecutive positive windows into segments.
    Only merges windows if gap between them is less than max_gap_sec.
    Uses window center for more precise segment boundaries.
    
    Args:
        binary_preds: (T_windows,) binary predictions
        window_sec: Window duration
        hop_sec: Hop size
        max_gap_sec: Maximum gap between windows to merge (in seconds)
    
    Returns:
        segments: List of (start_sec, end_sec) tuples
    """
    segments = []
    
    if len(binary_preds) == 0:
        return segments
    
    # Find all positive window indices
    positive_indices = np.where(binary_preds == 1)[0]
    
    if len(positive_indices) == 0:
        return segments
    
    # Group consecutive or nearby positive windows
    current_group = [positive_indices[0]]
    max_gap_windows = int(max_gap_sec / hop_sec)  # Convert gap to number of windows
    
    for i in range(1, len(positive_indices)):
        gap = positive_indices[i] - positive_indices[i-1]
        
        if gap <= max_gap_windows + 1:  # +1 because consecutive means gap=1
            # Continue current segment
            current_group.append(positive_indices[i])
        else:
            # Start new segment
            # Save current segment - use window centers for boundaries
            start_idx = current_group[0]
            end_idx = current_group[-1]
            # Start at beginning of first window, end at end of last window
            start_sec = start_idx * hop_sec
            end_sec = end_idx * hop_sec + window_sec
            segments.append((start_sec, end_sec))
            
            # Start new group
            current_group = [positive_indices[i]]
    
    # Don't forget the last group
    if current_group:
        start_idx = current_group[0]
        end_idx = current_group[-1]
        start_sec = start_idx * hop_sec
        end_sec = end_idx * hop_sec + window_sec
        segments.append((start_sec, end_sec))
    
    return segments


def _filter_min_duration(
    segments: List[Tuple[float, float]],
    min_duration_sec: float
) -> List[Tuple[float, float]]:
    """
    Filter segments by minimum duration.
    
    Args:
        segments: List of (start_sec, end_sec) tuples
        min_duration_sec: Minimum duration threshold
    
    Returns:
        filtered_segments: Filtered list of segments
    """
    filtered = []
    for start, end in segments:
        duration = end - start
        if duration >= min_duration_sec:
            filtered.append((start, end))
    
    return filtered


def apply_hysteresis(
    window_probs: np.ndarray,
    on_threshold: float = 0.6,
    off_threshold: float = 0.4
) -> np.ndarray:
    """
    Apply hysteresis thresholding to reduce flickering.
    
    Args:
        window_probs: (T_windows,) array of probabilities
        on_threshold: Threshold to turn on (higher)
        off_threshold: Threshold to turn off (lower)
    
    Returns:
        binary_preds: (T_windows,) binary predictions
    """
    binary_preds = np.zeros_like(window_probs, dtype=int)
    state = 0  # 0 = off, 1 = on
    
    for i, prob in enumerate(window_probs):
        if state == 0:
            # Currently off
            if prob >= on_threshold:
                state = 1
                binary_preds[i] = 1
        else:
            # Currently on
            if prob < off_threshold:
                state = 0
                binary_preds[i] = 0
            else:
                binary_preds[i] = 1
    
    return binary_preds
