"""
Metrics for sequence-level event detection evaluation.

Includes window-level metrics (AUROC, AUPRC, F1) and optional
segment-level metrics (event-based F1).
"""

import torch
import numpy as np
from typing import Tuple, Optional, List
from sklearn.metrics import roc_auc_score, average_precision_score
import sys
from pathlib import Path

# Add EventDetect to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
from utils.postprocess_segments import postprocess_predictions


def compute_window_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    threshold: float = 0.5
) -> dict:
    """
    Compute window-level metrics for sequence predictions.
    
    Args:
        predictions: (batch, T) - Event probabilities
        targets: (batch, T) - Binary ground truth (0 or 1)
        attention_mask: (batch, T) - Attention mask (1 for valid, 0 for pad)
        threshold: Probability threshold for binary classification
    
    Returns:
        metrics: Dictionary with AUROC, AUPRC, F1, precision, recall, accuracy
    """
    # Flatten and apply attention mask if provided
    if attention_mask is not None:
        # Only consider valid positions
        valid_mask = attention_mask.bool()
        pred_flat = predictions[valid_mask].cpu().numpy()
        target_flat = targets[valid_mask].cpu().numpy()
    else:
        pred_flat = predictions.view(-1).cpu().numpy()
        target_flat = targets.view(-1).cpu().numpy()
    
    # Binary predictions
    pred_binary = (pred_flat >= threshold).astype(int)
    
    # Calculate TP, FP, FN, TN
    tp = np.sum((pred_binary == 1) & (target_flat == 1))
    fp = np.sum((pred_binary == 1) & (target_flat == 0))
    fn = np.sum((pred_binary == 0) & (target_flat == 1))
    tn = np.sum((pred_binary == 0) & (target_flat == 0))
    
    # Precision, Recall, F1
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Accuracy
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    
    # AUROC and AUPRC
    try:
        if len(np.unique(target_flat)) > 1:  # Need both classes
            auroc = roc_auc_score(target_flat, pred_flat)
            auprc = average_precision_score(target_flat, pred_flat)
        else:
            # Only one class present
            auroc = 0.0
            auprc = 0.0
    except Exception as e:
        print(f"Warning: Could not compute AUROC/AUPRC: {e}")
        auroc = 0.0
        auprc = 0.0
    
    return {
        'auroc': float(auroc),
        'auprc': float(auprc),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'accuracy': float(accuracy),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }


def compute_segment_f1(
    window_probs: np.ndarray,
    gt_segments: List[Tuple[float, float]],
    window_sec: float,
    hop_sec: float,
    threshold: float = 0.5,
    min_duration_sec: float = 0.1,
    iou_threshold: float = 0.3
) -> dict:
    """
    Compute segment-level F1 score by converting window predictions to segments.
    
    Args:
        window_probs: (T_windows,) - Window probabilities
        gt_segments: List of (start_sec, end_sec) ground truth segments
        window_sec: Window duration
        hop_sec: Hop size
        threshold: Probability threshold
        min_duration_sec: Minimum segment duration
        iou_threshold: IoU threshold for matching segments
    
    Returns:
        metrics: Dictionary with segment-level metrics
    """
    # Convert window predictions to segments
    pred_segments = postprocess_predictions(
        window_probs,
        window_sec,
        hop_sec,
        threshold,
        min_duration_sec
    )
    
    # Compute segment-based F1
    tp, fp, fn = _match_segments(pred_segments, gt_segments, iou_threshold)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'segment_f1': float(f1),
        'segment_precision': float(precision),
        'segment_recall': float(recall),
        'segment_tp': int(tp),
        'segment_fp': int(fp),
        'segment_fn': int(fn),
        'num_pred_segments': len(pred_segments),
        'num_gt_segments': len(gt_segments)
    }


def _match_segments(
    pred_segments: List[Tuple[float, float]],
    gt_segments: List[Tuple[float, float]],
    iou_threshold: float = 0.3
) -> Tuple[int, int, int]:
    """
    Match predicted segments with ground truth segments.
    
    Args:
        pred_segments: List of (start, end) predicted segments
        gt_segments: List of (start, end) ground truth segments
        iou_threshold: IoU threshold for matching
    
    Returns:
        tp, fp, fn: True positives, false positives, false negatives
    """
    if len(gt_segments) == 0:
        # No ground truth segments
        return 0, len(pred_segments), 0
    
    if len(pred_segments) == 0:
        # No predicted segments
        return 0, 0, len(gt_segments)
    
    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_segments), len(gt_segments)))
    for i, pred_seg in enumerate(pred_segments):
        for j, gt_seg in enumerate(gt_segments):
            iou_matrix[i, j] = _compute_segment_iou(pred_seg, gt_seg)
    
    # Match segments (greedy matching)
    matched_gt = set()
    matched_pred = set()
    tp = 0
    
    # Sort by IoU (highest first)
    matches = []
    for i in range(len(pred_segments)):
        for j in range(len(gt_segments)):
            if iou_matrix[i, j] >= iou_threshold:
                matches.append((iou_matrix[i, j], i, j))
    
    matches.sort(reverse=True)
    
    for iou, i, j in matches:
        if i not in matched_pred and j not in matched_gt:
            matched_pred.add(i)
            matched_gt.add(j)
            tp += 1
    
    fp = len(pred_segments) - len(matched_pred)
    fn = len(gt_segments) - len(matched_gt)
    
    return tp, fp, fn


def _compute_segment_iou(
    seg1: Tuple[float, float],
    seg2: Tuple[float, float]
) -> float:
    """
    Compute IoU between two segments.
    
    Args:
        seg1: (start, end) tuple
        seg2: (start, end) tuple
    
    Returns:
        iou: Intersection over Union
    """
    start1, end1 = seg1
    start2, end2 = seg2
    
    # Intersection
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start)
    
    # Union
    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union = union_end - union_start
    
    if union == 0:
        return 0.0
    
    return intersection / union
