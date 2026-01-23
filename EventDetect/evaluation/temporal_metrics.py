"""Temporal event segmentation evaluation metrics."""

import numpy as np
from typing import List, Tuple, Dict


def compute_iou(gt_segment: Tuple[float, float], pred_segment: Tuple[float, float]) -> float:
    """
    Compute Intersection over Union (IoU) for two segments.
    
    Args:
        gt_segment: (start, end) tuple for ground truth
        pred_segment: (start, end) tuple for prediction
        
    Returns:
        IoU value between 0 and 1
    """
    gt_start, gt_end = gt_segment
    pred_start, pred_end = pred_segment
    
    # Intersection
    intersection_start = max(gt_start, pred_start)
    intersection_end = min(gt_end, pred_end)
    intersection = max(0, intersection_end - intersection_start)
    
    # Union
    union_start = min(gt_start, pred_start)
    union_end = max(gt_end, pred_end)
    union = union_end - union_start
    
    if union == 0:
        return 0.0
    
    return intersection / union


def match_segments_by_iou(ground_truth: List[Tuple[float, float]], 
                          predictions: List[Tuple[float, float]], 
                          iou_threshold: float = 0.5) -> Dict[int, int]:
    """
    Perform one-to-one matching between predicted and ground-truth segments based on maximum IoU.
    
    Args:
        ground_truth: List of (start, end) tuples
        predictions: List of (start, end) tuples
        iou_threshold: Minimum IoU for a valid match
        
    Returns:
        Dictionary mapping prediction index to ground truth index
    """
    matches = {}
    used_gt = set()
    
    # Compute IoU matrix
    iou_matrix = np.zeros((len(predictions), len(ground_truth)))
    for i, pred in enumerate(predictions):
        for j, gt in enumerate(ground_truth):
            iou_matrix[i, j] = compute_iou(gt, pred)
    
    # Greedy matching: match highest IoU pairs first
    while True:
        max_iou = -1
        best_pred_idx = None
        best_gt_idx = None
        
        for i in range(len(predictions)):
            if i in matches:
                continue
            for j in range(len(ground_truth)):
                if j in used_gt:
                    continue
                if iou_matrix[i, j] > max_iou:
                    max_iou = iou_matrix[i, j]
                    best_pred_idx = i
                    best_gt_idx = j
        
        if max_iou < iou_threshold or best_pred_idx is None:
            break
        
        matches[best_pred_idx] = best_gt_idx
        used_gt.add(best_gt_idx)
    
    return matches


def compute_mean_iou(ground_truth: List[Tuple[float, float]], 
                    predictions: List[Tuple[float, float]]) -> Dict[str, float]:
    """
    Compute mean and median IoU for matched segments.
    
    Args:
        ground_truth: List of (start, end) tuples
        predictions: List of (start, end) tuples
        
    Returns:
        Dictionary with 'mean' and 'median' IoU values
    """
    if len(predictions) == 0 or len(ground_truth) == 0:
        return {'mean': 0.0, 'median': 0.0}
    
    matches = match_segments_by_iou(ground_truth, predictions, iou_threshold=0.0)
    
    if len(matches) == 0:
        return {'mean': 0.0, 'median': 0.0}
    
    ious = []
    for pred_idx, gt_idx in matches.items():
        iou = compute_iou(ground_truth[gt_idx], predictions[pred_idx])
        ious.append(iou)
    
    return {
        'mean': np.mean(ious) if len(ious) > 0 else 0.0,
        'median': np.median(ious) if len(ious) > 0 else 0.0
    }


def compute_recall_at_iou(ground_truth: List[Tuple[float, float]], 
                          predictions: List[Tuple[float, float]], 
                          threshold: float) -> float:
    """
    Compute recall for IoU >= threshold.
    
    Args:
        ground_truth: List of (start, end) tuples
        predictions: List of (start, end) tuples
        threshold: IoU threshold
        
    Returns:
        Recall value (0-1)
    """
    if len(ground_truth) == 0:
        return 1.0 if len(predictions) == 0 else 0.0
    
    matches = match_segments_by_iou(ground_truth, predictions, iou_threshold=threshold)
    return len(matches) / len(ground_truth)


def compute_onset_error(gt_segment: Tuple[float, float], 
                       pred_segment: Tuple[float, float]) -> float:
    """
    Compute onset (start time) error in seconds.
    
    Args:
        gt_segment: (start, end) tuple for ground truth
        pred_segment: (start, end) tuple for prediction
        
    Returns:
        Absolute error in seconds
    """
    return abs(gt_segment[0] - pred_segment[0])


def compute_offset_error(gt_segment: Tuple[float, float], 
                        pred_segment: Tuple[float, float]) -> float:
    """
    Compute offset (end time) error in seconds.
    
    Args:
        gt_segment: (start, end) tuple for ground truth
        pred_segment: (start, end) tuple for prediction
        
    Returns:
        Absolute error in seconds
    """
    return abs(gt_segment[1] - pred_segment[1])


def compute_boundary_f1(ground_truth: List[Tuple[float, float]], 
                       predictions: List[Tuple[float, float]], 
                       tolerance: float) -> Dict[str, float]:
    """
    Compute F1 score where a boundary is considered correct if within ±tolerance seconds.
    
    Args:
        ground_truth: List of (start, end) tuples
        predictions: List of (start, end) tuples
        tolerance: Tolerance in seconds
        
    Returns:
        Dictionary with 'precision', 'recall', 'f1' for boundaries
    """
    # Extract all boundaries
    gt_boundaries = []
    for seg in ground_truth:
        gt_boundaries.append(('start', seg[0]))
        gt_boundaries.append(('end', seg[1]))
    
    pred_boundaries = []
    for seg in predictions:
        pred_boundaries.append(('start', seg[0]))
        pred_boundaries.append(('end', seg[1]))
    
    # Match boundaries
    matched_pred = set()
    matched_gt = set()
    
    for i, (gt_type, gt_time) in enumerate(gt_boundaries):
        for j, (pred_type, pred_time) in enumerate(pred_boundaries):
            if j in matched_pred:
                continue
            if gt_type == pred_type and abs(gt_time - pred_time) <= tolerance:
                matched_pred.add(j)
                matched_gt.add(i)
                break
    
    tp = len(matched_gt)
    fp = len(pred_boundaries) - len(matched_pred)
    fn = len(gt_boundaries) - len(matched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}


def compute_duration_error(gt_segment: Tuple[float, float], 
                          pred_segment: Tuple[float, float]) -> float:
    """
    Compute duration error in seconds.
    
    Args:
        gt_segment: (start, end) tuple for ground truth
        pred_segment: (start, end) tuple for prediction
        
    Returns:
        Absolute duration difference in seconds
    """
    gt_duration = gt_segment[1] - gt_segment[0]
    pred_duration = pred_segment[1] - pred_segment[0]
    return abs(gt_duration - pred_duration)


def compute_coverage(gt_segment: Tuple[float, float], 
                    pred_segment: Tuple[float, float]) -> float:
    """
    Compute coverage: intersection / ground truth length.
    
    Args:
        gt_segment: (start, end) tuple for ground truth
        pred_segment: (start, end) tuple for prediction
        
    Returns:
        Coverage value (0-1)
    """
    gt_start, gt_end = gt_segment
    pred_start, pred_end = pred_segment
    
    intersection_start = max(gt_start, pred_start)
    intersection_end = min(gt_end, pred_end)
    intersection = max(0, intersection_end - intersection_start)
    
    gt_length = gt_end - gt_start
    if gt_length == 0:
        return 0.0
    
    return intersection / gt_length


def compute_purity(gt_segment: Tuple[float, float], 
                  pred_segment: Tuple[float, float]) -> float:
    """
    Compute purity: intersection / prediction length.
    
    Args:
        gt_segment: (start, end) tuple for ground truth
        pred_segment: (start, end) tuple for prediction
        
    Returns:
        Purity value (0-1)
    """
    gt_start, gt_end = gt_segment
    pred_start, pred_end = pred_segment
    
    intersection_start = max(gt_start, pred_start)
    intersection_end = min(gt_end, pred_end)
    intersection = max(0, intersection_end - intersection_start)
    
    pred_length = pred_end - pred_start
    if pred_length == 0:
        return 0.0
    
    return intersection / pred_length


def compute_segment_precision_recall_f1(ground_truth: List[Tuple[float, float]], 
                                       predictions: List[Tuple[float, float]], 
                                       iou_threshold: float) -> Dict[str, float]:
    """
    Compute segment-level precision, recall, and F1.
    
    Args:
        ground_truth: List of (start, end) tuples
        predictions: List of (start, end) tuples
        iou_threshold: Minimum IoU for a true positive
        
    Returns:
        Dictionary with 'precision', 'recall', 'f1'
    """
    matches = match_segments_by_iou(ground_truth, predictions, iou_threshold=iou_threshold)
    
    tp = len(matches)
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}


def compute_false_positives_per_hour(ground_truth: List[Tuple[float, float]], 
                                     predictions: List[Tuple[float, float]], 
                                     recording_duration_seconds: float) -> float:
    """
    Compute false positives per hour.
    
    Args:
        ground_truth: List of (start, end) tuples
        predictions: List of (start, end) tuples
        recording_duration_seconds: Total recording duration
        
    Returns:
        False positives per hour
    """
    if recording_duration_seconds == 0:
        return 0.0
    
    matches = match_segments_by_iou(ground_truth, predictions, iou_threshold=0.5)
    fp = len(predictions) - len(matches)
    
    hours = recording_duration_seconds / 3600.0
    return fp / hours if hours > 0 else 0.0


def evaluate_segmentation(ground_truth: List[Tuple[float, float]], 
                         predictions: List[Tuple[float, float]], 
                         tolerances: List[float] = [0.05, 0.1, 0.25], 
                         iou_thresholds: List[float] = [0.3, 0.5, 0.7],
                         recording_duration_seconds: float = None) -> Dict:
    """
    Complete evaluation of temporal segmentation.
    
    Args:
        ground_truth: List of (start, end) tuples
        predictions: List of (start, end) tuples
        tolerances: List of boundary tolerance values in seconds
        iou_thresholds: List of IoU thresholds for evaluation
        recording_duration_seconds: Total recording duration (for FP/hour)
        
    Returns:
        Dictionary with all metrics
    """
    results = {}
    
    # Mean IoU
    mean_iou = compute_mean_iou(ground_truth, predictions)
    results['mean_iou'] = mean_iou['mean']
    results['median_iou'] = mean_iou['median']
    
    # Recall at IoU thresholds
    results['recall_at_iou'] = {}
    for threshold in iou_thresholds:
        results['recall_at_iou'][f'{threshold:.1f}'] = compute_recall_at_iou(
            ground_truth, predictions, threshold
        )
    
    # Precision/Recall/F1 at IoU thresholds
    results['precision_recall_f1'] = {}
    for threshold in iou_thresholds:
        prf1 = compute_segment_precision_recall_f1(ground_truth, predictions, threshold)
        results['precision_recall_f1'][f'{threshold:.1f}'] = prf1
    
    # Boundary errors (for matched segments)
    matches = match_segments_by_iou(ground_truth, predictions, iou_threshold=0.0)
    onset_errors = []
    offset_errors = []
    duration_errors = []
    
    for pred_idx, gt_idx in matches.items():
        onset_err = compute_onset_error(ground_truth[gt_idx], predictions[pred_idx])
        offset_err = compute_offset_error(ground_truth[gt_idx], predictions[pred_idx])
        duration_err = compute_duration_error(ground_truth[gt_idx], predictions[pred_idx])
        onset_errors.append(onset_err)
        offset_errors.append(offset_err)
        duration_errors.append(duration_err)
    
    results['median_onset_error'] = np.median(onset_errors) if len(onset_errors) > 0 else 0.0
    results['median_offset_error'] = np.median(offset_errors) if len(offset_errors) > 0 else 0.0
    results['median_duration_error'] = np.median(duration_errors) if len(duration_errors) > 0 else 0.0
    
    # Boundary F1 at different tolerances
    results['boundary_f1'] = {}
    for tolerance in tolerances:
        bf1 = compute_boundary_f1(ground_truth, predictions, tolerance)
        results['boundary_f1'][f'{tolerance:.2f}'] = bf1
    
    # False positives per hour
    if recording_duration_seconds is not None:
        results['false_positives_per_hour'] = compute_false_positives_per_hour(
            ground_truth, predictions, recording_duration_seconds
        )
    else:
        results['false_positives_per_hour'] = None
    
    # Additional stats
    results['num_ground_truth'] = len(ground_truth)
    results['num_predictions'] = len(predictions)
    results['num_matched'] = len(matches)
    
    return results
