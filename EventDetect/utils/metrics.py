"""
Metrics for temporal segmentation evaluation.
"""

import torch
import numpy as np
from typing import Tuple


def compute_temporal_iou(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute temporal IoU (Intersection over Union) for binary segmentation.
    
    Args:
        predictions: (batch, time_steps) - Event probabilities
        targets: (batch, time_steps) - Binary ground truth (0 or 1)
        threshold: Probability threshold for binary prediction
    
    Returns:
        iou: Average IoU across batch
    """
    # Convert probabilities to binary predictions
    pred_binary = (predictions > threshold).float()
    targets = targets.float()
    
    # Flatten
    pred_flat = pred_binary.view(-1)
    target_flat = targets.view(-1)
    
    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    # Avoid division by zero
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    return iou.item()


def compute_temporal_precision_recall_f1(
    predictions: torch.Tensor, 
    targets: torch.Tensor, 
    threshold: float = 0.5
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score for temporal segmentation.
    
    Args:
        predictions: (batch, time_steps) - Event probabilities
        targets: (batch, time_steps) - Binary ground truth (0 or 1)
        threshold: Probability threshold for binary prediction
    
    Returns:
        precision: Precision score
        recall: Recall score
        f1: F1 score
    """
    # Convert probabilities to binary predictions
    pred_binary = (predictions > threshold).float()
    targets = targets.float()
    
    # Flatten
    pred_flat = pred_binary.view(-1)
    target_flat = targets.view(-1)
    
    # Calculate TP, FP, FN
    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    fn = ((1 - pred_flat) * target_flat).sum()
    
    # Calculate metrics
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return precision.item(), recall.item(), f1.item()


def compute_temporal_accuracy(
    predictions: torch.Tensor, 
    targets: torch.Tensor, 
    threshold: float = 0.5
) -> float:
    """
    Compute accuracy for temporal segmentation.
    
    Args:
        predictions: (batch, time_steps) - Event probabilities
        targets: (batch, time_steps) - Binary ground truth (0 or 1)
        threshold: Probability threshold for binary prediction
    
    Returns:
        accuracy: Accuracy score
    """
    # Convert probabilities to binary predictions
    pred_binary = (predictions > threshold).float()
    targets = targets.float()
    
    # Flatten
    pred_flat = pred_binary.view(-1)
    target_flat = targets.view(-1)
    
    # Calculate accuracy
    correct = (pred_flat == target_flat).float().sum()
    total = pred_flat.numel()
    
    accuracy = correct / total
    return accuracy.item()
