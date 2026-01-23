"""
Utility functions for training, evaluation, and visualization.
Includes metrics computation, plotting, and model checkpointing.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
import json
from pathlib import Path
from typing import Dict, List, Tuple
import config


def compute_class_weights(class_counts: Dict[str, int]) -> torch.Tensor:
    """
    Calculate inverse frequency class weights for imbalanced dataset.
    
    Args:
        class_counts: Dictionary mapping class names to counts
    
    Returns:
        weights: Tensor of class weights
    """
    counts = np.array([class_counts[name] for name in config.CLASS_NAMES])
    total = counts.sum()
    
    # Inverse frequency weighting
    weights = total / (len(counts) * counts)
    
    # Normalize
    weights = weights / weights.sum() * len(counts)
    
    return torch.from_numpy(weights).float()


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    
    Returns:
        metrics: Dictionary of metric values
    """
    if class_names is None:
        class_names = config.CLASS_NAMES
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0)
    }
    
    # Per-class F1 scores
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, class_name in enumerate(class_names):
        metrics[f'f1_{class_name}'] = f1_per_class[i]
    
    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None
):
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    if class_names is None:
        class_names = config.CLASS_NAMES
    
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    print(report)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    save_path: str = None,
    normalize: bool = True
):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save figure
        normalize: Whether to normalize the confusion matrix
    """
    if class_names is None:
        class_names = config.CLASS_NAMES
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str = None
):
    """
    Plot training curves (loss and metrics over epochs).
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc', etc.
        save_path: Path to save figure
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1-macro
    axes[1, 0].plot(epochs, history['train_f1_macro'], 'b-', label='Train F1', linewidth=2)
    axes[1, 0].plot(epochs, history['val_f1_macro'], 'r-', label='Val F1', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('F1-Macro', fontsize=12)
    axes[1, 0].set_title('Training and Validation F1-Macro', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate (if available)
    if 'lr' in history:
        axes[1, 1].plot(epochs, history['lr'], 'g-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training curves saved to {save_path}")
    
    plt.close()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: Dict[str, float],
    save_path: str,
    class_names: List[str] = None
):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        metrics: Dictionary of metrics
        save_path: Path to save checkpoint
        class_names: List of class names
    """
    if class_names is None:
        class_names = config.CLASS_NAMES
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'class_names': class_names,
        'config': {
            'num_classes': config.NUM_CLASSES,
            'embedding_dim': config.EMBEDDING_DIM,
            'hidden_dim': config.HIDDEN_DIM,
            'dropout': config.DROPOUT
        }
    }
    
    torch.save(checkpoint, save_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    device: torch.device = config.DEVICE
) -> Tuple[int, Dict[str, float]]:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model
        optimizer: Optimizer (optional)
        scheduler: Learning rate scheduler (optional)
        device: Device to load on
    
    Returns:
        epoch: Epoch number
        metrics: Dictionary of metrics
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    return epoch, metrics


def save_training_history(history: Dict[str, List[float]], save_path: str):
    """
    Save training history to JSON file.
    
    Args:
        history: Dictionary with training metrics
        save_path: Path to save JSON file
    """
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Training history saved to {save_path}")


def load_training_history(load_path: str) -> Dict[str, List[float]]:
    """
    Load training history from JSON file.
    
    Args:
        load_path: Path to JSON file
    
    Returns:
        history: Dictionary with training metrics
    """
    with open(load_path, 'r') as f:
        history = json.load(f)
    return history


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for metrics like accuracy/F1
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        
        if mode == 'min':
            self.is_better = lambda current, best: current < best - min_delta
        else:
            self.is_better = lambda current, best: current > best + min_delta
    
    def __call__(self, current_value: float) -> bool:
        """
        Args:
            current_value: Current metric value
        
        Returns:
            early_stop: Whether to stop training
        """
        if self.best_value is None:
            self.best_value = current_value
            return False
        
        if self.is_better(current_value, self.best_value):
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\n⚠ Early stopping triggered after {self.counter} epochs without improvement")
        
        return self.early_stop


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    """Test utility functions."""
    
    print("Testing utility functions...")
    
    # Test metrics computation
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    y_pred = np.array([0, 1, 1, 1, 2, 0, 3, 3])
    
    metrics = compute_metrics(y_true, y_pred, class_names=['A', 'B', 'C', 'D'])
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test class weights
    class_counts = config.CLASS_COUNTS
    weights = compute_class_weights(class_counts)
    print("\nClass weights:")
    for name, weight in zip(config.CLASS_NAMES, weights):
        print(f"  {name}: {weight:.4f}")
    
    # Test early stopping
    print("\nTesting early stopping...")
    early_stop = EarlyStopping(patience=3, mode='min')
    losses = [1.0, 0.9, 0.85, 0.84, 0.84, 0.84, 0.84]
    for epoch, loss in enumerate(losses):
        stop = early_stop(loss)
        print(f"  Epoch {epoch}: loss={loss:.2f}, counter={early_stop.counter}, stop={stop}")
        if stop:
            break
    
    print("\n✓ Utility tests passed!")
