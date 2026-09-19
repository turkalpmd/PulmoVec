#!/usr/bin/env python3
"""
scripts/train_hear_lora.py

Advanced Training Pipeline for HeAR with LoRA & Attentive Patch Pooling:
- Preprocessing: 100-1800 Hz Butterworth Bandpass + 0.90 Peak Normalization
- Option B Centering & Physiological Margin Padding (0% neighbor leakage)
- Zero-Leakage 80:10:10 Stratified Patient-Grouped Split
- High-Throughput GPU Pipeline: Batch Size 64, num_workers=8, pin_memory, prefetch_factor=2
- Gradient Descent: AdamW + Gradient Clipping (max_norm=1.0) + Grad Norm Tracking
- LR Scheduler: Linear Warmup (2 epochs) + Cosine Annealing decay
- Multi-Class Focal Loss (gamma=2.0) with Inverse Frequency Class Weights
- Double Checkpoint Saving: Best Model ('hear_lora_best.pth') & Last Model ('hear_lora_last.pth')
- Early Stopping (Patience = 6 epochs)
- Unbiased Held-Out Test Set Evaluation
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

# Add project root and src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import config
from sprsound_dataset import stratified_train_val_test_split, SPRSoundDatasetFromDF
from models_hear_lora import HeARLoRAClassifier

class FocalLoss(nn.Module):
    """
    Multi-Class Focal Loss to combat severe class imbalance:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Gradient Modulation:
    Well-classified 'Normal' samples (p_t -> 1) have their gradient scaled down by (1 - p_t)^2 -> 0,
    forcing 90%+ of parameter updates toward hard adventitious sounds (Crackles, Wheezes).
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        return focal_loss.mean()


class WarmupCosineScheduler:
    """Linear Warmup followed by Cosine Annealing Learning Rate Decay."""
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, base_lr: float, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = max(1, warmup_epochs)
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

    def step(self, epoch: int):
        if epoch <= self.warmup_epochs:
            lr = self.min_lr + (self.base_lr - self.min_lr) * (epoch / float(self.warmup_epochs))
        else:
            progress = (epoch - self.warmup_epochs) / float(max(1, self.total_epochs - self.warmup_epochs))
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + np.cos(np.pi * progress))
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float = 1.0
) -> Tuple[float, float, float, float]:
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    total_grad_norm = 0.0
    num_steps = 0
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    for audio, targets, _ in loader:
        audio = audio.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(audio)
            loss = criterion(logits, targets)
            
        loss.backward()
        
        # Gradient Clipping & Norm Tracking
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
        optimizer.step()
        
        total_grad_norm += float(grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm)
        total_loss += loss.item() * len(targets)
        num_steps += 1
        
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())
        
    avg_loss = total_loss / len(all_targets)
    avg_grad_norm = total_grad_norm / max(1, num_steps)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    return avg_loss, acc, macro_f1, avg_grad_norm


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    for audio, targets, _ in loader:
        audio = audio.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(audio)
            loss = criterion(logits, targets)
            
        total_loss += loss.item() * len(targets)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())
        
    avg_loss = total_loss / len(all_targets)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)
    
    return {
        'loss': avg_loss,
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class_f1': per_class_f1,
        'confusion_matrix': confusion_matrix(all_targets, all_preds),
        'targets': all_targets,
        'preds': all_preds
    }


def main():
    parser = argparse.ArgumentParser(description="Train HeAR with LoRA & Attentive Patch Pooling")
    parser.add_argument('--epochs', type=int, default=25, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size (default: 32 for thermal & power safety)")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of DataLoader worker processes")
    parser.add_argument('--lr', type=float, default=2e-4, help="Peak learning rate for LoRA & Pooling head")
    parser.add_argument('--patience', type=int, default=6, help="Early stopping patience (epochs without val improvement)")
    parser.add_argument('--lora_r', type=int, default=8, help="LoRA rank")
    parser.add_argument('--lora_alpha', type=float, default=16.0, help="LoRA scaling alpha")
    parser.add_argument('--label_col', type=str, default='model1_label', help="Target label column (model1_label, model2_label, model3_label)")
    parser.add_argument('--save_dir', type=str, default=str(project_root / "models" / "hear_lora_checkpoints"))
    parser.add_argument('--resume', action='store_true', help="Resume training from last checkpoint if available")
    args = parser.parse_args()
    
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    print("="*75)
    print(f"PULMOVEC: HeAR LoRA (Layers 18-23) + ATTENTIVE PATCH POOLING TRAINING")
    print(f"GPU Hardware:   {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Target Task:    {args.label_col}")
    print(f"Batch Size:     {args.batch_size} (num_workers={args.num_workers}, pin_memory=True)")
    print(f"Optimizer:      AdamW (lr={args.lr}, weight_decay=0.01, clip_norm=1.0)")
    print(f"Scheduler:      Warmup (2 epochs) + Cosine Annealing")
    print(f"Early Stopping: Patience = {args.patience} epochs on Val Macro F1")
    print("="*75)
    
    # 1. 3-Way Zero-Leakage Dataset Split (80:10:10)
    train_df, val_df, test_df = stratified_train_val_test_split(
        label_column=args.label_col,
        random_seed=42
    )
    
    train_dataset = SPRSoundDatasetFromDF(train_df, label_column=args.label_col, is_training=True)
    val_dataset = SPRSoundDatasetFromDF(val_df, label_column=args.label_col, is_training=False)
    test_dataset = SPRSoundDatasetFromDF(test_df, label_column=args.label_col, is_training=False)
    
    # Multi-worker prefetching for high GPU throughput
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, prefetch_factor=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, prefetch_factor=2
    )
    
    num_classes = len(train_dataset.class_names)
    print(f"\nTask Classes ({num_classes}): {train_dataset.class_names}")
    
    # Inverse Frequency Class Weights for Focal Loss
    class_weights = train_dataset.get_class_weights().to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    
    # 2. Build Model
    model = HeARLoRAClassifier(
        num_classes=num_classes,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_layers=6,
        dropout=0.3
    ).to(device)
    
    # Only optimize trainable parameters (LoRA + Attention Pooling + BatchNorm/Classifier)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=2, total_epochs=args.epochs, base_lr=args.lr)
    
    best_val_f1 = 0.0
    best_epoch = -1
    epochs_no_improve = 0
    start_epoch = 1
    history = []
    
    best_ckpt_path = save_path / f"hear_lora_best_{args.label_col}.pth"
    last_ckpt_path = save_path / f"hear_lora_last_{args.label_col}.pth"
    
    # Resume from checkpoint if requested
    if args.resume and last_ckpt_path.exists():
        print(f"\nResuming training from: {last_ckpt_path}")
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_f1 = ckpt.get('val_f1', 0.0)
        best_epoch = ckpt.get('epoch', -1)
        print(f"✓ Resumed successfully at Epoch {start_epoch}! Current Best Val F1: {best_val_f1*100:.2f}%")
    
    print("\nStarting High-Performance Training Loop...")
    print(f"{'Epoch':<7} | {'Train Loss':<10} | {'Train F1':<8} | {'GradNorm':<8} | {'LR':<9} | {'Val Loss':<8} | {'Val Acc':<8} | {'Val F1 (Macro)':<14} | {'Status'}")
    print("-" * 105)
    
    for epoch in range(start_epoch, args.epochs + 1):
        start_t = time.time()
        
        current_lr = scheduler.step(epoch)
        train_loss, train_acc, train_f1, grad_norm = train_one_epoch(
            model, train_loader, optimizer, criterion, device, max_grad_norm=1.0
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        elapsed = time.time() - start_t
        val_f1 = val_metrics['macro_f1']
        val_acc = val_metrics['accuracy']
        
        status_msg = ""
        # 1. Always Save Last Model (for resuming & full state inspection)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': val_f1,
            'val_metrics': val_metrics,
            'class_names': train_dataset.class_names,
            'args': vars(args)
        }, last_ckpt_path)
        
        # 2. Check for Best Model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_metrics': val_metrics,
                'class_names': train_dataset.class_names,
                'args': vars(args)
            }, best_ckpt_path)
            status_msg = f"⭐ BEST ({val_f1*100:.2f}%)"
        else:
            epochs_no_improve += 1
            status_msg = f"({epochs_no_improve}/{args.patience})"
            
        print(f"{epoch:02d}/{args.epochs:02d}  | {train_loss:<10.4f} | {train_f1*100:<7.1f}% | {grad_norm:<8.3f} | {current_lr:<9.2e} | {val_metrics['loss']:<8.4f} | {val_acc*100:<7.1f}% | {val_f1*100:<13.2f}% | {status_msg} [{elapsed:.1f}s]")
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_f1': train_f1,
            'grad_norm': grad_norm,
            'lr': current_lr,
            'val_loss': val_metrics['loss'],
            'val_f1': val_f1,
            'val_acc': val_acc
        })
        
        # Early Stopping Trigger
        if epochs_no_improve >= args.patience:
            print(f"\n⏹️ EARLY STOPPING TRIGGERED: No validation improvement for {args.patience} consecutive epochs.")
            break
            
    print("\n" + "="*75)
    print(f"TRAINING FINISHED! Best Model from Epoch {best_epoch} with Val Macro F1: {best_val_f1*100:.2f}%")
    print(f"Best Checkpoint Saved: {best_ckpt_path}")
    print(f"Last Checkpoint Saved: {last_ckpt_path}")
    print("="*75)
    
    # 3. Final Evaluation on Held-Out Test Set (Zero-Leakage Unseen Cohort)
    print("\nLoading Best Model Checkpoint for Final Held-Out Test Set Evaluation...")
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    print("\n" + "="*75)
    print(f"FINAL UNBIASED HELD-OUT TEST SET RESULTS ({args.label_col}):")
    print("="*75)
    print(f"  Test Accuracy:     {test_metrics['accuracy']*100:.2f}%")
    print(f"  Test Macro F1:     {test_metrics['macro_f1']*100:.2f}%")
    print(f"  Test Weighted F1:  {test_metrics['weighted_f1']*100:.2f}%")
    print(f"  Test Loss:         {test_metrics['loss']:.4f}")
    print("\nPer-Class Test Performance Breakdown:")
    for i, c_name in enumerate(train_dataset.class_names):
        f1_c = test_metrics['per_class_f1'][i]
        print(f"  - {c_name:<18}: {f1_c*100:.2f}%")
        
    print(f"\nConfusion Matrix:\n{test_metrics['confusion_matrix']}")
    print("="*75)
    
    # Save test metrics json
    res_path = save_path / f"test_metrics_{args.label_col}.json"
    test_summary = {
        'task': args.label_col,
        'best_epoch': best_epoch,
        'val_macro_f1': best_val_f1,
        'test_accuracy': test_metrics['accuracy'],
        'test_macro_f1': test_metrics['macro_f1'],
        'test_weighted_f1': test_metrics['weighted_f1'],
        'per_class_f1': {c_name: float(test_metrics['per_class_f1'][i]) for i, c_name in enumerate(train_dataset.class_names)},
        'args': vars(args)
    }
    with open(res_path, 'w') as f:
        json.dump(test_summary, f, indent=2)
    print(f"Test summary saved to: {res_path}")

if __name__ == '__main__':
    main()
