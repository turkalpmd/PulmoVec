"""
Resume training script - Continue from best checkpoint for additional epochs.

Usage:
    python resume_training.py --additional-epochs 10
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

# Import local modules
import config
from models import HeARClassifier
from sprsound_dataset import SPRSoundDatasetFromDF, stratified_train_val_split
from utils import (
    compute_metrics,
    print_classification_report,
    plot_confusion_matrix,
    plot_training_curves,
    save_checkpoint,
    save_training_history,
    EarlyStopping,
    AverageMeter
)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    epoch: int
) -> dict:
    """Train for one epoch."""
    model.train()
    
    loss_meter = AverageMeter()
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (audio, labels, metadata) in enumerate(pbar):
        audio = audio.to(device)
        labels = labels.to(device)
        
        # Mixed precision training
        with autocast(enabled=config.USE_AMP):
            logits = model(audio)
            loss = criterion(logits, labels)
            # Scale loss for gradient accumulation
            loss = loss / config.ACCUMULATION_STEPS
        
        # Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠ NaN/Inf loss detected at batch {batch_idx}! Skipping this batch...")
            continue
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Only step optimizer every ACCUMULATION_STEPS
        if (batch_idx + 1) % config.ACCUMULATION_STEPS == 0:
            # Gradient clipping to prevent NaN
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Metrics (multiply back by accumulation steps for true loss)
        loss_meter.update(loss.item() * config.ACCUMULATION_STEPS, audio.size(0))
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
    
    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Check if we have any valid predictions
    if len(all_preds) == 0 or len(all_labels) == 0:
        print("\n❌ ERROR: All batches had NaN/Inf loss! No valid predictions.")
        raise ValueError("Training failed - all batches produced NaN/Inf loss")
    
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = loss_meter.avg
    
    return metrics


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> tuple:
    """Validate the model."""
    model.eval()
    
    loss_meter = AverageMeter()
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]  ")
    
    with torch.no_grad():
        for audio, labels, metadata in pbar:
            audio = audio.to(device)
            labels = labels.to(device)
            
            logits = model(audio)
            loss = criterion(logits, labels)
            
            loss_meter.update(loss.item(), audio.size(0))
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
    
    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = loss_meter.avg
    
    return metrics, all_preds, all_labels


def resume_training(additional_epochs: int = 10):
    """Resume training from best checkpoint."""
    
    print("="*70)
    print("Resume Training from Checkpoint")
    print("="*70)
    
    # Check if checkpoint exists
    if not config.BEST_MODEL_PATH.exists():
        print(f"\n❌ Checkpoint not found: {config.BEST_MODEL_PATH}")
        print("Please train the model first using train_hear_classifier.py")
        return
    
    # Load training history
    if config.TRAINING_HISTORY_PATH.exists():
        with open(config.TRAINING_HISTORY_PATH, 'r') as f:
            history = json.load(f)
        print(f"\n✓ Loaded training history with {len(history['train_loss'])} epochs")
        start_epoch = len(history['train_loss']) + 1
    else:
        print("\n⚠ No training history found, starting fresh history")
        history = {
            'train_loss': [], 'train_acc': [], 'train_f1_macro': [], 'train_f1_weighted': [],
            'val_loss': [], 'val_acc': [], 'val_f1_macro': [], 'val_f1_weighted': [], 'lr': []
        }
        start_epoch = 1
    
    end_epoch = start_epoch + additional_epochs - 1
    
    print(f"Will train from epoch {start_epoch} to {end_epoch}")
    print(f"Previous best F1-macro: {max(history['val_f1_macro']) if history['val_f1_macro'] else 'N/A'}")
    
    # Set device
    device = config.DEVICE
    print(f"\n✓ Device: {device}")
    
    # Load data
    print("\n" + "="*70)
    print("Loading Dataset")
    print("="*70)
    
    train_df, val_df = stratified_train_val_split(
        csv_path=config.CSV_PATH,
        train_ratio=config.TRAIN_SPLIT,
        random_seed=config.RANDOM_SEED
    )
    
    train_dataset = SPRSoundDatasetFromDF(train_df)
    val_dataset = SPRSoundDatasetFromDF(val_df)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"\n✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    # Calculate class weights
    class_weights = train_dataset.get_class_weights().to(device)
    
    # Create model
    print("\n" + "="*70)
    print("Loading Model from Checkpoint")
    print("="*70)
    
    model = HeARClassifier()
    model.to(device)
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {config.BEST_MODEL_PATH}")
    checkpoint = torch.load(config.BEST_MODEL_PATH, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"✓ Previous val F1-macro: {checkpoint.get('metrics', {}).get('f1_macro', 'N/A')}")
    
    # Model should already be unfrozen from Phase 2
    model.unfreeze_encoder()
    model.count_parameters()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.PHASE2_LR,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Restore optimizer state if available
    if 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✓ Optimizer state restored")
        except:
            print("⚠ Could not restore optimizer state, starting fresh")
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE
    )
    
    # Restore scheduler state if available
    if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("✓ Scheduler state restored")
        except:
            print("⚠ Could not restore scheduler state, starting fresh")
    
    # Best model tracking
    best_f1_macro = max(history['val_f1_macro']) if history['val_f1_macro'] else 0.0
    best_epoch = start_epoch - 1
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        mode='max'
    )
    
    # Gradient scaler
    scaler = GradScaler(enabled=config.USE_AMP)
    
    # Training loop
    print("\n" + "="*70)
    print(f"Resuming Training (Epochs {start_epoch}-{end_epoch})")
    print("="*70)
    
    for epoch in range(start_epoch, end_epoch + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{end_epoch}")
        print(f"{'='*70}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch
        )
        
        # Validate
        val_metrics, val_preds, val_labels = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Update scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_metrics['f1_macro'])
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"  📉 Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
        
        # Log metrics
        print(f"\n  Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"F1-macro: {train_metrics['f1_macro']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, "
              f"F1-macro: {val_metrics['f1_macro']:.4f}")
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1_macro'].append(train_metrics['f1_macro'])
        history['train_f1_weighted'].append(train_metrics['f1_weighted'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1_macro'].append(val_metrics['f1_macro'])
        history['val_f1_weighted'].append(val_metrics['f1_weighted'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Save best model
        if val_metrics['f1_macro'] > best_f1_macro:
            best_f1_macro = val_metrics['f1_macro']
            best_epoch = epoch
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_metrics, config.BEST_MODEL_PATH
            )
            print(f"  ✓ Best model saved (F1-macro: {best_f1_macro:.4f})")
        
        # Check early stopping
        if early_stopping(val_metrics['f1_macro']):
            print(f"\n⚠ Early stopping at epoch {epoch}")
            break
    
    # Final evaluation
    print("\n" + "="*70)
    print("Final Evaluation on Best Model")
    print("="*70)
    
    # Load best model
    checkpoint = torch.load(config.BEST_MODEL_PATH, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Evaluate on validation set
    val_metrics, val_preds, val_labels = validate(
        model, val_loader, criterion, device, epoch=0
    )
    
    print(f"\n✓ Best Model (Epoch {best_epoch}):")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  F1-macro: {val_metrics['f1_macro']:.4f}")
    print(f"  F1-weighted: {val_metrics['f1_weighted']:.4f}")
    
    # Detailed classification report
    print_classification_report(val_labels, val_preds, config.CLASS_NAMES)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        val_labels, val_preds,
        class_names=config.CLASS_NAMES,
        save_path=config.CONFUSION_MATRIX_PATH,
        normalize=True
    )
    
    # Plot training curves
    plot_training_curves(history, save_path=config.TRAINING_CURVES_PATH)
    
    # Save training history
    save_training_history(history, config.TRAINING_HISTORY_PATH)
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, epoch,
        val_metrics, config.LAST_MODEL_PATH
    )
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"\n✓ Best model: {config.BEST_MODEL_PATH}")
    print(f"✓ Last model: {config.LAST_MODEL_PATH}")
    print(f"✓ Training history: {config.TRAINING_HISTORY_PATH}")
    print(f"✓ Confusion matrix: {config.CONFUSION_MATRIX_PATH}")
    print(f"✓ Training curves: {config.TRAINING_CURVES_PATH}")
    print(f"\n✓ Best F1-macro: {best_f1_macro:.4f} (Epoch {best_epoch})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resume training from checkpoint')
    parser.add_argument('--additional-epochs', type=int, default=10,
                        help='Number of additional epochs to train (default: 10)')
    
    args = parser.parse_args()
    
    resume_training(additional_epochs=args.additional_epochs)
