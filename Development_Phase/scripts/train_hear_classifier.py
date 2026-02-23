"""
Main training script for HeAR-based respiratory event classification.

Two-phase training:
    Phase 1: Freeze HeAR encoder, train only classification head (10 epochs)
    Phase 2: Unfreeze all layers for end-to-end fine-tuning (40 epochs)

Usage:
    python train_hear_classifier.py
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
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
    """
    Train for one epoch.
    
    Returns:
        metrics: Dictionary with loss, accuracy, f1_macro, f1_weighted
    """
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
        print("This usually means:")
        print("  1. Learning rate is too high")
        print("  2. Model weights have exploded")
        print("  3. Input data has extreme values")
        print("\nTry:")
        print("  - Reducing PHASE2_LR to 1e-7")
        print("  - Checking data for anomalies")
        print("  - Restarting training from scratch")
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
) -> dict:
    """
    Validate the model.
    
    Returns:
        metrics: Dictionary with loss, accuracy, f1_macro, f1_weighted
        all_preds: Predictions
        all_labels: True labels
    """
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


def train_model():
    """Main training function."""
    
    print("="*70)
    print("HeAR SPRSound Respiratory Event Classification Training")
    print("="*70)
    
    # Load HuggingFace token from .env
    print("\n" + "="*70)
    print("HuggingFace Authentication")
    print("="*70)
    
    if config.ENV_FILE.exists():
        load_dotenv(config.ENV_FILE)
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token
            print("✓ HuggingFace token loaded from .env")
        else:
            print("⚠ HF_TOKEN not found in .env file")
            print("\nTo authenticate, either:")
            print("  1. Run: huggingface-cli login")
            print("  2. Create .env with: HF_TOKEN=your_token")
    else:
        print(f"⚠ .env file not found at {config.ENV_FILE}")
        print("\nTo authenticate, either:")
        print("  1. Run: huggingface-cli login")
        print("  2. Create .env with: HF_TOKEN=your_token")
    
    print("\n⚠ IMPORTANT: HeAR model requires access approval!")
    print("Visit: https://huggingface.co/google/hear-pytorch")
    print("Click 'Request Access' and wait for approval (usually instant)\n")
    
    # Set device
    device = config.DEVICE
    print(f"\n✓ Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
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
    
    # Create dataloaders
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
    print("Creating Model")
    print("="*70)
    
    model = HeARClassifier()
    model.to(device)
    model.count_parameters()
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1_macro': [],
        'train_f1_weighted': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1_macro': [],
        'val_f1_weighted': [],
        'lr': []
    }
    
    # Best model tracking
    best_f1_macro = 0.0
    best_epoch = 0
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        mode='max'  # Maximize F1-macro
    )
    
    # Gradient scaler for mixed precision
    scaler = GradScaler(enabled=config.USE_AMP)
    
    # ========================================================================
    # PHASE 1: Train only classification head (freeze encoder)
    # ========================================================================
    
    print("\n" + "="*70)
    print("PHASE 1: Training Classification Head (Encoder Frozen)")
    print("="*70)
    
    model.freeze_encoder()
    model.count_parameters()
    
    optimizer_phase1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.PHASE1_LR,
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler_phase1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_phase1,
        mode='max',  # Maximize F1
        factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE
    )
    
    for epoch in range(1, config.PHASE1_EPOCHS + 1):
        print(f"\n{'='*70}")
        print(f"Phase 1 - Epoch {epoch}/{config.PHASE1_EPOCHS}")
        print(f"{'='*70}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer_phase1, device, scaler, epoch
        )
        
        # Validate
        val_metrics, val_preds, val_labels = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Update scheduler
        old_lr = optimizer_phase1.param_groups[0]['lr']
        scheduler_phase1.step(val_metrics['f1_macro'])
        new_lr = optimizer_phase1.param_groups[0]['lr']
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
        history['lr'].append(optimizer_phase1.param_groups[0]['lr'])
        
        # Save best model
        if val_metrics['f1_macro'] > best_f1_macro:
            best_f1_macro = val_metrics['f1_macro']
            best_epoch = epoch
            save_checkpoint(
                model, optimizer_phase1, scheduler_phase1, epoch,
                val_metrics, config.BEST_MODEL_PATH
            )
            print(f"  ✓ Best model saved (F1-macro: {best_f1_macro:.4f})")
        
        # Check early stopping
        if early_stopping(val_metrics['f1_macro']):
            print(f"\n⚠ Early stopping at epoch {epoch}")
            break
    
    # ========================================================================
    # PHASE 2: Fine-tune entire model (unfreeze encoder)
    # ========================================================================
    
    print("\n" + "="*70)
    print("PHASE 2: Fine-tuning Entire Model (Encoder Unfrozen)")
    print("="*70)
    
    model.unfreeze_encoder()
    model.count_parameters()
    
    optimizer_phase2 = torch.optim.AdamW(
        model.parameters(),
        lr=config.PHASE2_LR,
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler_phase2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_phase2,
        mode='max',
        factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE
    )
    
    # Reset early stopping for phase 2
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        mode='max'
    )
    
    for epoch in range(config.PHASE1_EPOCHS + 1, config.TOTAL_EPOCHS + 1):
        print(f"\n{'='*70}")
        print(f"Phase 2 - Epoch {epoch}/{config.TOTAL_EPOCHS}")
        print(f"{'='*70}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer_phase2, device, scaler, epoch
        )
        
        # Validate
        val_metrics, val_preds, val_labels = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Update scheduler
        old_lr = optimizer_phase2.param_groups[0]['lr']
        scheduler_phase2.step(val_metrics['f1_macro'])
        new_lr = optimizer_phase2.param_groups[0]['lr']
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
        history['lr'].append(optimizer_phase2.param_groups[0]['lr'])
        
        # Save best model
        if val_metrics['f1_macro'] > best_f1_macro:
            best_f1_macro = val_metrics['f1_macro']
            best_epoch = epoch
            save_checkpoint(
                model, optimizer_phase2, scheduler_phase2, epoch,
                val_metrics, config.BEST_MODEL_PATH
            )
            print(f"  ✓ Best model saved (F1-macro: {best_f1_macro:.4f})")
        
        # Check early stopping
        if early_stopping(val_metrics['f1_macro']):
            print(f"\n⚠ Early stopping at epoch {epoch}")
            break
    
    # ========================================================================
    # Final Evaluation
    # ========================================================================
    
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
        model, optimizer_phase2, scheduler_phase2, epoch,
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
    train_model()
