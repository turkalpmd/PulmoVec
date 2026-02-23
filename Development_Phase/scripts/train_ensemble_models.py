"""
Unified training script for all 3 ensemble models.

Trains models sequentially:
    1. Model 1: Event Type Classification (3 classes)
    2. Model 2: Binary Abnormality Detection (2 classes)
    3. Model 3: Disease Group Classification (3 classes)

Each model uses two-phase training:
    Phase 1: Freeze HeAR encoder, train only classification head
    Phase 2: Unfreeze all layers for end-to-end fine-tuning

Usage:
    python train_ensemble_models.py --model all
    python train_ensemble_models.py --model 1
    python train_ensemble_models.py --model 2
    python train_ensemble_models.py --model 3
"""

import os
import sys
import argparse
import importlib
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import warnings
import json
warnings.filterwarnings('ignore')

# Import base modules
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
    epoch: int,
    config_module
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
        with autocast(enabled=config_module.USE_AMP):
            logits = model(audio)
            loss = criterion(logits, labels)
            # Scale loss for gradient accumulation
            loss = loss / config_module.ACCUMULATION_STEPS
        
        # Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠ NaN/Inf loss detected at batch {batch_idx}! Skipping this batch...")
            continue
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Only step optimizer every ACCUMULATION_STEPS
        if (batch_idx + 1) % config_module.ACCUMULATION_STEPS == 0:
            # Gradient clipping to prevent NaN
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config_module.MAX_GRAD_NORM)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Metrics (multiply back by accumulation steps for true loss)
        loss_meter.update(loss.item() * config_module.ACCUMULATION_STEPS, audio.size(0))
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
    
    if len(all_preds) == 0 or len(all_labels) == 0:
        raise ValueError("Training failed - all batches produced NaN/Inf loss")
    
    metrics = compute_metrics(all_preds, all_labels, class_names=config_module.CLASS_NAMES)
    metrics['loss'] = loss_meter.avg
    
    return metrics


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    config_module
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
    metrics = compute_metrics(all_preds, all_labels, class_names=config_module.CLASS_NAMES)
    metrics['loss'] = loss_meter.avg
    
    return metrics, all_preds, all_labels


def train_single_model(config_module_name: str):
    """
    Train a single model using specified config module.
    
    Args:
        config_module_name: Name of config module (e.g., 'config_model1')
    """
    print("\n" + "="*80)
    print(f"TRAINING: {config_module_name}")
    print("="*80)
    
    # Import config module dynamically
    config_module = importlib.import_module(config_module_name)
    
    print(f"\nModel: {config_module.MODEL_NAME}")
    print(f"Classes: {config_module.NUM_CLASSES}")
    print(f"Class Names: {config_module.CLASS_NAMES}")
    print(f"Label Column: {config_module.LABEL_COLUMN}")
    print(f"Save Directory: {config_module.MODEL_SAVE_DIR}")
    
    # Load HuggingFace token
    if config_module.ENV_FILE.exists():
        load_dotenv(config_module.ENV_FILE)
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token
            print("✓ HuggingFace token loaded")
    
    # Device
    device = config_module.DEVICE
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load data
    print("\n" + "-"*80)
    print("Loading Data")
    print("-"*80)
    
    df_train, df_val = stratified_train_val_split(
        csv_path=config_module.CSV_PATH,
        label_column=config_module.LABEL_COLUMN,
        train_ratio=config_module.TRAIN_SPLIT,
        random_seed=config_module.RANDOM_SEED
    )
    
    print(f"Training events: {len(df_train)}")
    print(f"Validation events: {len(df_val)}")
    
    # Create datasets
    train_dataset = SPRSoundDatasetFromDF(df_train, label_column=config_module.LABEL_COLUMN)
    val_dataset = SPRSoundDatasetFromDF(df_val, label_column=config_module.LABEL_COLUMN)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config_module.BATCH_SIZE,
        shuffle=True,
        num_workers=config_module.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config_module.BATCH_SIZE,
        shuffle=False,
        num_workers=config_module.NUM_WORKERS,
        pin_memory=True
    )
    
    # Calculate class weights
    train_labels = df_train[config_module.LABEL_COLUMN].values
    class_counts = np.bincount(train_labels, minlength=config_module.NUM_CLASSES)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * config_module.NUM_CLASSES
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f"\nClass distribution (train):")
    for i, name in enumerate(config_module.CLASS_NAMES):
        print(f"  {name}: {class_counts[i]} (weight: {class_weights[i]:.4f})")
    
    # Create model
    print("\n" + "-"*80)
    print("Creating Model")
    print("-"*80)
    
    model = HeARClassifier(num_classes=config_module.NUM_CLASSES).to(device)
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Gradient scaler for AMP
    scaler = GradScaler(enabled=config_module.USE_AMP)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1_macro': [], 'train_f1_weighted': [],
        'val_loss': [], 'val_acc': [], 'val_f1_macro': [], 'val_f1_weighted': [],
        'lr': []
    }
    
    best_f1_macro = 0.0
    early_stopping = EarlyStopping(patience=config_module.EARLY_STOPPING_PATIENCE, mode='max')
    
    # ====================================================================================
    # PHASE 1: Train only classification head (freeze HeAR encoder)
    # ====================================================================================
    print("\n" + "="*80)
    print("PHASE 1: Training Classification Head (HeAR Encoder Frozen)")
    print("="*80)
    
    # Freeze HeAR encoder
    for param in model.hear_encoder.parameters():
        param.requires_grad = False
    
    # Optimizer for classification head only
    optimizer_phase1 = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config_module.PHASE1_LR,
        weight_decay=config_module.WEIGHT_DECAY
    )
    
    scheduler_phase1 = ReduceLROnPlateau(
        optimizer_phase1,
        mode='max',
        factor=config_module.LR_SCHEDULER_FACTOR,
        patience=config_module.LR_SCHEDULER_PATIENCE
    )
    
    for epoch in range(1, config_module.PHASE1_EPOCHS + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer_phase1, 
            device, scaler, epoch, config_module
        )
        
        # Validate
        val_metrics, val_preds, val_labels = validate(
            model, val_loader, criterion, device, epoch, config_module
        )
        
        # Update scheduler
        scheduler_phase1.step(val_metrics['f1_macro'])
        current_lr = optimizer_phase1.param_groups[0]['lr']
        
        # Log metrics
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1_macro'].append(train_metrics['f1_macro'])
        history['train_f1_weighted'].append(train_metrics['f1_weighted'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1_macro'].append(val_metrics['f1_macro'])
        history['val_f1_weighted'].append(val_metrics['f1_weighted'])
        history['lr'].append(current_lr)
        
        # Print summary
        print(f"\nEpoch {epoch}/{config_module.PHASE1_EPOCHS} [Phase 1]")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"F1-macro: {train_metrics['f1_macro']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"F1-macro: {val_metrics['f1_macro']:.4f}")
        print(f"  LR: {current_lr:.2e}")
        
        # Save best model
        if val_metrics['f1_macro'] > best_f1_macro:
            best_f1_macro = val_metrics['f1_macro']
            save_checkpoint(
                model, optimizer_phase1, scheduler_phase1, epoch, val_metrics,
                config_module.BEST_MODEL_PATH, class_names=config_module.CLASS_NAMES
            )
            print(f"  ✓ Best model saved (F1-macro: {best_f1_macro:.4f})")
    
    print("\n✓ Phase 1 completed!")
    
    # ====================================================================================
    # PHASE 2: Fine-tune entire model (unfreeze HeAR encoder)
    # ====================================================================================
    print("\n" + "="*80)
    print("PHASE 2: Fine-tuning Entire Model (HeAR Encoder Unfrozen)")
    print("="*80)
    
    # Aggressive memory cleanup before Phase 2
    del optimizer_phase1, scheduler_phase1
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"✓ Cleared GPU cache. Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3 - torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    # New optimizer for all parameters
    optimizer_phase2 = AdamW(
        model.parameters(),
        lr=config_module.PHASE2_LR,
        weight_decay=config_module.WEIGHT_DECAY
    )
    
    scheduler_phase2 = ReduceLROnPlateau(
        optimizer_phase2,
        mode='max',
        factor=config_module.LR_SCHEDULER_FACTOR,
        patience=config_module.LR_SCHEDULER_PATIENCE
    )
    
    for epoch in range(config_module.PHASE1_EPOCHS + 1, config_module.TOTAL_EPOCHS + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer_phase2,
            device, scaler, epoch, config_module
        )
        
        # Validate
        val_metrics, val_preds, val_labels = validate(
            model, val_loader, criterion, device, epoch, config_module
        )
        
        # Update scheduler
        scheduler_phase2.step(val_metrics['f1_macro'])
        current_lr = optimizer_phase2.param_groups[0]['lr']
        
        # Log metrics
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1_macro'].append(train_metrics['f1_macro'])
        history['train_f1_weighted'].append(train_metrics['f1_weighted'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1_macro'].append(val_metrics['f1_macro'])
        history['val_f1_weighted'].append(val_metrics['f1_weighted'])
        history['lr'].append(current_lr)
        
        # Print summary
        print(f"\nEpoch {epoch}/{config_module.TOTAL_EPOCHS} [Phase 2]")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"F1-macro: {train_metrics['f1_macro']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"F1-macro: {val_metrics['f1_macro']:.4f}")
        print(f"  LR: {current_lr:.2e}")
        
        # Save best model
        if val_metrics['f1_macro'] > best_f1_macro:
            best_f1_macro = val_metrics['f1_macro']
            save_checkpoint(
                model, optimizer_phase2, scheduler_phase2, epoch, val_metrics,
                config_module.BEST_MODEL_PATH, class_names=config_module.CLASS_NAMES
            )
            print(f"  ✓ Best model saved (F1-macro: {best_f1_macro:.4f})")
        
        # Early stopping
        if early_stopping(val_metrics['f1_macro']):
            print(f"\n⚠ Early stopping triggered at epoch {epoch}")
            break
    
    # Save final checkpoint
    save_checkpoint(model, optimizer_phase2, scheduler_phase2, epoch, val_metrics, 
                    config_module.LAST_MODEL_PATH, class_names=config_module.CLASS_NAMES)
    
    # Save training history
    save_training_history(history, config_module.TRAINING_HISTORY_PATH)
    
    # Plot training curves
    plot_training_curves(history, config_module.TRAINING_CURVES_PATH)
    
    # Load best model for final evaluation
    checkpoint = torch.load(config_module.BEST_MODEL_PATH, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION (Best Model)")
    print("="*80)
    
    val_metrics, val_preds, val_labels = validate(model, val_loader, criterion, device, 0, config_module)
    
    print("\nValidation Metrics:")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  F1-Macro: {val_metrics['f1_macro']:.4f}")
    print(f"  F1-Weighted: {val_metrics['f1_weighted']:.4f}")
    
    # Classification report
    print_classification_report(
        val_labels, val_preds, 
        class_names=config_module.CLASS_NAMES
    )
    
    # Confusion matrix
    plot_confusion_matrix(
        val_labels, val_preds,
        class_names=config_module.CLASS_NAMES,
        save_path=config_module.CONFUSION_MATRIX_PATH
    )
    
    print(f"\n✅ Training completed for {config_module.MODEL_NAME}!")
    print(f"   Best F1-Macro: {best_f1_macro:.4f}")
    print(f"   Model saved to: {config_module.BEST_MODEL_PATH}")
    
    # Aggressive cleanup at end of training
    del model, optimizer_phase2, scheduler_phase2, train_loader, val_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description='Train ensemble models')
    parser.add_argument('--model', type=str, default='all', 
                        choices=['all', '1', '2', '3'],
                        help='Which model to train: all, 1, 2, or 3')
    args = parser.parse_args()
    
    models_to_train = []
    if args.model == 'all':
        models_to_train = ['config_model1', 'config_model2', 'config_model3']
    else:
        models_to_train = [f'config_model{args.model}']
    
    print("\n" + "="*80)
    print("ENSEMBLE MODEL TRAINING")
    print("="*80)
    print(f"Models to train: {', '.join(models_to_train)}")
    
    for config_name in models_to_train:
        try:
            train_single_model(config_name)
            
            # Aggressive cleanup after each model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(f"\n✓ Cleaned GPU memory after {config_name}")
                
        except Exception as e:
            print(f"\n❌ Error training {config_name}: {e}")
            import traceback
            traceback.print_exc()
            
            # Cleanup on error too
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    
    print("\n" + "="*80)
    print("ALL TRAINING COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()
