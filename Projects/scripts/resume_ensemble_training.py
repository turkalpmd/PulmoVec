#!/usr/bin/env python3
"""
Resume Ensemble Training Script
Continues training from the best checkpoint for each model with additional epochs.
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

# Make Projects/src/ importable regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import config
from models import HeARClassifier
from utils import (
    compute_metrics, print_classification_report, 
    plot_confusion_matrix, save_checkpoint, load_checkpoint,
    AverageMeter, EarlyStopping
)
from sprsound_dataset import load_and_split_data


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
            print(f"⚠ NaN/Inf loss detected at batch {batch_idx}, skipping...")
            continue
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % config_module.ACCUMULATION_STEPS == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config_module.MAX_GRAD_NORM)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        loss_meter.update(loss.item() * config_module.ACCUMULATION_STEPS, audio.size(0))
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}', 'lr': f'{current_lr:.2e}'})
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, class_names=config_module.CLASS_NAMES)
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
    metrics = compute_metrics(all_labels, all_preds, class_names=config_module.CLASS_NAMES)
    metrics['loss'] = loss_meter.avg
    
    return metrics, np.array(all_preds), np.array(all_labels)


def resume_single_model(config_module_name: str, additional_epochs: int):
    """Resume training for a single model."""
    
    # Load config
    config_module = importlib.import_module(config_module_name)
    
    print("\n" + "="*80)
    print(f"RESUMING TRAINING: {config_module_name}")
    print("="*80)
    print(f"Model: {config_module.MODEL_NAME}")
    print(f"Classes: {config_module.NUM_CLASSES}")
    print(f"Additional Epochs: {additional_epochs}")
    print(f"Best Model Path: {config_module.BEST_MODEL_PATH}")
    
    # Load environment
    load_dotenv(config.ENV_FILE)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load data
    print("\n" + "-"*80)
    print("Loading Data")
    print("-"*80)
    
    train_loader, val_loader, train_df, val_df = load_and_split_data(
        csv_path=config.CSV_PATH,
        label_column=config_module.LABEL_COLUMN,
        class_names=config_module.CLASS_NAMES,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        train_ratio=config.TRAIN_VAL_SPLIT,
        random_seed=config.RANDOM_SEED
    )
    
    # Create model
    print("\n" + "-"*80)
    print("Loading Model from Checkpoint")
    print("-"*80)
    
    model = HeARClassifier(num_classes=config_module.NUM_CLASSES).to(device)
    
    # Load checkpoint
    if not config_module.BEST_MODEL_PATH.exists():
        print(f"❌ Best model checkpoint not found at {config_module.BEST_MODEL_PATH}")
        print("   Please train the model first using train_ensemble_models.py")
        return
    
    checkpoint = load_checkpoint(config_module.BEST_MODEL_PATH, model, device=device)
    print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    print(f"  Previous best F1-macro: {checkpoint.get('f1_macro', '?'):.4f}")
    
    # Unfreeze all parameters for Phase 2
    for param in model.parameters():
        param.requires_grad = True
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.PHASE2_LR,
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=config.PATIENCE,
        min_lr=1e-8
    )
    
    # Loss function
    class_counts = train_df[config_module.LABEL_COLUMN].value_counts().sort_index()
    class_weights = 1.0 / class_counts.values
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # AMP scaler
    scaler = GradScaler(enabled=config.USE_AMP)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)
    
    # Load training history if exists
    history_path = config_module.MODEL_SAVE_DIR / "training_history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
        print(f"✓ Loaded existing training history ({len(history['train_loss'])} epochs)")
    else:
        history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': []
        }
    
    # Training loop
    print("\n" + "="*80)
    print(f"CONTINUING PHASE 2: Fine-tuning with +{additional_epochs} epochs")
    print("="*80)
    
    best_f1_macro = checkpoint.get('f1_macro', 0.0)
    start_epoch = len(history['train_loss']) + 1
    end_epoch = start_epoch + additional_epochs
    
    for epoch in range(start_epoch, end_epoch + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch, config_module
        )
        
        # Validate
        val_metrics, val_preds, val_labels = validate(
            model, val_loader, criterion, device, epoch, config_module
        )
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1_macro'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1_macro'])
        
        # Learning rate scheduling
        scheduler.step(val_metrics['f1_macro'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{end_epoch} [Phase 2]")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"F1-macro: {train_metrics['f1_macro']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, "
              f"F1-macro: {val_metrics['f1_macro']:.4f}")
        print(f"  LR: {current_lr:.2e}")
        
        # Save best model
        if val_metrics['f1_macro'] > best_f1_macro:
            best_f1_macro = val_metrics['f1_macro']
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                config_module.BEST_MODEL_PATH, class_names=config_module.CLASS_NAMES
            )
            print(f"  ✓ Best model saved (F1-macro: {best_f1_macro:.4f})")
        
        # Early stopping
        if early_stopping(val_metrics['f1_macro']):
            print(f"\n⚠ Early stopping triggered at epoch {epoch}")
            break
    
    # Save final checkpoint
    save_checkpoint(model, optimizer, scheduler, end_epoch, val_metrics, 
                    config_module.MODEL_SAVE_DIR / "last.pth", class_names=config_module.CLASS_NAMES)
    
    # Save training history
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Training history saved to {history_path}")
    
    # Plot training curves
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs_range, history['train_loss'], label='Train')
    axes[0].plot(epochs_range, history['val_loss'], label='Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(epochs_range, history['train_acc'], label='Train')
    axes[1].plot(epochs_range, history['val_acc'], label='Val')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True)
    
    # F1-macro
    axes[2].plot(epochs_range, history['train_f1'], label='Train')
    axes[2].plot(epochs_range, history['val_f1'], label='Val')
    axes[2].set_title('F1-Macro')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    curves_path = config_module.MODEL_SAVE_DIR / "training_curves.png"
    plt.savefig(curves_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Training curves saved to {curves_path}")
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION (Best Model)")
    print("="*80)
    
    # Load best model
    model = HeARClassifier(num_classes=config_module.NUM_CLASSES).to(device)
    load_checkpoint(config_module.BEST_MODEL_PATH, model, device=device)
    
    val_metrics, val_preds, val_labels = validate(model, val_loader, criterion, device, 0, config_module)
    
    print(f"\nValidation Metrics:")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  F1-Macro: {val_metrics['f1_macro']:.4f}")
    print(f"  F1-Weighted: {val_metrics['f1_weighted']:.4f}")
    
    # Classification report
    print_classification_report(
        val_labels, val_preds, 
        class_names=config_module.CLASS_NAMES
    )
    
    # Confusion matrix
    cm_path = config_module.MODEL_SAVE_DIR / "confusion_matrix.png"
    plot_confusion_matrix(
        val_labels, val_preds, 
        config_module.CLASS_NAMES, 
        cm_path
    )
    
    print(f"\n✅ Training completed for {config_module.MODEL_NAME}!")
    print(f"   Best F1-Macro: {best_f1_macro:.4f}")
    print(f"   Model saved to: {config_module.BEST_MODEL_PATH}")
    
    # Aggressive cleanup at end of training
    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description="Resume ensemble model training")
    parser.add_argument('--model1_epochs', type=int, default=0, 
                        help='Additional epochs for Model 1 (Event Type)')
    parser.add_argument('--model2_epochs', type=int, default=0, 
                        help='Additional epochs for Model 2 (Binary)')
    parser.add_argument('--model3_epochs', type=int, default=0, 
                        help='Additional epochs for Model 3 (Disease)')
    
    args = parser.parse_args()
    
    # Train models that have additional epochs
    if args.model1_epochs > 0:
        print("\n" + "🔄 " * 40)
        print(f"Model 1 (Event Type): +{args.model1_epochs} epochs")
        print("🔄 " * 40)
        resume_single_model("config_model1", args.model1_epochs)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("\n✓ Cleaned GPU memory after Model 1")
    
    if args.model2_epochs > 0:
        print("\n" + "🔄 " * 40)
        print(f"Model 2 (Binary): +{args.model2_epochs} epochs")
        print("🔄 " * 40)
        resume_single_model("config_model2", args.model2_epochs)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("\n✓ Cleaned GPU memory after Model 2")
    
    if args.model3_epochs > 0:
        print("\n" + "🔄 " * 40)
        print(f"Model 3 (Disease): +{args.model3_epochs} epochs")
        print("🔄 " * 40)
        resume_single_model("config_model3", args.model3_epochs)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("\n✓ Cleaned GPU memory after Model 3")
    
    if args.model1_epochs == 0 and args.model2_epochs == 0 and args.model3_epochs == 0:
        print("\n⚠️  No additional epochs specified!")
        print("\nUsage examples:")
        print("  # Resume Model 1 with 10 more epochs:")
        print("  python scripts/resume_ensemble_training.py --model1_epochs 10")
        print("")
        print("  # Resume all models with different epoch counts:")
        print("  python scripts/resume_ensemble_training.py --model1_epochs 5 --model2_epochs 10 --model3_epochs 15")
    else:
        print("\n" + "="*80)
        print("✅ ALL RESUME TRAINING COMPLETED!")
        print("="*80)


if __name__ == "__main__":
    main()
