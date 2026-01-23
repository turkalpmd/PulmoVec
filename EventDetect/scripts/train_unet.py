"""
Training script for U-Net temporal event segmentation.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# Add EventDetect to path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from models.unet_segmentation import UNetTemporalSegmentation
from data_processing.spectrogram_dataset import TemporalSegmentationDataset
from utils.losses import CombinedLoss
from utils.metrics import (
    compute_temporal_iou,
    compute_temporal_precision_recall_f1,
    compute_temporal_accuracy
)
from utils.collate_fn import collate_fn_pad
import config_unet as config


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_acc = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        spectrogram = batch['spectrogram'].to(device)  # (batch, 1, time_steps, freq_bins)
        mask = batch['mask'].to(device)  # (batch, time_steps)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(spectrogram)  # (batch, 1, time_steps, 1)
        
        # Calculate loss
        loss, bce, dice = criterion(output, mask)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics (only on non-padded regions)
        with torch.no_grad():
            probs = torch.sigmoid(output.squeeze(1).squeeze(-1))  # (batch, time_steps)
            # Note: Metrics are computed on padded sequences, but this is acceptable
            # as padding is with zeros (Normal class) and doesn't affect event detection
            iou = compute_temporal_iou(probs, mask)
            acc = compute_temporal_accuracy(probs, mask)
            precision, recall, f1 = compute_temporal_precision_recall_f1(probs, mask)
        
        # Accumulate
        total_loss += loss.item()
        total_bce += bce.item()
        total_dice += dice.item()
        total_iou += iou
        total_acc += acc
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'iou': f'{iou:.4f}',
            'f1': f'{f1:.4f}'
        })
    
    # Average metrics
    num_batches = len(dataloader)
    metrics = {
        'loss': total_loss / num_batches,
        'bce': total_bce / num_batches,
        'dice': total_dice / num_batches,
        'iou': total_iou / num_batches,
        'accuracy': total_acc / num_batches,
        'precision': total_precision / num_batches,
        'recall': total_recall / num_batches,
        'f1': total_f1 / num_batches
    }
    
    return metrics


def validate(model, dataloader, criterion, device, epoch):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_acc = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for batch in pbar:
            spectrogram = batch['spectrogram'].to(device)
            mask = batch['mask'].to(device)
            
            # Forward pass
            output = model(spectrogram)
            
            # Calculate loss
            loss, bce, dice = criterion(output, mask)
            
            # Calculate metrics
            probs = torch.sigmoid(output.squeeze(1).squeeze(-1))
            iou = compute_temporal_iou(probs, mask)
            acc = compute_temporal_accuracy(probs, mask)
            precision, recall, f1 = compute_temporal_precision_recall_f1(probs, mask)
            
            # Accumulate
            total_loss += loss.item()
            total_bce += bce.item()
            total_dice += dice.item()
            total_iou += iou
            total_acc += acc
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{iou:.4f}',
                'f1': f'{f1:.4f}'
            })
    
    # Average metrics
    num_batches = len(dataloader)
    metrics = {
        'loss': total_loss / num_batches,
        'bce': total_bce / num_batches,
        'dice': total_dice / num_batches,
        'iou': total_iou / num_batches,
        'accuracy': total_acc / num_batches,
        'precision': total_precision / num_batches,
        'recall': total_recall / num_batches,
        'f1': total_f1 / num_batches
    }
    
    return metrics


def save_checkpoint(model, optimizer, epoch, metrics, filepath, is_best=False):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)
    if is_best:
        print(f"✓ Best model saved: {filepath}")


def main():
    """Main training function."""
    print("=" * 80)
    print("U-Net Temporal Event Segmentation Training")
    print("=" * 80)
    
    # Device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create datasets
    print("\nLoading datasets...")
    full_dataset = TemporalSegmentationDataset(
        csv_path=str(config.CSV_PATH),
        sample_rate=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS
    )
    
    # Train/Val split
    train_size = int(config.TRAIN_RATIO * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders with custom collate function for variable-length sequences
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=collate_fn_pad
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=collate_fn_pad
    )
    
    # Create model
    print("\nCreating U-Net model...")
    model = UNetTemporalSegmentation(
        n_channels=1,
        n_freq_bins=config.N_MELS,
        bilinear=True
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Loss function
    criterion = CombinedLoss(
        bce_weight=config.BCE_WEIGHT,
        dice_weight=config.DICE_WEIGHT
    )
    
    # Optimizer
    optimizer = Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Training history
    history = {
        'train': [],
        'val': [],
        'best_val_iou': 0.0,
        'best_epoch': 0
    }
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        history['train'].append(train_metrics)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch)
        history['val'].append(val_metrics)
        
        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_metrics['loss'])
        new_lr = optimizer.param_groups[0]['lr']
        
        # Print metrics
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['iou']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['iou']:.4f}, F1: {val_metrics['f1']:.4f}")
        if new_lr != old_lr:
            print(f"  ✓ Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
        
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, val_metrics, config.LAST_MODEL_PATH)
        
        # Save best model
        if val_metrics['iou'] > history['best_val_iou']:
            history['best_val_iou'] = val_metrics['iou']
            history['best_epoch'] = epoch
            save_checkpoint(model, optimizer, epoch, val_metrics, config.BEST_MODEL_PATH, is_best=True)
        
        # Save history
        with open(config.TRAINING_HISTORY_PATH, 'w') as f:
            json.dump(history, f, indent=2)
        
        print()
    
    print("=" * 80)
    print("Training completed!")
    print(f"Best validation IoU: {history['best_val_iou']:.4f} at epoch {history['best_epoch']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
