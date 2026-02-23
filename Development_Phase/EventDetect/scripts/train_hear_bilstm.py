"""
Training script for HeAR + BiLSTM temporal event segmentation.
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
import argparse
import multiprocessing

# Set multiprocessing start method to 'spawn' for CUDA compatibility
if torch.cuda.is_available():
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, ignore
        pass

# Add EventDetect to path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import config_hear_bilstm as config
from models.hear_encoder import HeAREncoder, EmbeddingCache
from models.bilstm_event_detector import BiLSTMEventDetector
from data_processing.hear_embedding_dataset import HeAREmbeddingDataset
from utils.collate_seq import collate_fn_sequences
from utils.metrics_seq import compute_window_metrics
from utils.seed import set_seed


def compute_pos_weight(dataset, device):
    """Compute positive weight for class imbalance."""
    print("Computing positive weight from training set...")
    all_labels = []
    
    for i in range(len(dataset)):
        item = dataset[i]
        labels = item['labels'].numpy()
        all_labels.extend(labels.tolist())
    
    all_labels = np.array(all_labels)
    num_pos = np.sum(all_labels == 1)
    num_neg = np.sum(all_labels == 0)
    
    if num_pos == 0:
        pos_weight = 1.0
    else:
        pos_weight = num_neg / num_pos
    
    print(f"  Positive samples: {num_pos}")
    print(f"  Negative samples: {num_neg}")
    print(f"  Pos weight: {pos_weight:.4f}")
    
    return torch.tensor(pos_weight, device=device)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    # Accumulate metrics across batches
    total_auroc = 0.0
    total_auprc = 0.0
    total_f1 = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_accuracy = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        embeddings = batch['embeddings'].to(device)  # (batch, T, 512)
        labels = batch['labels'].to(device)  # (batch, T)
        lengths = batch['lengths'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(embeddings, lengths)  # (batch, T)
        
        # Calculate loss (only on valid positions)
        valid_mask = attention_mask.bool()
        loss = criterion(logits[valid_mask], labels[valid_mask])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute metrics for this batch
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            batch_metrics = compute_window_metrics(
                probs.detach().cpu(),
                labels.detach().cpu(),
                attention_mask.detach().cpu(),
                threshold=config.EVAL_THRESHOLD
            )
            
            # Accumulate metrics
            total_auroc += batch_metrics['auroc']
            total_auprc += batch_metrics['auprc']
            total_f1 += batch_metrics['f1']
            total_precision += batch_metrics['precision']
            total_recall += batch_metrics['recall']
            total_accuracy += batch_metrics['accuracy']
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'f1': f'{batch_metrics["f1"]:.4f}'
        })
        total_loss += loss.item()
    
    # Average metrics across batches
    num_batches = len(dataloader)
    metrics = {
        'loss': total_loss / num_batches,
        'auroc': total_auroc / num_batches,
        'auprc': total_auprc / num_batches,
        'f1': total_f1 / num_batches,
        'precision': total_precision / num_batches,
        'recall': total_recall / num_batches,
        'accuracy': total_accuracy / num_batches
    }
    
    return metrics


def validate(model, dataloader, criterion, device, epoch):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    
    # Accumulate metrics across batches
    total_auroc = 0.0
    total_auprc = 0.0
    total_f1 = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_accuracy = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for batch in pbar:
            embeddings = batch['embeddings'].to(device)
            labels = batch['labels'].to(device)
            lengths = batch['lengths'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            logits = model(embeddings, lengths)
            
            # Calculate loss
            valid_mask = attention_mask.bool()
            loss = criterion(logits[valid_mask], labels[valid_mask])
            
            # Compute metrics for this batch
            probs = torch.sigmoid(logits)
            batch_metrics = compute_window_metrics(
                probs.cpu(),
                labels.cpu(),
                attention_mask.cpu(),
                threshold=config.EVAL_THRESHOLD
            )
            
            # Accumulate metrics
            total_auroc += batch_metrics['auroc']
            total_auprc += batch_metrics['auprc']
            total_f1 += batch_metrics['f1']
            total_precision += batch_metrics['precision']
            total_recall += batch_metrics['recall']
            total_accuracy += batch_metrics['accuracy']
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'f1': f'{batch_metrics["f1"]:.4f}'
            })
            total_loss += loss.item()
    
    # Average metrics across batches
    num_batches = len(dataloader)
    metrics = {
        'loss': total_loss / num_batches,
        'auroc': total_auroc / num_batches,
        'auprc': total_auprc / num_batches,
        'f1': total_f1 / num_batches,
        'precision': total_precision / num_batches,
        'recall': total_recall / num_batches,
        'accuracy': total_accuracy / num_batches
    }
    
    return metrics


def save_checkpoint(model, optimizer, epoch, metrics, filepath, is_best=False):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': {
            'hidden_dim': config.BILSTM_HIDDEN_DIM,
            'num_layers': config.BILSTM_NUM_LAYERS,
            'dropout': config.BILSTM_DROPOUT,
            'input_dim': config.HEAR_EMBEDDING_DIM
        }
    }
    torch.save(checkpoint, filepath)
    if is_best:
        print(f"✓ Best model saved: {filepath}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train HeAR + BiLSTM event detector')
    parser.add_argument('--recompute-embeddings', action='store_true',
                        help='Recompute embeddings even if cached')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("HeAR + BiLSTM Temporal Event Segmentation Training")
    print("=" * 80)
    
    # Set seed
    set_seed(config.RANDOM_SEED)
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create HeAR encoder
    print("\nLoading HeAR encoder...")
    hear_encoder = HeAREncoder(
        model_name=config.HEAR_MODEL_NAME,
        frozen=config.HEAR_ENCODER_FROZEN,
        fine_tune_last_n_layers=config.HEAR_FINE_TUNE_LAST_N_LAYERS,
        device=device
    )
    
    # Create embedding cache
    embedding_cache = EmbeddingCache(config.EMBEDDING_CACHE_DIR)
    
    # Create datasets
    print("\nLoading datasets...")
    full_dataset = HeAREmbeddingDataset(
        csv_path=str(config.CSV_PATH),
        hear_encoder=hear_encoder,
        embedding_cache=embedding_cache,
        window_sec=config.WINDOW_SEC,
        hop_sec=config.HOP_SEC,
        overlap_ratio_threshold=config.OVERLAP_RATIO_THRESHOLD,
        sample_rate=config.SAMPLE_RATE,
        recompute_embeddings=args.recompute_embeddings
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
    
    # Compute positive weight
    if config.POS_WEIGHT_MODE == "auto":
        pos_weight = compute_pos_weight(train_dataset, device)
    elif config.POS_WEIGHT_MODE == "manual" and config.POS_WEIGHT_MANUAL is not None:
        pos_weight = torch.tensor(config.POS_WEIGHT_MANUAL, device=device)
    else:
        pos_weight = torch.tensor(1.0, device=device)
    
    # Create dataloaders
    # Note: When using CUDA with HeAR encoder in dataset, we need to set num_workers=0
    # because CUDA cannot be re-initialized in forked subprocesses.
    # Since embeddings are cached, this shouldn't significantly impact performance.
    if device.type == 'cuda':
        num_workers = 0  # CUDA + multiprocessing requires spawn method, but simpler to use 0 workers
        print("Note: Using num_workers=0 for CUDA compatibility (embeddings are cached)")
    else:
        num_workers = config.NUM_WORKERS
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # Not needed with num_workers=0
        collate_fn=collate_fn_sequences
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,  # Not needed with num_workers=0
        collate_fn=collate_fn_sequences
    )
    
    # Create model
    print("\nCreating BiLSTM model...")
    model = BiLSTMEventDetector(
        input_dim=config.HEAR_EMBEDDING_DIM,
        hidden_dim=config.BILSTM_HIDDEN_DIM,
        num_layers=config.BILSTM_NUM_LAYERS,
        dropout=config.BILSTM_DROPOUT
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer (only train BiLSTM if encoder is frozen)
    if config.HEAR_ENCODER_FROZEN:
        optimizer = Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        # Train both encoder and BiLSTM
        optimizer = Adam(
            list(model.parameters()) + list(hear_encoder.parameters()),
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
        'best_val_metric': 0.0,
        'best_epoch': 0,
        'best_by': config.BEST_BY
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
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"AUROC: {train_metrics['auroc']:.4f}, "
              f"AUPRC: {train_metrics['auprc']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
              f"AUROC: {val_metrics['auroc']:.4f}, "
              f"AUPRC: {val_metrics['auprc']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}")
        if new_lr != old_lr:
            print(f"  ✓ Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
        
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, val_metrics, config.LAST_MODEL_PATH)
        
        # Save best model
        best_metric_key = config.BEST_BY  # 'val_auprc' or 'val_f1'
        current_metric = val_metrics.get(best_metric_key, 0.0)
        
        if current_metric > history['best_val_metric']:
            history['best_val_metric'] = current_metric
            history['best_epoch'] = epoch
            save_checkpoint(model, optimizer, epoch, val_metrics, config.BEST_MODEL_PATH, is_best=True)
        
        # Save history
        with open(config.TRAINING_HISTORY_PATH, 'w') as f:
            json.dump(history, f, indent=2)
        
        print()
    
    print("=" * 80)
    print("Training completed!")
    print(f"Best validation {config.BEST_BY}: {history['best_val_metric']:.4f} at epoch {history['best_epoch']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
