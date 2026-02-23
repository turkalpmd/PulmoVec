#!/usr/bin/env python3
"""
Extract probability outputs from all 3 trained ensemble models and save to CSV.

This script:
1. Loads all 3 trained models (best checkpoints)
2. Extracts probability outputs for each sample
3. Combines with demographic features (age, gender) and ground truth labels
4. Saves to separate CSV files for train and validation sets

Output CSV columns:
- patient_number, age, gender (demographic features)
- Model1_Normalpp, Model1_Cracklespp, Model1_Rhonchipp (3 probabilities)
- Model2_Normalpp, Model2_Abnormalpp (2 probabilities)
- Model3_Normalpp, Model3_Pneumoniapp, Model3_Bronchiolitispp (3 probabilities)
- disease, event_type (ground truth)
- model1_label, model2_label, model3_label (ground truth labels)

Usage:
    python scripts/extract_ensemble_probabilities.py
"""

import sys
import os
from pathlib import Path

# Make Projects/src/ importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from dotenv import load_dotenv

# Import config modules
import config
import config_model1
import config_model2
import config_model3
from models import HeARClassifier
from sprsound_dataset import stratified_train_val_split, SPRSoundDatasetFromDF

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output paths
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
TRAIN_CSV_PATH = DATA_DIR / "ensemble_probabilities_train.csv"
VAL_CSV_PATH = DATA_DIR / "ensemble_probabilities_val.csv"


def load_trained_model(config_module):
    """Load a trained model from checkpoint."""
    model = HeARClassifier(num_classes=config_module.NUM_CLASSES).to(DEVICE)
    
    checkpoint_path = config_module.BEST_MODEL_PATH
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logging.info(f"✓ Loaded {config_module.MODEL_NAME} from {checkpoint_path}")
    
    return model


def extract_probabilities(models, dataloader, df_source):
    """
    Extract probability outputs from all 3 models.
    
    Args:
        models: Tuple of (model1, model2, model3)
        dataloader: DataLoader for the dataset
        df_source: Source DataFrame with metadata (must match dataset order)
        
    Returns:
        results_df: DataFrame with probabilities and metadata
    """
    model1, model2, model3 = models
    
    all_probs1 = []
    all_probs2 = []
    all_probs3 = []
    
    logging.info("Extracting probabilities from all models...")
    
    with torch.no_grad():
        for batch_idx, (audio, labels, metadata) in enumerate(tqdm(dataloader, desc="Extracting")):
            audio = audio.to(DEVICE)
            
            # Get probability outputs from each model
            logits1 = model1(audio)
            logits2 = model2(audio)
            logits3 = model3(audio)
            
            probs1 = F.softmax(logits1, dim=1).cpu().numpy()  # [batch, 3]
            probs2 = F.softmax(logits2, dim=1).cpu().numpy()  # [batch, 2]
            probs3 = F.softmax(logits3, dim=1).cpu().numpy()  # [batch, 3]
            
            all_probs1.append(probs1)
            all_probs2.append(probs2)
            all_probs3.append(probs3)
    
    # Concatenate all probabilities
    probs1_array = np.vstack(all_probs1)  # [N, 3]
    probs2_array = np.vstack(all_probs2)  # [N, 2]
    probs3_array = np.vstack(all_probs3)  # [N, 3]
    
    # Create DataFrame with probabilities
    # df_source should be in the same order as dataloader (no shuffle)
    results_df = df_source.copy().reset_index(drop=True)
    
    # Add Model 1 probabilities
    results_df['Model1_Normalpp'] = probs1_array[:, 0]
    results_df['Model1_Cracklespp'] = probs1_array[:, 1]
    results_df['Model1_Rhonchipp'] = probs1_array[:, 2]
    
    # Add Model 2 probabilities
    results_df['Model2_Normalpp'] = probs2_array[:, 0]
    results_df['Model2_Abnormalpp'] = probs2_array[:, 1]
    
    # Add Model 3 probabilities
    # Note: Model 3 classes are: ["Pneumonia", "Bronchitis-Asthma-Bronchiolitis", "Normal/Other"]
    results_df['Model3_Pneumoniapp'] = probs3_array[:, 0]
    results_df['Model3_Bronchiolitispp'] = probs3_array[:, 1]
    results_df['Model3_Normalpp'] = probs3_array[:, 2]
    
    # Verify probabilities sum to ~1.0
    model1_sum = (results_df['Model1_Normalpp'] + 
                  results_df['Model1_Cracklespp'] + 
                  results_df['Model1_Rhonchipp'])
    model2_sum = (results_df['Model2_Normalpp'] + 
                  results_df['Model2_Abnormalpp'])
    model3_sum = (results_df['Model3_Pneumoniapp'] + 
                  results_df['Model3_Bronchiolitispp'] + 
                  results_df['Model3_Normalpp'])
    
    logging.info(f"Model 1 probability sum: min={model1_sum.min():.6f}, max={model1_sum.max():.6f}, mean={model1_sum.mean():.6f}")
    logging.info(f"Model 2 probability sum: min={model2_sum.min():.6f}, max={model2_sum.max():.6f}, mean={model2_sum.mean():.6f}")
    logging.info(f"Model 3 probability sum: min={model3_sum.min():.6f}, max={model3_sum.max():.6f}, mean={model3_sum.mean():.6f}")
    
    return results_df


def main():
    logging.info("=" * 80)
    logging.info("ENSEMBLE PROBABILITY EXTRACTION")
    logging.info("=" * 80)
    
    # Load environment for HuggingFace token
    load_dotenv(config.ENV_FILE)
    
    # Load all 3 trained models
    logging.info("\nLoading trained models...")
    model1 = load_trained_model(config_model1)
    model2 = load_trained_model(config_model2)
    model3 = load_trained_model(config_model3)
    
    models = (model1, model2, model3)
    
    # Load and split data (use same split as training)
    logging.info("\nLoading and splitting data...")
    train_df, val_df = stratified_train_val_split(
        csv_path=config_model3.CSV_PATH,  # Now points to data/SPRSound_Event_Level_Ensemble_Dataset.csv
        label_column=config_model3.LABEL_COLUMN,  # Use model3's label for consistency
        train_ratio=0.8,
        random_seed=42
    )
    
    logging.info(f"Training samples: {len(train_df)}")
    logging.info(f"Validation samples: {len(val_df)}")
    
    # Create datasets
    train_dataset = SPRSoundDatasetFromDF(
        train_df,
        label_column=config_model3.LABEL_COLUMN
    )
    val_dataset = SPRSoundDatasetFromDF(
        val_df,
        label_column=config_model3.LABEL_COLUMN
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    # Extract probabilities for training set
    logging.info("\n" + "-" * 80)
    logging.info("Extracting Training Set Probabilities")
    logging.info("-" * 80)
    train_results = extract_probabilities(models, train_loader, train_df)
    
    # Extract probabilities for validation set
    logging.info("\n" + "-" * 80)
    logging.info("Extracting Validation Set Probabilities")
    logging.info("-" * 80)
    val_results = extract_probabilities(models, val_loader, val_df)
    
    # Select and reorder columns for output
    output_columns = [
        # Demographics
        'patient_number', 'age', 'gender',
        # Model 1 probabilities
        'Model1_Normalpp', 'Model1_Cracklespp', 'Model1_Rhonchipp',
        # Model 2 probabilities
        'Model2_Normalpp', 'Model2_Abnormalpp',
        # Model 3 probabilities
        'Model3_Normalpp', 'Model3_Pneumoniapp', 'Model3_Bronchiolitispp',
        # Ground truth
        'disease', 'event_type',
        'model1_label', 'model2_label', 'model3_label'
    ]
    
    # Ensure all columns exist
    missing_cols = [col for col in output_columns if col not in train_results.columns]
    if missing_cols:
        logging.warning(f"Missing columns: {missing_cols}")
        output_columns = [col for col in output_columns if col in train_results.columns]
    
    train_output = train_results[output_columns].copy()
    val_output = val_results[output_columns].copy()
    
    # Save to CSV
    logging.info(f"\nSaving training probabilities to: {TRAIN_CSV_PATH}")
    train_output.to_csv(TRAIN_CSV_PATH, index=False)
    logging.info(f"✓ Saved {len(train_output)} training samples")
    
    logging.info(f"\nSaving validation probabilities to: {VAL_CSV_PATH}")
    val_output.to_csv(VAL_CSV_PATH, index=False)
    logging.info(f"✓ Saved {len(val_output)} validation samples")
    
    # Print summary
    logging.info("\n" + "=" * 80)
    logging.info("EXTRACTION COMPLETE!")
    logging.info("=" * 80)
    logging.info(f"\nTraining CSV: {TRAIN_CSV_PATH}")
    logging.info(f"  Rows: {len(train_output)}")
    logging.info(f"  Columns: {len(train_output.columns)}")
    logging.info(f"\nValidation CSV: {VAL_CSV_PATH}")
    logging.info(f"  Rows: {len(val_output)}")
    logging.info(f"  Columns: {len(val_output.columns)}")
    logging.info("\n✓ Ready for meta-model training!")


if __name__ == "__main__":
    main()
