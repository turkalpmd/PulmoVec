"""
Prepare ensemble dataset with 3 new label columns for multi-model training.

Adds the following columns to SPRSound_Event_Level_Dataset_CLEAN.csv:
    - model1_label: Event type classification (3 classes)
    - model2_label: Binary abnormality detection (2 classes)
    - model3_label: Disease group classification (3 classes)

Output: SPRSound_Event_Level_Ensemble_Dataset.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Paths
PROJECT_ROOT = Path(__file__).parent.absolute()
INPUT_CSV = PROJECT_ROOT / "SPRSound_Event_Level_Dataset_CLEAN.csv"
OUTPUT_CSV = PROJECT_ROOT / "SPRSound_Event_Level_Ensemble_Dataset.csv"


def create_model1_labels(df):
    """
    Model 1: Event Type Classification (3 classes)
    
    Class 0: Normal
    Class 1: Crackles (Fine Crackle, Coarse Crackle, Wheeze+Crackle)
    Class 2: Wheeze/Rhonchi (Wheeze, Rhonchi)
    
    Exclude: Stridor, No Event (will be filtered out)
    """
    logging.info("Creating Model 1 labels (Event Type - 3 classes)...")
    
    def map_event_type(event_type):
        if event_type == "Normal":
            return 0
        elif event_type in ["Fine Crackle", "Coarse Crackle", "Wheeze+Crackle"]:
            return 1
        elif event_type in ["Wheeze", "Rhonchi"]:
            return 2
        else:
            # Stridor, No Event -> -1 (will be filtered)
            return -1
    
    df['model1_label'] = df['event_type'].apply(map_event_type)
    
    # Log distribution
    valid_mask = df['model1_label'] != -1
    logging.info(f"Model 1 Label Distribution:")
    logging.info(f"  Class 0 (Normal): {(df['model1_label'] == 0).sum()}")
    logging.info(f"  Class 1 (Crackles): {(df['model1_label'] == 1).sum()}")
    logging.info(f"  Class 2 (Wheeze/Rhonchi): {(df['model1_label'] == 2).sum()}")
    logging.info(f"  Excluded (Stridor/No Event): {(~valid_mask).sum()}")
    
    return df, valid_mask


def create_model2_labels(df):
    """
    Model 2: Binary Abnormality Detection (2 classes)
    
    Class 0: Normal event (event_type == "Normal")
    Class 1: Abnormal event (all other event types)
    """
    logging.info("Creating Model 2 labels (Binary Abnormal/Normal - 2 classes)...")
    
    def map_binary(event_type):
        if event_type == "Normal":
            return 0
        else:
            return 1
    
    df['model2_label'] = df['event_type'].apply(map_binary)
    
    # Log distribution
    logging.info(f"Model 2 Label Distribution:")
    logging.info(f"  Class 0 (Normal): {(df['model2_label'] == 0).sum()}")
    logging.info(f"  Class 1 (Abnormal): {(df['model2_label'] == 1).sum()}")
    
    return df


def create_model3_labels(df):
    """
    Model 3: Disease Group Classification (3 classes)
    
    Class 0: Pneumonia (severe + non-severe)
    Class 1: Bronchitis/Asthma/Bronchiolitis
    Class 2: Normal (all other diseases including their normal events)
    """
    logging.info("Creating Model 3 labels (Disease Groups - 3 classes)...")
    
    def map_disease(disease):
        if disease in ["Pneumonia (severe)", "Pneumonia (non-severe)"]:
            return 0
        elif disease in ["Bronchitis", "Asthma", "Bronchiolitis"]:
            return 1
        else:
            # All other diseases -> Normal/Other group
            return 2
    
    df['model3_label'] = df['disease'].apply(map_disease)
    
    # Log distribution
    logging.info(f"Model 3 Label Distribution:")
    logging.info(f"  Class 0 (Pneumonia): {(df['model3_label'] == 0).sum()}")
    logging.info(f"  Class 1 (Bronchitis/Asthma/Bronchiolitis): {(df['model3_label'] == 1).sum()}")
    logging.info(f"  Class 2 (Normal/Other): {(df['model3_label'] == 2).sum()}")
    
    return df


def main():
    logging.info("=" * 80)
    logging.info("ENSEMBLE LABEL PREPARATION")
    logging.info("=" * 80)
    
    # Load CSV
    logging.info(f"Loading CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    logging.info(f"Total events loaded: {len(df)}")
    
    # Ensure only valid WAV files
    df = df[df['wav_exists'] == 'yes'].copy()
    logging.info(f"Events with valid WAV files: {len(df)}")
    
    # Create Model 1 labels (this will also identify rows to exclude)
    df, valid_mask = create_model1_labels(df)
    
    # Create Model 2 labels
    df = create_model2_labels(df)
    
    # Create Model 3 labels
    df = create_model3_labels(df)
    
    # Filter out invalid rows (Stridor, No Event for Model 1)
    df_filtered = df[valid_mask].copy()
    logging.info(f"\nTotal events after filtering: {len(df_filtered)}")
    
    # Save to CSV
    logging.info(f"\nSaving ensemble dataset to: {OUTPUT_CSV}")
    df_filtered.to_csv(OUTPUT_CSV, index=False)
    
    # Summary
    logging.info("\n" + "=" * 80)
    logging.info("SUMMARY")
    logging.info("=" * 80)
    logging.info(f"Input file: {INPUT_CSV}")
    logging.info(f"Output file: {OUTPUT_CSV}")
    logging.info(f"Total events: {len(df_filtered)}")
    logging.info(f"New columns added: model1_label, model2_label, model3_label")
    
    # Verify label consistency
    logging.info("\nLabel Statistics:")
    for col in ['model1_label', 'model2_label', 'model3_label']:
        logging.info(f"\n{col}:")
        logging.info(df_filtered[col].value_counts().sort_index())
    
    # Cross-tabulation for verification
    logging.info("\nCross-tabulation (Event Type vs Model1 Label):")
    logging.info(pd.crosstab(df_filtered['event_type'], df_filtered['model1_label']))
    
    logging.info("\nCross-tabulation (Disease vs Model3 Label):")
    logging.info(pd.crosstab(df_filtered['disease'], df_filtered['model3_label']))
    
    logging.info("\n✅ Ensemble dataset created successfully!")


if __name__ == "__main__":
    main()
