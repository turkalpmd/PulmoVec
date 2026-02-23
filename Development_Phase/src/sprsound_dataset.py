"""
SPRSound PyTorch Dataset for event-level respiratory sound classification.
Loads audio events from CSV, applies temporal overlap, and prepares clips for HeAR.
"""

import pandas as pd
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import config


class SPRSoundDataset(Dataset):
    """
    PyTorch Dataset for SPRSound event-level data.
    
    Each sample corresponds to one event from the CSV file:
    - Extracts the event segment from the WAV file
    - Applies ±10% temporal overlap around the event
    - Resamples to 16kHz mono
    - Pads or trims to exactly 2 seconds (32,000 samples)
    """
    
    def __init__(
        self,
        csv_path: str = None,
        dataset_filter: Optional[str] = None,
        transform=None
    ):
        """
        Args:
            csv_path: Path to SPRSound_Event_Level_Dataset_CLEAN.csv
            dataset_filter: Optional filter for 'dataset' column (e.g., 'Classification-Train')
            transform: Optional audio transformation function
        """
        if csv_path is None:
            csv_path = config.CSV_PATH
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} events from {csv_path}")
        
        # Filter by dataset if specified
        if dataset_filter:
            self.df = self.df[self.df['dataset'] == dataset_filter].reset_index(drop=True)
            print(f"Filtered to {len(self.df)} events from dataset: {dataset_filter}")
        
        # Only keep rows where WAV exists
        self.df = self.df[self.df['wav_exists'] == 'yes'].reset_index(drop=True)
        print(f"Using {len(self.df)} events with existing WAV files")
        
        # Create label mapping
        self.class_names = config.CLASS_NAMES
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for idx, name in enumerate(self.class_names)}
        
        # Convert event_type to label indices
        self.df['label'] = self.df['event_type'].map(self.class_to_idx)
        
        # Check for any missing labels
        missing_labels = self.df[self.df['label'].isna()]
        if len(missing_labels) > 0:
            print(f"Warning: {len(missing_labels)} events have unknown labels:")
            print(missing_labels['event_type'].value_counts())
            self.df = self.df.dropna(subset=['label']).reset_index(drop=True)
        
        self.transform = transform
        
        # Print class distribution
        print("\nClass distribution:")
        for class_name in self.class_names:
            count = len(self.df[self.df['event_type'] == class_name])
            percentage = 100 * count / len(self.df)
            print(f"  {class_name}: {count} ({percentage:.2f}%)")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Returns:
            audio_tensor: (32000,) tensor of audio samples
            label: Integer class label
            metadata: Dictionary with event information
        """
        row = self.df.iloc[idx]
        
        # Get event information
        wav_path = Path(row['wav_path'])
        event_start_ms = float(row['event_start_ms'])
        event_end_ms = float(row['event_end_ms'])
        event_duration_ms = event_end_ms - event_start_ms
        label = int(row['label'])
        
        # Apply ±10% temporal overlap
        overlap_ms = event_duration_ms * (config.OVERLAP_PERCENT / 100.0)
        clip_start_ms = max(0, event_start_ms - overlap_ms)
        clip_end_ms = event_end_ms + overlap_ms
        
        # Load audio file
        try:
            audio, sr = librosa.load(wav_path, sr=None, mono=False)
            
            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = librosa.to_mono(audio)
            
            # Resample to 16kHz if needed
            if sr != config.SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=config.SAMPLE_RATE)
                sr = config.SAMPLE_RATE
            
            # Convert milliseconds to sample indices
            clip_start_sample = int(clip_start_ms * sr / 1000)
            clip_end_sample = int(clip_end_ms * sr / 1000)
            
            # Ensure we don't go beyond audio length
            clip_end_sample = min(clip_end_sample, len(audio))
            clip_start_sample = min(clip_start_sample, len(audio))
            
            # Extract event clip
            event_clip = audio[clip_start_sample:clip_end_sample]
            
            # Pad or trim to exactly CLIP_LENGTH (32000 samples = 2 seconds)
            if len(event_clip) < config.CLIP_LENGTH:
                # Pad with zeros
                pad_length = config.CLIP_LENGTH - len(event_clip)
                event_clip = np.pad(event_clip, (0, pad_length), mode='constant')
            elif len(event_clip) > config.CLIP_LENGTH:
                # Trim from center to preserve temporal context
                excess = len(event_clip) - config.CLIP_LENGTH
                start_trim = excess // 2
                event_clip = event_clip[start_trim:start_trim + config.CLIP_LENGTH]
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(event_clip).float()
            
            # Apply transforms if any
            if self.transform:
                audio_tensor = self.transform(audio_tensor)
            
            # Metadata
            metadata = {
                'filename': row['filename'],
                'patient_number': row['patient_number'],
                'event_type': row['event_type'],
                'event_start_ms': event_start_ms,
                'event_end_ms': event_end_ms,
                'event_duration_ms': event_duration_ms,
                'dataset': row['dataset']
            }
            
            return audio_tensor, label, metadata
            
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            # Return a zero tensor as fallback
            audio_tensor = torch.zeros(config.CLIP_LENGTH)
            metadata = {'error': str(e)}
            return audio_tensor, label, metadata
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced dataset.
        Uses inverse frequency weighting.
        """
        class_counts = []
        for class_name in self.class_names:
            count = len(self.df[self.df['event_type'] == class_name])
            class_counts.append(count)
        
        class_counts = np.array(class_counts, dtype=np.float32)
        total = class_counts.sum()
        
        # Inverse frequency weighting
        weights = total / (len(class_counts) * class_counts)
        
        # Normalize weights
        weights = weights / weights.sum() * len(class_counts)
        
        print("\nClass weights:")
        for class_name, weight in zip(self.class_names, weights):
            print(f"  {class_name}: {weight:.2f}")
        
        return torch.from_numpy(weights).float()


def stratified_train_val_split(
    csv_path: str = None,
    label_column: str = 'label',
    train_ratio: float = 0.8,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/val split by label column.
    Ensures class distribution is maintained in both splits.
    
    Args:
        csv_path: Path to CSV file
        label_column: Column name to use for stratification (default 'label')
        train_ratio: Ratio of training data (default 0.8)
        random_seed: Random seed for reproducibility
    
    Returns:
        train_df: Training DataFrame
        val_df: Validation DataFrame
    """
    if csv_path is None:
        csv_path = config.CSV_PATH
    
    df = pd.read_csv(csv_path)
    df = df[df['wav_exists'] == 'yes'].reset_index(drop=True)
    
    # Filter out invalid labels if needed (e.g., -1 for excluded events)
    if label_column in df.columns:
        df = df[df[label_column] >= 0].reset_index(drop=True)
    
    # Stratified split by label column
    from sklearn.model_selection import train_test_split
    
    stratify_column = df[label_column] if label_column in df.columns else df['event_type']
    
    train_df, val_df = train_test_split(
        df,
        train_size=train_ratio,
        stratify=stratify_column,
        random_state=random_seed
    )
    
    print(f"\nStratified split (by {label_column}):")
    print(f"  Training: {len(train_df)} events ({train_ratio*100:.0f}%)")
    print(f"  Validation: {len(val_df)} events ({(1-train_ratio)*100:.0f}%)")
    
    # Verify class distribution
    if label_column in df.columns:
        print(f"\nTraining set class distribution ({label_column}):")
        label_counts = train_df[label_column].value_counts().sort_index()
        for label, count in label_counts.items():
            percentage = 100 * count / len(train_df)
            print(f"  Class {label}: {count} ({percentage:.2f}%)")
        
        print(f"\nValidation set class distribution ({label_column}):")
        label_counts = val_df[label_column].value_counts().sort_index()
        for label, count in label_counts.items():
            percentage = 100 * count / len(val_df)
            print(f"  Class {label}: {count} ({percentage:.2f}%)")
    else:
        # Fallback to event_type if label column doesn't exist
        print("\nTraining set class distribution:")
        for class_name in config.CLASS_NAMES:
            count = len(train_df[train_df['event_type'] == class_name])
            percentage = 100 * count / len(train_df)
            print(f"  {class_name}: {count} ({percentage:.2f}%)")
        
        print("\nValidation set class distribution:")
        for class_name in config.CLASS_NAMES:
            count = len(val_df[val_df['event_type'] == class_name])
            percentage = 100 * count / len(val_df)
            print(f"  {class_name}: {count} ({percentage:.2f}%)")
    
    return train_df, val_df


class SPRSoundDatasetFromDF(SPRSoundDataset):
    """
    SPRSound Dataset that takes a DataFrame instead of loading from CSV.
    Useful for train/val splits.
    """
    
    def __init__(self, df: pd.DataFrame, label_column: str = 'label', transform=None):
        """
        Args:
            df: DataFrame with event annotations
            label_column: Column name containing labels (default 'label')
            transform: Optional audio transformation function
        """
        self.df = df.reset_index(drop=True)
        self.label_column = label_column
        
        # If using a pre-existing label column (e.g., model1_label, model2_label)
        if label_column in self.df.columns:
            # Use the label column directly
            self.df['label'] = self.df[label_column].astype(int)
            
            # Determine class names from unique labels
            unique_labels = sorted(self.df['label'].unique())
            self.class_names = [f"Class_{i}" for i in unique_labels]
            self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
            self.idx_to_class = {idx: name for idx, name in enumerate(self.class_names)}
            
            print(f"Using label column '{label_column}' with {len(unique_labels)} classes")
        else:
            # Fallback to event_type mapping (original behavior)
            self.class_names = config.CLASS_NAMES
            self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
            self.idx_to_class = {idx: name for idx, name in enumerate(self.class_names)}
            
            # Convert event_type to label indices
            self.df['label'] = self.df['event_type'].map(self.class_to_idx)
            
            # Check for any missing labels
            missing_labels = self.df[self.df['label'].isna()]
            if len(missing_labels) > 0:
                print(f"Warning: {len(missing_labels)} events have unknown labels")
                self.df = self.df.dropna(subset=['label']).reset_index(drop=True)
        
        self.transform = transform
        
        print(f"Created dataset with {len(self.df)} events")


if __name__ == "__main__":
    # Test the dataset
    print("Testing SPRSoundDataset...")
    
    # Test stratified split
    train_df, val_df = stratified_train_val_split()
    
    # Create datasets
    train_dataset = SPRSoundDatasetFromDF(train_df)
    val_dataset = SPRSoundDatasetFromDF(val_df)
    
    # Test loading a sample
    print("\nTesting sample loading...")
    audio, label, metadata = train_dataset[0]
    print(f"Audio shape: {audio.shape}")
    print(f"Label: {label} ({train_dataset.idx_to_class[label]})")
    print(f"Metadata: {metadata}")
    
    # Test class weights
    print("\nCalculating class weights...")
    weights = train_dataset.get_class_weights()
    print(f"Weights shape: {weights.shape}")
    
    print("\n✓ Dataset tests passed!")
