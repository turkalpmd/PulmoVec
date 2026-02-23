"""
Dataset for U-Net temporal segmentation training.

Converts audio files to spectrograms and creates temporal segmentation masks
from ground truth event annotations.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from typing import Tuple, Optional
import json


class TemporalSegmentationDataset(Dataset):
    """
    Dataset for temporal event segmentation.
    
    Input: Full audio file → STFT/Mel-Spectrogram
    Output: Temporal mask (time_steps x num_classes)
    
    Ground truth: Generated from JSON event annotations
    """
    
    def __init__(
        self,
        csv_path: str,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        transform=None
    ):
        """
        Args:
            csv_path: Path to CSV with event-level annotations
            sample_rate: Audio sample rate
            n_fft: FFT window size for STFT
            hop_length: Hop length for STFT
            n_mels: Number of mel filter banks
            transform: Optional data augmentation
        """
        self.df = pd.read_csv(csv_path)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.transform = transform
        
        # Group by audio file (wav_path) to get unique files
        self.file_groups = self.df.groupby('wav_path').groups
        
        # Filter files that exist
        self.valid_files = []
        for wav_path in self.file_groups.keys():
            if Path(wav_path).exists():
                self.valid_files.append(wav_path)
            else:
                print(f"Warning: File not found: {wav_path}")
        
        print(f"Loaded {len(self.valid_files)} unique audio files")
        print(f"Total events: {len(self.df)}")
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            spectrogram: (1, time_steps, freq_bins) - Input spectrogram
            mask: (time_steps,) - Binary ground truth temporal mask (0=Normal, 1=Event)
            file_path: str - Path to audio file
        """
        wav_path = self.valid_files[idx]
        
        # Load audio
        audio, sr = librosa.load(wav_path, sr=self.sample_rate, mono=True)
        duration_sec = len(audio) / sr
        
        # Compute Mel-Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        # Shape: (freq_bins, time_steps) -> (1, time_steps, freq_bins)
        mel_spec_db = mel_spec_db.T  # (time_steps, freq_bins)
        mel_spec_db = mel_spec_db[np.newaxis, :, :]  # (1, time_steps, freq_bins)
        
        # Get ground truth binary temporal mask from JSON
        json_path = self._get_json_path(wav_path)
        mask = self._create_binary_temporal_mask(json_path, duration_sec, mel_spec_db.shape[1])
        
        # Convert to tensors
        spectrogram = torch.FloatTensor(mel_spec_db)  # (1, time_steps, freq_bins)
        mask = torch.FloatTensor(mask)  # (time_steps,) - Binary: 0=Normal, 1=Event
        
        # Apply transforms if provided
        if self.transform:
            spectrogram = self.transform(spectrogram)
        
        return {
            'spectrogram': spectrogram,
            'mask': mask,
            'file_path': wav_path,
            'duration_sec': duration_sec
        }
    
    def _get_json_path(self, wav_path: str) -> Optional[str]:
        """Get corresponding JSON annotation file path."""
        wav_path = Path(wav_path)
        
        # Try different JSON path patterns
        # Pattern 1: Replace .wav with .json in same directory
        json_path = wav_path.with_suffix('.json')
        if json_path.exists():
            return str(json_path)
        
        # Pattern 2: Look in corresponding _json directory
        json_dir = wav_path.parent.parent / (wav_path.parent.name.replace('_wav', '_json'))
        json_path = json_dir / wav_path.name.replace('.wav', '.json')
        if json_path.exists():
            return str(json_path)
        
        # Pattern 3: Look in same directory with different naming
        json_path = wav_path.parent / wav_path.name.replace('.wav', '.json')
        if json_path.exists():
            return str(json_path)
        
        return None
    
    def _create_binary_temporal_mask(
        self,
        json_path: Optional[str],
        duration_sec: float,
        num_time_steps: int
    ) -> np.ndarray:
        """
        Create binary temporal segmentation mask from JSON annotations.
        
        Args:
            json_path: Path to JSON annotation file
            duration_sec: Audio duration in seconds
            num_time_steps: Number of time steps in spectrogram
        
        Returns:
            mask: (time_steps,) - Binary mask: 0=Normal, 1=Event
        """
        # Initialize mask: all Normal (0)
        mask = np.zeros(num_time_steps, dtype=np.float32)
        
        if json_path is None or not Path(json_path).exists():
            return mask
        
        # Load JSON annotations
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON {json_path}: {e}")
            return mask
        
        # Get events from JSON
        events = data.get('event_annotation', [])
        
        # Time step duration
        time_step_duration = duration_sec / num_time_steps
        
        # Map events to time steps
        for event in events:
            start_ms = float(event.get('start', 0))
            end_ms = float(event.get('end', 0))
            event_type = event.get('type', 'Normal')
            
            # Skip Normal events (they are already 0 in mask)
            if event_type == "Normal":
                continue
            
            # Convert ms to seconds
            start_sec = start_ms / 1000.0
            end_sec = end_ms / 1000.0
            
            # Convert to time step indices
            start_idx = int(start_sec / time_step_duration)
            end_idx = int(end_sec / time_step_duration)
            
            # Clamp to valid range
            start_idx = max(0, min(start_idx, num_time_steps - 1))
            end_idx = max(0, min(end_idx, num_time_steps - 1))
            
            # Set mask to 1 (Event) for this time range
            mask[start_idx:end_idx+1] = 1.0
        
        return mask
