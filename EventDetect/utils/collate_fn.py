"""
Custom collate function for variable-length spectrograms.
Pads spectrograms and masks to the same length within a batch.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict


def collate_fn_pad(batch: List[Dict]) -> Dict:
    """
    Custom collate function that pads variable-length spectrograms and masks.
    
    Args:
        batch: List of dictionaries with 'spectrogram', 'mask', 'file_path', 'duration_sec'
    
    Returns:
        Dictionary with batched, padded tensors
    """
    # Extract all spectrograms and masks
    spectrograms = [item['spectrogram'] for item in batch]
    masks = [item['mask'] for item in batch]
    file_paths = [item['file_path'] for item in batch]
    durations = [item['duration_sec'] for item in batch]
    
    # Get max time_steps in batch
    max_time_steps = max(spec.shape[1] for spec in spectrograms)
    
    # Pad spectrograms and masks
    padded_spectrograms = []
    padded_masks = []
    
    for spec, mask in zip(spectrograms, masks):
        # spec: (1, time_steps, freq_bins)
        # mask: (time_steps,)
        
        current_time_steps = spec.shape[1]
        pad_length = max_time_steps - current_time_steps
        
        if pad_length > 0:
            # Pad spectrogram: pad last dim (time_steps dimension)
            # Padding format: (pad_left, pad_right, pad_top, pad_bottom)
            # For 3D tensor (1, time, freq): pad on time dimension (dim 1)
            spec_padded = F.pad(spec, (0, 0, 0, pad_length), mode='constant', value=0.0)
            # Pad mask: pad on last dimension
            mask_padded = F.pad(mask, (0, pad_length), mode='constant', value=0.0)
        else:
            spec_padded = spec
            mask_padded = mask
        
        padded_spectrograms.append(spec_padded)
        padded_masks.append(mask_padded)
    
    # Stack into batches
    batch_spectrograms = torch.stack(padded_spectrograms, dim=0)  # (batch, 1, max_time_steps, freq_bins)
    batch_masks = torch.stack(padded_masks, dim=0)  # (batch, max_time_steps)
    
    return {
        'spectrogram': batch_spectrograms,
        'mask': batch_masks,
        'file_path': file_paths,
        'duration_sec': durations
    }
