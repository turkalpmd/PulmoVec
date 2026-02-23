"""
Custom collate function for variable-length embedding sequences.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict


def collate_fn_sequences(batch: List[Dict]) -> Dict:
    """
    Custom collate function for variable-length embedding sequences.
    
    Pads sequences to the same length within a batch and creates
    attention masks and length tensors for pack_padded_sequence.
    
    Args:
        batch: List of dictionaries with:
            - 'embeddings': (T, 512) tensor
            - 'labels': (T,) tensor
            - 'length': int
            - 'file_path': str
            - (optional) 'duration_sec': float
    
    Returns:
        Dictionary with batched tensors:
            - 'embeddings': (batch, max_T, 512) - Padded embeddings
            - 'labels': (batch, max_T) - Padded labels
            - 'lengths': (batch,) - Actual sequence lengths
            - 'attention_mask': (batch, max_T) - Attention mask (1 for valid, 0 for pad)
            - 'file_paths': List[str]
            - (optional) 'durations': List[float]
    """
    # Extract components
    embeddings_list = [item['embeddings'] for item in batch]
    labels_list = [item['labels'] for item in batch]
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long)
    file_paths = [item['file_path'] for item in batch]
    
    # Get max sequence length
    max_length = int(lengths.max().item())
    
    # Pad embeddings and labels
    padded_embeddings = []
    padded_labels = []
    attention_masks = []
    
    for emb, label, length in zip(embeddings_list, labels_list, lengths):
        # emb: (T, 512), label: (T,)
        current_length = emb.shape[0]
        
        # Pad embeddings: pad on time dimension (dim 0)
        if current_length < max_length:
            pad_length = max_length - current_length
            # Pad: (pad_left, pad_right) for 1D, (pad_left, pad_right, pad_top, pad_bottom) for 2D
            emb_padded = F.pad(emb, (0, 0, 0, pad_length), mode='constant', value=0.0)
            label_padded = F.pad(label, (0, pad_length), mode='constant', value=0.0)
            
            # Attention mask: 1 for valid, 0 for padding
            attention_mask = torch.ones(max_length, dtype=torch.float)
            attention_mask[current_length:] = 0.0
        else:
            emb_padded = emb
            label_padded = label
            attention_mask = torch.ones(max_length, dtype=torch.float)
        
        padded_embeddings.append(emb_padded)
        padded_labels.append(label_padded)
        attention_masks.append(attention_mask)
    
    # Stack into batches
    batch_embeddings = torch.stack(padded_embeddings, dim=0)  # (batch, max_T, 512)
    batch_labels = torch.stack(padded_labels, dim=0)  # (batch, max_T)
    batch_attention_mask = torch.stack(attention_masks, dim=0)  # (batch, max_T)
    
    # Prepare output
    output = {
        'embeddings': batch_embeddings,
        'labels': batch_labels,
        'lengths': lengths,
        'attention_mask': batch_attention_mask,
        'file_paths': file_paths
    }
    
    # Add durations if available
    if 'duration_sec' in batch[0]:
        durations = [item['duration_sec'] for item in batch]
        output['durations'] = durations
    
    return output
