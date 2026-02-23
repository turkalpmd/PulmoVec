"""
BiLSTM model for frame-level binary event detection.

Takes HeAR embeddings as input and outputs binary event probabilities
for each time window.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class BiLSTMEventDetector(nn.Module):
    """
    BiLSTM sequence tagger for binary event detection.
    
    Architecture:
        Input: (batch, T, 512) - HeAR embeddings
        BiLSTM: hidden_dim, num_layers, dropout
        Output: (batch, T) - Binary logits per window
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        """
        Initialize BiLSTM event detector.
        
        Args:
            input_dim: Input embedding dimension (default: 512 for HeAR)
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate (only active if num_layers > 1)
            bidirectional: If True, use bidirectional LSTM
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout if num_layers > 1 else 0.0
        self.bidirectional = bidirectional
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=self.dropout,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output projection
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 (common LSTM initialization)
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1)
        
        # Initialize FC layer
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)
    
    def forward(
        self,
        embeddings: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            embeddings: (batch, T, 512) - Input embeddings
            lengths: (batch,) - Actual sequence lengths (for pack_padded_sequence)
        
        Returns:
            logits: (batch, T) - Binary logits for each window
        """
        batch_size, seq_len, _ = embeddings.shape
        
        # Pack sequences if lengths provided
        if lengths is not None:
            # Sort by length (descending) for pack_padded_sequence
            lengths = lengths.cpu()
            sorted_lengths, sorted_indices = torch.sort(lengths, descending=True)
            sorted_embeddings = embeddings[sorted_indices]
            
            # Pack padded sequence
            packed_embeddings = nn.utils.rnn.pack_padded_sequence(
                sorted_embeddings,
                sorted_lengths,
                batch_first=True,
                enforce_sorted=True
            )
            
            # LSTM forward
            packed_output, (hidden, cell) = self.lstm(packed_embeddings)
            
            # Unpack
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=seq_len
            )
            
            # Unsort to original order
            _, unsort_indices = torch.sort(sorted_indices)
            output = output[unsort_indices]
        else:
            # Standard forward without packing
            output, (hidden, cell) = self.lstm(embeddings)
        
        # Project to binary logits
        logits = self.fc(output).squeeze(-1)  # (batch, T)
        
        return logits
    
    def predict_proba(
        self,
        embeddings: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get probability predictions.
        
        Args:
            embeddings: (batch, T, 512) - Input embeddings
            lengths: (batch,) - Actual sequence lengths
        
        Returns:
            probs: (batch, T) - Event probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(embeddings, lengths)
            probs = torch.sigmoid(logits)
        return probs
