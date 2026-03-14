"""
HeAR-based models for respiratory sound classification.
Includes HeAR encoder integration and classification head.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
import sys
from pathlib import Path

# Add hear/python to path for audio_utils
# hear/ is in project root, not in src/
hear_python_path = Path(__file__).parent.parent / "hear" / "python"
sys.path.insert(0, str(hear_python_path))
from data_processing import audio_utils

import config


class HeARClassifier(nn.Module):
    """
    HeAR-based classifier for respiratory event classification.
    
    Architecture:
        1. HeAR PyTorch encoder (google/hear-pytorch) - generates 512D embeddings
        2. Classification head with dropout and hidden layer
    
    Training strategy:
        - Phase 1: Freeze HeAR encoder, train only classification head
        - Phase 2: Unfreeze all layers for end-to-end fine-tuning
    """
    
    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        hidden_dim: int = config.HIDDEN_DIM,
        dropout: float = config.DROPOUT,
        pretrained_model_name: str = config.HEAR_MODEL_NAME
    ):
        """
        Args:
            num_classes: Number of output classes
            hidden_dim: Hidden dimension in classification head
            dropout: Dropout probability
            pretrained_model_name: HuggingFace model name for HeAR
        """
        super(HeARClassifier, self).__init__()
        
        print(f"Loading HeAR model from HuggingFace: {pretrained_model_name}")
        print("Note: This is a gated model. Ensure you have:")
        print("  1. Requested access: https://huggingface.co/google/hear-pytorch")
        print("  2. Authenticated: Run 'huggingface-cli login' or set HF_TOKEN in .env")
        
        # Load pre-trained HeAR encoder from HuggingFace
        # Token will be automatically used if logged in via CLI or .env
        self.hear_encoder = AutoModel.from_pretrained(
            pretrained_model_name,
            use_auth_token=True  # Use token from environment or HF CLI
        )
        
        # Get embedding dimension from HeAR model
        self.embedding_dim = config.EMBEDDING_DIM
        
        print(f"HeAR encoder loaded. Embedding dimension: {self.embedding_dim}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Flag to track if encoder is frozen
        self.encoder_frozen = False
        
        print(f"Classification head created: {self.embedding_dim} -> {hidden_dim} -> {num_classes}")
    
    def freeze_encoder(self):
        """Freeze HeAR encoder parameters for phase 1 training."""
        for param in self.hear_encoder.parameters():
            param.requires_grad = False
        self.encoder_frozen = True
        print("✓ HeAR encoder frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze HeAR encoder parameters for phase 2 fine-tuning."""
        for param in self.hear_encoder.parameters():
            param.requires_grad = True
        self.encoder_frozen = False
        print("✓ HeAR encoder unfrozen")
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            audio: Raw audio tensor of shape (batch_size, 32000)
                   Must be 16kHz mono, 2-second clips
        
        Returns:
            logits: Classification logits of shape (batch_size, num_classes)
        """
        # Preprocess audio using HeAR's preprocessing
        # This applies PCEN (Per-Channel Energy Normalization)
        preprocessed = audio_utils.preprocess_audio(audio)
        
        # Get embeddings from HeAR encoder
        # The model returns a dictionary with 'pooler_output'
        outputs = self.hear_encoder(
            preprocessed,
            return_dict=True,
            output_hidden_states=False
        )
        
        # Extract the pooled embedding
        embeddings = outputs.pooler_output  # Shape: (batch_size, 512)
        
        # Pass through classification head
        logits = self.classifier(embeddings)  # Shape: (batch_size, num_classes)
        
        return logits
    
    def get_embeddings(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract HeAR embeddings without classification.
        Useful for visualization and analysis.
        
        Args:
            audio: Raw audio tensor of shape (batch_size, 32000)
        
        Returns:
            embeddings: HeAR embeddings of shape (batch_size, 512)
        """
        with torch.no_grad():
            preprocessed = audio_utils.preprocess_audio(audio)
            outputs = self.hear_encoder(
                preprocessed,
                return_dict=True,
                output_hidden_states=False
            )
            embeddings = outputs.pooler_output
        
        return embeddings
    
    def count_parameters(self):
        """Count trainable and total parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\nModel parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Frozen: {total_params - trainable_params:,}")
        
        return total_params, trainable_params


def load_model_checkpoint(
    checkpoint_path: str,
    device: torch.device = config.DEVICE
) -> HeARClassifier:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        device: Device to load model on
    
    Returns:
        model: Loaded HeARClassifier model
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create model
    model = HeARClassifier()
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Print checkpoint info
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'val_f1_macro' in checkpoint:
        print(f"  Val F1-macro: {checkpoint['val_f1_macro']:.4f}")
    
    print("✓ Model loaded successfully")
    
    return model


if __name__ == "__main__":
    """Test model creation and forward pass."""
    
    print("Testing HeARClassifier...")
    
    # Create model
    model = HeARClassifier()
    model.to(config.DEVICE)
    
    # Count parameters
    model.count_parameters()
    
    # Test forward pass with random audio
    batch_size = 4
    audio = torch.randn(batch_size, config.CLIP_LENGTH).to(config.DEVICE)
    
    print(f"\nTesting forward pass...")
    print(f"Input shape: {audio.shape}")
    
    # Test with frozen encoder (Phase 1)
    model.freeze_encoder()
    with torch.no_grad():
        logits = model(audio)
    print(f"Output shape (frozen encoder): {logits.shape}")
    model.count_parameters()
    
    # Test with unfrozen encoder (Phase 2)
    model.unfreeze_encoder()
    with torch.no_grad():
        logits = model(audio)
    print(f"Output shape (unfrozen encoder): {logits.shape}")
    model.count_parameters()
    
    # Test embeddings extraction
    print(f"\nTesting embedding extraction...")
    embeddings = model.get_embeddings(audio)
    print(f"Embeddings shape: {embeddings.shape}")
    
    print("\n✓ Model tests passed!")
