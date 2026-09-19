#!/usr/bin/env python3
"""
src/models_hear_lora.py

High-Performance Google HeAR Foundation Model with:
1. Offline ViT-Large backbone loading from cached weights
2. LoRA (Low-Rank Adaptation, r=8, alpha=16) on layers 18-23
3. Layers 0-17 fully frozen (preserving 313k hours of health acoustic priors)
4. Multi-Head Attention Patch Pooling over all 96 time-frequency patch tokens
5. Dual-Branch (CLS + Attended Patches) 1536D fusion
"""

import sys
import math
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTConfig, ViTModel

# Add hear/python to path for audio_utils
hear_python_path = Path(__file__).parent.parent / "hear" / "python"
if str(hear_python_path) not in sys.path:
    sys.path.insert(0, str(hear_python_path))

from data_processing import audio_utils


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer wrapping a pre-trained frozen linear layer.
    h = W_orig(x) + (alpha / r) * B(A(x))
    """
    def __init__(self, original_linear: nn.Linear, r: int = 8, lora_alpha: float = 16.0):
        super().__init__()
        self.original_linear = original_linear
        # Freeze original weight
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
            
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / float(r)
        
        # Trainable low-rank decomposition matrices
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = self.original_linear(x)
        lora = (F.linear(x, self.lora_A) @ self.lora_B.t()) * self.scaling
        return orig + lora


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-Head Attention Pooling across 96 time-frequency patch embeddings.
    Allows the model to learn which temporal slices (e.g. crackle clicks) 
    and frequency bands (e.g. wheeze oscillations) contain the pathology.
    """
    def __init__(self, d_model: int = 1024, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        self.query = nn.Parameter(torch.randn(1, num_heads, 1, self.d_k))
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        nn.init.normal_(self.query, std=0.02)

    def forward(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patches: [Batch, 96, 1024]
        Returns:
            attended: [Batch, 1024]
            attn_weights: [Batch, num_heads, 96]
        """
        B, N, D = patches.shape
        # Key & Value projections
        K = self.k_proj(patches).view(B, N, self.num_heads, self.d_k).transpose(1, 2) # [B, H, N, d_k]
        V = self.v_proj(patches).view(B, N, self.num_heads, self.d_k).transpose(1, 2) # [B, H, N, d_k]
        
        Q = self.query.expand(B, -1, -1, -1) # [B, H, 1, d_k]
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale # [B, H, 1, N]
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        attended = torch.matmul(attn, V).squeeze(2) # [B, H, d_k]
        attended = attended.reshape(B, D)           # [B, 1024]
        attended = self.out_proj(attended)          # [B, 1024]
        
        return attended, attn.squeeze(2)


class HeARPooler(nn.Module):
    """Native HeAR 512D Pooler on CLS token."""
    def __init__(self, in_dim: int = 1024, out_dim: int = 512):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token = hidden_states[:, 0]
        return self.activation(self.dense(first_token))


class HeARLoRAClassifier(nn.Module):
    """
    Advanced HeAR Classifier with:
    - ViT-Large backbone (1024D, 24 layers)
    - Layers 0-17 frozen
    - Layers 18-23 LoRA adapted (r=8, alpha=16)
    - Multi-Head Attention Patch Pooling over 96 time-frequency tokens
    - Dual-Branch 1536D Fusion
    """
    def __init__(
        self,
        num_classes: int = 3,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_layers: int = 6, # Layers 18 to 23
        weights_path: Optional[str] = None,
        dropout: float = 0.3
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # 1. Instantiate ViT-Large architecture matching Google HeAR
        config = ViTConfig(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
            image_size=(192, 128),
            patch_size=16,
            num_channels=1,
            qkv_bias=True,
            add_pooling_layer=False
        )
        self.hear_encoder = ViTModel(config)
        self.hear_encoder.pooler = HeARPooler(1024, 512)
        
        # 2. Load Offline Pretrained Weights
        if weights_path is None:
            candidates = [
                Path('/home/izzet/Desktop/PulmoVec_Workspace/models/hear_sprsound_best.pth'),
                Path('/home/izzet/Desktop/PulmoVec/models/hear_sprsound_best.pth')
            ]
            for c in candidates:
                if c.exists():
                    weights_path = str(c)
                    break
                    
        if weights_path and Path(weights_path).exists():
            print(f"Loading pretrained HeAR weights from: {weights_path}")
            ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
            sd = ckpt.get('model_state_dict', ckpt)
            encoder_sd = {k[len('hear_encoder.'):]: v for k, v in sd.items() if k.startswith('hear_encoder.')}
            self.hear_encoder.load_state_dict(encoder_sd, strict=True)
            print("✓ Offline HeAR ViT-Large backbone loaded with 100% bitwise accuracy!")
        else:
            print("⚠️ Pretrained weights not found at default paths, initializing randomly!")
            
        # 3. Freeze All Base Backbone Parameters
        for param in self.hear_encoder.parameters():
            param.requires_grad = False
            
        # 4. Attach LoRA to Layers 18-23 Query and Value Projections
        total_layers = len(self.hear_encoder.encoder.layer) # 24
        start_lora_idx = total_layers - lora_layers          # 18
        
        self.lora_params = []
        for i in range(start_lora_idx, total_layers):
            layer = self.hear_encoder.encoder.layer[i]
            # Wrap query
            layer.attention.attention.query = LoRALinear(
                layer.attention.attention.query, r=lora_r, lora_alpha=lora_alpha
            )
            # Wrap value
            layer.attention.attention.value = LoRALinear(
                layer.attention.attention.value, r=lora_r, lora_alpha=lora_alpha
            )
            
        print(f"✓ LoRA (r={lora_r}, alpha={lora_alpha}) attached to Layers {start_lora_idx}..{total_layers-1} (Q & V projections)!")
        
        # 5. Attentive Patch Pooling (96 time-frequency patches)
        self.patch_pooler = MultiHeadAttentionPooling(d_model=1024, num_heads=4, dropout=dropout)
        
        # 6. Dual-Branch Fusion Head: CLS (512D) + Attended Patches (1024D) = 1536D
        self.classifier = nn.Sequential(
            nn.Linear(512 + 1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Print parameter statistics
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model Parameters: {trainable_params:,} trainable / {total_params:,} total ({trainable_params/total_params*100:.2f}% trainable)")

    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            audio: [Batch, 32000] raw audio (16kHz, 2.0s)
        Returns:
            logits: [Batch, num_classes]
            attn_weights: [Batch, 4, 96] attention map over time-frequency patches
        """
        # Preprocess to [Batch, 1, 192, 128] Mel-PCEN
        preprocessed = audio_utils.preprocess_audio(audio).to(audio.device)
        
        outputs = self.hear_encoder(
            preprocessed,
            output_hidden_states=False,
            return_dict=True
        )
        
        # 1. Global CLS embedding [Batch, 512]
        cls_embedding = outputs.pooler_output
        
        # 2. Extract 96 time-frequency patch tokens [Batch, 96, 1024]
        # Token 0 is CLS, Tokens 1..96 are the 12x8 patches
        patch_tokens = outputs.last_hidden_state[:, 1:, :]
        
        # 3. Attentive Patch Pooling
        attended_patches, attn_weights = self.patch_pooler(patch_tokens) # [Batch, 1024]
        
        # 4. Dual-Branch Fusion [Batch, 1536]
        fused_embedding = torch.cat([cls_embedding, attended_patches], dim=-1)
        
        # 5. Classification Logits
        logits = self.classifier(fused_embedding)
        
        return logits, attn_weights

    def get_fused_embeddings(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract 1536D fused representation for stacking / TSNE."""
        with torch.no_grad():
            preprocessed = audio_utils.preprocess_audio(audio).to(audio.device)
            outputs = self.hear_encoder(preprocessed, return_dict=True)
            cls_emb = outputs.pooler_output
            patches = outputs.last_hidden_state[:, 1:, :]
            attended, _ = self.patch_pooler(patches)
            return torch.cat([cls_emb, attended], dim=-1)


if __name__ == '__main__':
    print("Testing HeARLoRAClassifier...")
    model = HeARLoRAClassifier(num_classes=3)
    dummy_audio = torch.randn(2, 32000)
    logits, attn = model(dummy_audio)
    print("Forward pass successful!")
    print(f"Logits shape: {logits.shape}")
    print(f"Attention weights shape: {attn.shape}")
