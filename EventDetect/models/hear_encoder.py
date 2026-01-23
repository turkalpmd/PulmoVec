"""
HeAR encoder module for extracting audio embeddings.

Uses HuggingFace 'google/hear-pytorch' model to extract 512-d embeddings
from 2-second audio windows with sliding window support.
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
import hashlib
import json
from transformers import AutoModel, AutoConfig
import warnings
import sys
import importlib

# Add hear package to Python path if available
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
HEAR_PACKAGE_PATH = PROJECT_ROOT / "hear"
if HEAR_PACKAGE_PATH.exists():
    # Add the parent directory to path so we can import 'hear' package
    parent_path = str(HEAR_PACKAGE_PATH.parent)
    if parent_path not in sys.path:
        sys.path.insert(0, parent_path)

# Try to import HeAR audio utils for preprocessing
HEAR_PREPROCESSING_AVAILABLE = False
preprocess_audio = None

try:
    # Try importing the audio_utils module
    audio_utils = importlib.import_module("hear.python.data_processing.audio_utils")
    if hasattr(audio_utils, 'preprocess_audio'):
        preprocess_audio = audio_utils.preprocess_audio
        HEAR_PREPROCESSING_AVAILABLE = True
        print("✓ HeAR preprocessing module loaded successfully")
    else:
        raise AttributeError("preprocess_audio function not found in audio_utils")
except (ImportError, ModuleNotFoundError, AttributeError) as e:
    # If hear package is not available, we'll need to implement preprocessing
    HEAR_PREPROCESSING_AVAILABLE = False
    print(f"Warning: hear.python.data_processing.audio_utils not found: {e}")
    print(f"  Searched in: {HEAR_PACKAGE_PATH}")
    print(f"  Python path includes: {[p for p in sys.path[:5] if 'hear' in p or 'PulmoVec' in p]}")
    print("  Preprocessing may not work correctly.")


class HeAREncoder(nn.Module):
    """
    HeAR encoder wrapper for extracting audio embeddings.
    
    Supports:
    - Frozen encoder (default): only BiLSTM is trainable
    - Fine-tuning: optionally unfreeze last N layers
    - Sliding window extraction with configurable hop
    """
    
    def __init__(
        self,
        model_name: str = "google/hear-pytorch",
        frozen: bool = True,
        fine_tune_last_n_layers: int = 0,
        device: Union[str, torch.device] = "cpu"
    ):
        """
        Initialize HeAR encoder.
        
        Args:
            model_name: HuggingFace model name
            frozen: If True, freeze all encoder parameters
            fine_tune_last_n_layers: Number of last layers to unfreeze (only if frozen=False)
            device: Device to load model on
        """
        super().__init__()
        self.model_name = model_name
        self.frozen = frozen
        self.fine_tune_last_n_layers = fine_tune_last_n_layers
        self.device = torch.device(device)
        self.embedding_dim = 512  # HeAR default embedding dimension
        
        # Load model
        self._load_model()
        
        # Set trainability
        self._set_trainability()
    
    def _load_model(self):
        """Load HeAR model from HuggingFace."""
        try:
            print(f"Loading HeAR model: {self.model_name}")
            print("Note: This may require accepting HuggingFace terms/conditions.")
            
            # Try to load model
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.model.to(self.device)
            self.model.eval()  # Set to eval mode by default
            
            # Get model config for version tracking
            try:
                config = AutoConfig.from_pretrained(self.model_name)
                self.model_version = getattr(config, 'model_type', 'unknown')
            except:
                self.model_version = 'unknown'
            
            print(f"✓ HeAR model loaded successfully")
            print(f"  Model version: {self.model_version}")
            print(f"  Embedding dim: {self.embedding_dim}")
            
        except Exception as e:
            error_msg = (
                f"Failed to load HeAR model '{self.model_name}': {str(e)}\n"
                f"This may require:\n"
                f"  1. Accepting HuggingFace terms/conditions at: https://huggingface.co/{self.model_name}\n"
                f"  2. Logging in with: huggingface-cli login\n"
                f"  3. Or using a local model path if available"
            )
            raise RuntimeError(error_msg) from e
    
    def _set_trainability(self):
        """Set which parameters are trainable."""
        if self.frozen:
            # Freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False
            print("✓ Encoder frozen (all parameters)")
        else:
            # Unfreeze last N layers if specified
            if self.fine_tune_last_n_layers > 0:
                # Get all layers (this is model-specific, may need adjustment)
                # For most transformer models, we can iterate through named modules
                all_layers = []
                for name, module in self.model.named_modules():
                    if 'layer' in name.lower() or 'block' in name.lower():
                        all_layers.append((name, module))
                
                # Unfreeze last N layers
                if len(all_layers) > 0:
                    layers_to_unfreeze = all_layers[-self.fine_tune_last_n_layers:]
                    for name, module in layers_to_unfreeze:
                        for param in module.parameters():
                            param.requires_grad = True
                    print(f"✓ Unfrozen last {self.fine_tune_last_n_layers} layers")
                else:
                    # If we can't identify layers, unfreeze all
                    for param in self.model.parameters():
                        param.requires_grad = True
                    print("✓ All encoder parameters unfrozen (could not identify specific layers)")
            else:
                # Unfreeze all
                for param in self.model.parameters():
                    param.requires_grad = True
                print("✓ All encoder parameters unfrozen")
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from audio.
        
        Args:
            audio: Audio tensor (batch, samples) or (samples,)
                  Expected: 16kHz, 2 seconds (32000 samples)
        
        Returns:
            embeddings: (batch, 512) or (512,) embedding tensor
        """
        # Ensure correct shape: (batch, samples)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Ensure audio is float32
        if audio.dtype != torch.float32:
            audio = audio.float()
        
        # Move to device
        audio = audio.to(self.device)
        
        # Preprocess audio using HeAR's preprocessing
        # This converts raw audio to mel-pcen spectrograms
        if HEAR_PREPROCESSING_AVAILABLE:
            # HeAR preprocessing expects (batch, samples) -> (batch, 1, 192, 128)
            preprocessed = preprocess_audio(audio)
        else:
            # Fallback: try to use raw audio (may not work)
            preprocessed = audio
            warnings.warn("HeAR preprocessing not available, using raw audio. This may fail.")
        
        # HeAR model forward pass
        with torch.no_grad() if self.frozen else torch.enable_grad():
            try:
                # Call model with preprocessed input
                outputs = self.model(
                    preprocessed,
                    return_dict=True,
                    output_hidden_states=False
                )
                
                # Extract embeddings from model output
                # HeAR model returns a dictionary with 'pooler_output'
                if isinstance(outputs, dict):
                    if 'pooler_output' in outputs:
                        embeddings = outputs['pooler_output']
                    elif 'last_hidden_state' in outputs:
                        # If no pooler, use mean pooling over sequence dimension
                        embeddings = outputs['last_hidden_state']
                        if embeddings.dim() == 3:
                            embeddings = embeddings.mean(dim=1)  # Mean pooling
                    else:
                        # Try to get first tensor value
                        embeddings = list(outputs.values())[0]
                        if embeddings.dim() == 3:
                            embeddings = embeddings.mean(dim=1)
                elif hasattr(outputs, 'pooler_output'):
                    embeddings = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    embeddings = outputs.last_hidden_state
                    if embeddings.dim() == 3:
                        embeddings = embeddings.mean(dim=1)
                else:
                    raise ValueError(f"Unexpected model output type: {type(outputs)}")
                
                # Ensure correct shape
                if embeddings.dim() == 1:
                    embeddings = embeddings.unsqueeze(0)
                
                # Ensure correct dimension (should be 512 for HeAR)
                if embeddings.shape[-1] != self.embedding_dim:
                    # If dimension mismatch, use linear projection (shouldn't happen normally)
                    if not hasattr(self, 'proj'):
                        self.proj = nn.Linear(embeddings.shape[-1], self.embedding_dim).to(self.device)
                    embeddings = self.proj(embeddings)
                
                return embeddings
                
            except Exception as e:
                raise RuntimeError(
                    f"Error during HeAR forward pass: {str(e)}\n"
                    f"Audio shape: {audio.shape}\n"
                    f"Preprocessed shape: {preprocessed.shape if 'preprocessed' in locals() else 'N/A'}\n"
                    f"Expected: (batch, 32000) for 2s @ 16kHz"
                ) from e
    
    def extract_embeddings_sliding_window(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        window_sec: float = 2.0,
        hop_sec: float = 0.25
    ) -> np.ndarray:
        """
        Extract embeddings using sliding window approach.
        
        Args:
            audio: Audio array (samples,)
            sample_rate: Audio sample rate
            window_sec: Window duration in seconds (must be 2.0 for HeAR)
            hop_sec: Hop size in seconds
        
        Returns:
            embeddings: (T_windows, 512) array of embeddings
        """
        if window_sec != 2.0:
            warnings.warn(
                f"HeAR model expects 2.0s windows, but {window_sec}s provided. "
                f"Using 2.0s windows instead."
            )
            window_sec = 2.0
        
        window_samples = int(window_sec * sample_rate)  # 32000 for 2s @ 16kHz
        hop_samples = int(hop_sec * sample_rate)
        
        # Convert to torch tensor
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio.float()
        
        # Ensure mono
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.mean(dim=0)
        
        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            audio_tensor = resampler(audio_tensor)
            sample_rate = 16000
        
        # Extract windows
        embeddings_list = []
        audio_length = len(audio_tensor)
        
        start_idx = 0
        while start_idx + window_samples <= audio_length:
            window = audio_tensor[start_idx:start_idx + window_samples]
            
            # Extract embedding
            embedding = self.forward(window)
            embeddings_list.append(embedding.cpu().numpy())
            
            start_idx += hop_samples
        
        # Handle last window if audio is longer than window
        if start_idx < audio_length:
            # Take last window_samples from end
            window = audio_tensor[-window_samples:]
            embedding = self.forward(window)
            embeddings_list.append(embedding.cpu().numpy())
        
        if len(embeddings_list) == 0:
            # If audio is shorter than window, pad or repeat
            if audio_length < window_samples:
                # Pad to window_samples
                padding = window_samples - audio_length
                window = torch.cat([audio_tensor, torch.zeros(padding)])
            else:
                window = audio_tensor[:window_samples]
            embedding = self.forward(window)
            embeddings_list.append(embedding.cpu().numpy())
        
        embeddings = np.vstack(embeddings_list)  # (T_windows, 512)
        return embeddings


class EmbeddingCache:
    """
    Helper class for caching HeAR embeddings.
    
    Cache key is based on: audio_path + encoder_version + window_sec + hop_sec + sample_rate
    """
    
    def __init__(self, cache_dir: Union[str, Path]):
        """
        Initialize embedding cache.
        
        Args:
            cache_dir: Directory to store cached embeddings
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(
        self,
        audio_path: Union[str, Path],
        encoder_version: str,
        window_sec: float,
        hop_sec: float,
        sample_rate: int
    ) -> str:
        """Generate cache key from parameters."""
        audio_path_str = str(Path(audio_path).absolute())
        key_string = f"{audio_path_str}_{encoder_version}_{window_sec}_{hop_sec}_{sample_rate}"
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return key_hash
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a key."""
        return self.cache_dir / f"{cache_key}.npz"
    
    def get(
        self,
        audio_path: Union[str, Path],
        encoder_version: str,
        window_sec: float,
        hop_sec: float,
        sample_rate: int
    ) -> Optional[np.ndarray]:
        """
        Get cached embeddings if available.
        
        Returns:
            embeddings: (T_windows, 512) array or None if not cached
        """
        cache_key = self._get_cache_key(audio_path, encoder_version, window_sec, hop_sec, sample_rate)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                data = np.load(cache_path)
                embeddings = data['embeddings']
                print(f"✓ Loaded cached embeddings: {cache_path.name}")
                return embeddings
            except Exception as e:
                print(f"Warning: Failed to load cache {cache_path}: {e}")
                return None
        
        return None
    
    def save(
        self,
        embeddings: np.ndarray,
        audio_path: Union[str, Path],
        encoder_version: str,
        window_sec: float,
        hop_sec: float,
        sample_rate: int
    ):
        """
        Save embeddings to cache.
        
        Args:
            embeddings: (T_windows, 512) array
            audio_path: Path to audio file
            encoder_version: Encoder version string
            window_sec: Window duration
            hop_sec: Hop size
            sample_rate: Sample rate
        """
        cache_key = self._get_cache_key(audio_path, encoder_version, window_sec, hop_sec, sample_rate)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            np.savez_compressed(
                cache_path,
                embeddings=embeddings,
                audio_path=str(audio_path),
                encoder_version=encoder_version,
                window_sec=window_sec,
                hop_sec=hop_sec,
                sample_rate=sample_rate
            )
            print(f"✓ Saved embeddings to cache: {cache_path.name}")
        except Exception as e:
            print(f"Warning: Failed to save cache {cache_path}: {e}")
