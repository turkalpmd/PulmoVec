"""
U-Net model for temporal event segmentation in respiratory audio.

Architecture:
    Input: STFT/Mel-Spectrogram (time x frequency)
    Output: Segmentation mask (time x num_classes)
    
    - Encoder: Downsampling with Conv2D + MaxPooling
    - Bottleneck: Dense feature representation
    - Decoder: Upsampling with Transposed Conv2D + Skip connections
    - Output: Temporal mask (time x num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2D -> BN -> ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling block: MaxPool -> DoubleConv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling block: TransposedConv -> Concatenate -> DoubleConv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        x1: Features from decoder (upsampled)
        x2: Skip connection from encoder
        """
        x1 = self.up(x1)
        
        # Handle dimension mismatch (padding if needed)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetTemporalSegmentation(nn.Module):
    """
    U-Net for binary temporal event segmentation in audio spectrograms.
    
    Input: Spectrogram (batch, 1, time_steps, freq_bins)
    Output: Binary segmentation mask (batch, 1, time_steps, 1) - Event probability per time step
    
    Architecture:
        Encoder: Extract features from spectrogram (downsampling)
        Bottleneck: Dense feature representation
        Decoder: Reconstruct temporal resolution (upsampling with skip connections)
        Output: Binary temporal mask (event probability)
    """
    
    def __init__(
        self,
        n_channels: int = 1,  # Input channels (mono spectrogram)
        n_freq_bins: int = 128,  # Frequency bins in spectrogram
        bilinear: bool = True
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = 1  # Binary segmentation: Event (1) or Normal (0)
        self.bilinear = bilinear
        
        # Encoder (downsampling)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder (upsampling with skip connections)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Output head: Binary temporal mask prediction
        # Output shape: (batch, 1, time_steps, 1) - Event probability
        self.outc = nn.Conv2d(64, 1, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input spectrogram (batch, 1, time_steps, freq_bins)
        
        Returns:
            mask: Binary segmentation mask (batch, 1, time_steps, 1) - Event logits
        """
        # Encoder
        x1 = self.inc(x)  # (batch, 64, time, freq)
        x2 = self.down1(x1)  # (batch, 128, time/2, freq/2)
        x3 = self.down2(x2)  # (batch, 256, time/4, freq/4)
        x4 = self.down3(x3)  # (batch, 512, time/8, freq/8)
        x5 = self.down4(x4)  # (batch, 512, time/16, freq/16)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)  # (batch, 256, time/8, freq/8)
        x = self.up2(x, x3)  # (batch, 128, time/4, freq/4)
        x = self.up3(x, x2)  # (batch, 64, time/2, freq/2)
        x = self.up4(x, x1)  # (batch, 64, time, freq)
        
        # Output: Binary temporal mask (event logits)
        mask = self.outc(x)  # (batch, 1, time, freq)
        
        # Average over frequency dimension to get temporal mask
        # (batch, 1, time, 1)
        mask = torch.mean(mask, dim=3, keepdim=True)
        
        return mask
    
    def predict_temporal_mask(self, x, threshold=0.5):
        """
        Predict binary temporal segmentation mask with thresholding.
        
        Args:
            x: Input spectrogram (batch, 1, time_steps, freq_bins)
            threshold: Probability threshold for event detection (default: 0.5)
        
        Returns:
            mask: Binary mask (batch, time_steps) - 1=Event, 0=Normal
            probs: Event probability mask (batch, time_steps)
        """
        self.eval()
        with torch.no_grad():
            # Get logits
            logits = self.forward(x)  # (batch, 1, time_steps, 1)
            logits = logits.squeeze(1).squeeze(-1)  # (batch, time_steps)
            
            # Apply sigmoid for binary probabilities
            probs = torch.sigmoid(logits)  # (batch, time_steps) - Event probability
            
            # Get binary predictions
            mask = (probs > threshold).long()  # (batch, time_steps) - 1=Event, 0=Normal
        
        return mask, probs
