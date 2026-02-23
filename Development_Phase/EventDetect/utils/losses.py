"""
Loss functions for U-Net temporal segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    
    Dice coefficient = 2 * |A ∩ B| / (|A| + |B|)
    Dice Loss = 1 - Dice coefficient
    """
    
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (batch, 1, time_steps, 1) - Event probabilities
            targets: (batch, time_steps) - Binary ground truth (0 or 1)
        
        Returns:
            dice_loss: Scalar loss value
        """
        # Flatten predictions and targets
        pred_flat = predictions.squeeze(1).squeeze(-1).view(-1)  # (batch * time_steps,)
        target_flat = targets.view(-1).float()  # (batch * time_steps,)
        
        # Apply sigmoid to predictions if not already applied
        pred_flat = torch.sigmoid(pred_flat)
        
        # Calculate Dice coefficient
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice
        
        return dice_loss


class CombinedLoss(nn.Module):
    """
    Combined Binary Cross-Entropy + Dice Loss.
    
    This combination works well for binary segmentation tasks:
    - BCE: Good for pixel-level classification
    - Dice: Good for handling class imbalance
    """
    
    def __init__(self, bce_weight=1.0, dice_weight=1.0, dice_smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=dice_smooth)
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (batch, 1, time_steps, 1) - Event logits
            targets: (batch, time_steps) - Binary ground truth (0 or 1)
        
        Returns:
            combined_loss: Scalar loss value
        """
        # Prepare targets for BCE (same shape as predictions)
        targets_bce = targets.unsqueeze(1).unsqueeze(-1).float()  # (batch, 1, time_steps, 1)
        
        # Calculate losses
        bce = self.bce_loss(predictions, targets_bce)
        dice = self.dice_loss(predictions, targets)
        
        # Combined loss
        combined = self.bce_weight * bce + self.dice_weight * dice
        
        return combined, bce, dice
