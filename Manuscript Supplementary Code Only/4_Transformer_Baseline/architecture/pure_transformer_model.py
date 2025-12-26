#!/usr/bin/env python3
"""
PureTransformerMultiOrgan Model Architecture

This is the model architecture used for step2_fold1_best.pth checkpoint.
Two-stage Swin Transformer Tiny with multi-organ classification heads.

Note: This architecture differs from plan_models.py (fold 0) in:
- Uses global_pool='avg' instead of global_pool=''
- Uses Sequential heads with Dropout(0.3) + Linear(768, 1)
- 768 feature dimensions (vs 7 in fold 0 architecture)

Author: Extracted from step2_simple_corrected.py
Date: December 6, 2025
"""

import torch
import torch.nn as nn
import timm
from typing import Dict


class PureTransformerMultiOrgan(nn.Module):
    """
    Two-Stage Multi-Organ Classification Model

    Stage 1: Slice-level bowel detection (pre-trained backbone)
    Stage 2: Patient-level multi-organ classification

    Architecture:
        - Backbone: Swin Transformer Tiny (swin_tiny_patch4_window7_224)
        - Feature dimension: 768
        - Organ heads: Sequential(Dropout(0.3), Linear(768, 1))
    """

    def __init__(self, step1_model_path: str = None, device: torch.device = None):
        """
        Initialize PureTransformerMultiOrgan model.

        Args:
            step1_model_path: Optional path to Stage 1 bowel expert checkpoint
            device: Optional device for model placement
        """
        super().__init__()

        # Swin Transformer backbone with average pooling
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=False,
            num_classes=0,
            in_chans=3,
            global_pool='avg'  # Key difference from fold 0
        )

        # Feature dimension from Swin Tiny
        self.feature_dim = 768

        # Multi-organ classification heads
        # Each head: Dropout(0.3) -> Linear(768, 1) -> Sigmoid (applied in forward)
        self.bowel_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 1)
        )
        self.liver_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 1)
        )
        self.kidney_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 1)
        )
        self.spleen_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 1)
        )
        self.extravasation_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 1)
        )

        # Optionally load Stage 1 backbone weights
        if step1_model_path:
            self._load_stage1_backbone(step1_model_path, device)

    def _load_stage1_backbone(self, path: str, device: torch.device):
        """Load backbone weights from Stage 1 bowel expert model."""
        checkpoint = torch.load(path, map_location=device or 'cpu')

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Filter backbone weights
        backbone_weights = {
            k.replace('backbone.', ''): v
            for k, v in state_dict.items()
            if k.startswith('backbone.')
        }

        if backbone_weights:
            self.backbone.load_state_dict(backbone_weights, strict=False)
            print(f"Loaded {len(backbone_weights)} backbone weights from Stage 1")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-organ classification.

        Args:
            x: Input tensor [B, 3, 224, 224]

        Returns:
            Dictionary with sigmoid probabilities for each organ:
            {
                'bowel': [B, 1],
                'liver': [B, 1],
                'kidney': [B, 1],
                'spleen': [B, 1],
                'extravasation': [B, 1]
            }
        """
        # Extract features from backbone [B, 768]
        features = self.backbone(x)

        # Classify each organ with sigmoid activation
        outputs = {
            'bowel': torch.sigmoid(self.bowel_head(features)),
            'liver': torch.sigmoid(self.liver_head(features)),
            'kidney': torch.sigmoid(self.kidney_head(features)),
            'spleen': torch.sigmoid(self.spleen_head(features)),
            'extravasation': torch.sigmoid(self.extravasation_head(features))
        }

        return outputs


def load_fold1_checkpoint(checkpoint_path: str, device: torch.device = None):
    """
    Load the fold 1 checkpoint with correct architecture.

    Args:
        checkpoint_path: Path to step2_fold1_best.pth
        device: Device for model placement

    Returns:
        Loaded model in eval mode

    Example:
        model = load_fold1_checkpoint('weights/step2_fold1_best.pth', device)
        outputs = model(batch)  # Returns dict of probabilities
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = PureTransformerMultiOrgan()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Load weights
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    # Print checkpoint metadata
    if 'epoch' in checkpoint:
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    if 'best_val_metrics' in checkpoint:
        metrics = checkpoint['best_val_metrics']
        print(f"Validation metrics: {metrics}")

    return model


if __name__ == '__main__':
    # Test model architecture
    print("Testing PureTransformerMultiOrgan architecture...")

    model = PureTransformerMultiOrgan()
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        outputs = model(dummy_input)

    print(f"\nOutput shapes:")
    for organ, prob in outputs.items():
        print(f"  {organ}: {prob.shape}")

    print("\nArchitecture test passed!")
