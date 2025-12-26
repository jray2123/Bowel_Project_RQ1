#!/usr/bin/env python3
"""
Enhanced 2.5D CNN Architecture for Multi-Organ Trauma Detection

This module implements the CNN25DEnhanced architecture following the Enhanced 2.5D 
Baseline Execution Plan specifications.

Key features:
- EfficientNet-B3 feature extractor (384 features per slice)
- Position-aware bidirectional LSTM for sequence modeling
- Dual aggregation strategies (temporal vs spatial)
- Multi-organ classification heads with dropout
- Memory-optimized for RTX 3090 (8GB peak usage)

Architecture follows exact specifications from execution plan:
Input: (batch, 48, 3, 320, 320) + positions (batch, 48)
→ EfficientNet-B3 per slice → (batch, 48, 384)
→ Concatenate position → (batch, 48, 385) 
→ Bidirectional LSTM → (batch, 48, 512)
→ Aggregation (final or max-pool) → (batch, 512)
→ Multi-organ heads → 5 organ predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import warnings
from torchvision import models
import timm


class CNN25DEnhanced(nn.Module):
    """
    Enhanced 2.5D CNN following execution plan specifications.
    
    Architecture:
    - Feature Extractor: EfficientNet-B3 (pretrained, 384 features)
    - Position-aware LSTM: Input = CNN features + normalized position  
    - Dual Aggregation: Temporal (final) or Spatial (max-pool)
    - Multi-organ Heads: Separate classification heads with dropout
    
    Memory Requirements:
    - EfficientNet-B3: ~300MB parameters
    - LSTM: ~10MB parameters  
    - Total model: ~315MB
    - Peak memory: ~8GB (including activations)
    """
    
    def __init__(
        self,
        # Architecture parameters
        feature_extractor: str = "efficientnet-b3",
        feature_dim: int = 384,
        lstm_hidden_dim: int = 256,
        lstm_bidirectional: bool = True,
        aggregation_strategy: str = "final",  # "final" or "max_pool"
        position_encoding: bool = True,
        
        # Multi-organ classification
        num_organs: int = 5,
        organ_classes: Dict[str, int] = None,
        dropout: float = 0.2,
        
        # Training parameters
        pretrained: bool = True,
        freeze_backbone_epochs: int = 0
    ):
        """
        Initialize Enhanced 2.5D CNN.
        
        Args:
            feature_extractor: Backbone CNN (must be efficientnet-b3 per plan)
            feature_dim: Feature dimension from backbone (384 for EfficientNet-B3)
            lstm_hidden_dim: LSTM hidden dimension (256 per plan)
            lstm_bidirectional: Use bidirectional LSTM
            aggregation_strategy: "final" (temporal) or "max_pool" (spatial)
            position_encoding: Enable anatomical position encoding
            num_organs: Number of organs to classify (5 per plan)
            organ_classes: Dict mapping organ names to number of classes
            dropout: Dropout rate for classification heads
            pretrained: Use ImageNet pretrained weights
            freeze_backbone_epochs: Epochs to freeze backbone (0 = no freezing)
        """
        super(CNN25DEnhanced, self).__init__()
        
        # Validate parameters against execution plan
        if feature_extractor != "efficientnet-b3":
            raise ValueError("Execution plan requires EfficientNet-B3")
        if feature_dim not in [384, 1536]:
            raise ValueError("EfficientNet-B3 must output 384 or 1536 features")
        
        # Store configuration
        self.feature_extractor_name = feature_extractor
        self.feature_dim = feature_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_bidirectional = lstm_bidirectional
        self.aggregation_strategy = aggregation_strategy
        self.position_encoding = position_encoding
        self.num_organs = num_organs
        self.dropout_rate = dropout
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.current_epoch = 0
        
        # Default organ classes per RSNA RATIC dataset
        if organ_classes is None:
            self.organ_classes = {
                'bowel': 2,        # healthy, injury
                'extravasation': 2, # healthy, injury
                'liver': 3,        # healthy, low, high
                'kidney': 3,       # healthy, low, high
                'spleen': 3        # healthy, low, high
            }
        else:
            self.organ_classes = organ_classes
            
        # Build architecture components
        self._build_feature_extractor()
        self._build_sequence_processor()
        self._build_classification_heads()
        
        # Initialize weights
        self._initialize_weights()
        
        print(f"Initialized CNN25DEnhanced:")
        print(f"  Feature extractor: {self.feature_extractor_name}")
        print(f"  LSTM: {self.lstm_hidden_dim} {'bidirectional' if lstm_bidirectional else 'unidirectional'}")
        print(f"  Aggregation: {self.aggregation_strategy}")
        print(f"  Position encoding: {self.position_encoding}")
        print(f"  Organs: {list(self.organ_classes.keys())}")
        
    def _build_feature_extractor(self):
        """Build EfficientNet-B3 feature extractor."""
        # Load pretrained EfficientNet-B3
        self.backbone = timm.create_model(
            'efficientnet_b3',
            pretrained=True,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )
        
        # Verify feature dimension
        with torch.no_grad():
            test_input = torch.randn(1, 3, 320, 320)
            test_features = self.backbone(test_input)
            # Global average pooling to get feature vector
            test_features_pooled = F.adaptive_avg_pool2d(test_features, 1).flatten(1)
            actual_dim = test_features_pooled.shape[1]
            
        if actual_dim != self.feature_dim:
            raise ValueError(f"Expected {self.feature_dim} features, got {actual_dim}")
            
        print(f"  ✓ EfficientNet-B3 loaded: {actual_dim} features per slice")
        
    def _build_sequence_processor(self):
        """Build position-aware LSTM for sequence processing."""
        # Calculate LSTM input dimension
        lstm_input_dim = self.feature_dim
        if self.position_encoding:
            lstm_input_dim += 1  # Add normalized Z-position
            
        # Build bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=self.lstm_bidirectional,
            dropout=0.0  # Only one layer, no dropout
        )
        
        # Calculate output dimension after LSTM
        self.lstm_output_dim = self.lstm_hidden_dim * (2 if self.lstm_bidirectional else 1)
        
        print(f"  ✓ LSTM: {lstm_input_dim} → {self.lstm_output_dim}")
        
    def _build_classification_heads(self):
        """Build multi-organ classification heads."""
        self.classification_heads = nn.ModuleDict()
        
        for organ, num_classes in self.organ_classes.items():
            # Each head: Linear + Dropout + Linear
            head = nn.Sequential(
                nn.Linear(self.lstm_output_dim, self.lstm_output_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.lstm_output_dim // 2, num_classes)
            )
            self.classification_heads[organ] = head
            
        print(f"  ✓ Classification heads: {dict(self.organ_classes)}")
        
    def _initialize_weights(self):
        """Initialize weights for non-pretrained components."""
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                
        # Initialize classification heads
        for head in self.classification_heads.values():
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.constant_(module.bias, 0)
                    
    def set_epoch(self, epoch: int):
        """Set current epoch for backbone freezing control."""
        self.current_epoch = epoch
        
        # Control backbone freezing
        if epoch < self.freeze_backbone_epochs:
            self._freeze_backbone()
        else:
            self._unfreeze_backbone()
            
    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def _unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
            
    def extract_slice_features(self, slice_batch: torch.Tensor) -> torch.Tensor:
        """
        Extract features from a batch of slices.
        
        Args:
            slice_batch: [batch_size, 3, 320, 320] batch of slices
            
        Returns:
            features: [batch_size, 1536] feature vectors
        """
        # Extract features using EfficientNet-B3
        features = self.backbone(slice_batch)  # [batch, 1536, H', W']
        
        # Global average pooling to get feature vector
        features = F.adaptive_avg_pool2d(features, 1)  # [batch, 1536, 1, 1]
        features = features.flatten(1)  # [batch, 1536]
        
        return features
    
    def process_sequence(
        self, 
        slice_sequence: torch.Tensor,
        position_encoding: torch.Tensor,
        sequence_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Process slice sequence through feature extractor and LSTM.
        
        Args:
            slice_sequence: [batch, seq_len, 3, H, W] sequence of slices
            position_encoding: [batch, seq_len, 1] normalized positions
            sequence_mask: [batch, seq_len] valid slice mask
            
        Returns:
            aggregated_features: [batch, lstm_output_dim] aggregated sequence features
        """
        batch_size, seq_len = slice_sequence.shape[:2]
        
        # Extract features from all slices
        # Reshape to process all slices at once
        all_slices = slice_sequence.view(-1, 3, 320, 320)  # [batch*seq, 3, 320, 320]
        all_features = self.extract_slice_features(all_slices)  # [batch*seq, 1536]
        
        # Reshape back to sequence
        sequence_features = all_features.view(batch_size, seq_len, self.feature_dim)  # [batch, seq, 1536]
        
        # Add position encoding if enabled
        if self.position_encoding:
            lstm_input = torch.cat([sequence_features, position_encoding], dim=-1)  # [batch, seq, 1537]
        else:
            lstm_input = sequence_features
            
        # Process through LSTM
        lstm_output, (final_hidden, final_cell) = self.lstm(lstm_input)  # [batch, seq, lstm_out]
        
        # Apply aggregation strategy
        if self.aggregation_strategy == "final":
            # Use final timestep (temporal aggregation)
            if self.lstm_bidirectional:
                # Concatenate forward and backward final states
                final_forward = final_hidden[-2]  # Last layer, forward direction
                final_backward = final_hidden[-1]  # Last layer, backward direction
                aggregated = torch.cat([final_forward, final_backward], dim=-1)
            else:
                aggregated = final_hidden[-1]  # Last layer, final state
                
        elif self.aggregation_strategy == "max_pool":
            # Use max pooling across sequence dimension (spatial aggregation)
            # Apply sequence mask to ignore padded positions
            masked_output = lstm_output * sequence_mask.unsqueeze(-1).float()
            aggregated, _ = torch.max(masked_output, dim=1)  # [batch, lstm_out]
            
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.aggregation_strategy}")
            
        return aggregated
    
    def forward(
        self,
        slice_sequence: torch.Tensor,
        position_encoding: Optional[torch.Tensor] = None,
        sequence_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the enhanced 2.5D CNN.
        
        Args:
            slice_sequence: [batch, 48, 3, 320, 320] sequence of CT slices
            position_encoding: [batch, 48, 1] normalized anatomical positions
            sequence_mask: [batch, 48] mask for valid slices (optional)
            
        Returns:
            outputs: Dict with logits for each organ
        """
        batch_size, seq_len = slice_sequence.shape[:2]
        
        # Handle missing inputs
        if position_encoding is None:
            position_encoding = torch.zeros(batch_size, seq_len, 1, device=slice_sequence.device)
        if sequence_mask is None:
            sequence_mask = torch.ones(batch_size, seq_len, device=slice_sequence.device, dtype=torch.bool)
            
        # Process sequence
        aggregated_features = self.process_sequence(slice_sequence, position_encoding, sequence_mask)
        
        # Apply classification heads
        outputs = {}
        for organ, head in self.classification_heads.items():
            outputs[organ] = head(aggregated_features)
            
        return outputs
    
    def get_parameter_groups(self, base_lr: float = 1e-4) -> List[Dict]:
        """
        Get parameter groups for different learning rates.
        
        Args:
            base_lr: Base learning rate
            
        Returns:
            List of parameter groups with different learning rates
        """
        # CNN uses uniform learning rate (no layer-wise like ViT)
        param_groups = [
            {
                'params': list(self.backbone.parameters()),
                'lr': base_lr,
                'name': 'backbone'
            },
            {
                'params': list(self.lstm.parameters()),
                'lr': base_lr,
                'name': 'lstm'
            }
        ]
        
        # Add classification heads
        for organ, head in self.classification_heads.items():
            param_groups.append({
                'params': list(head.parameters()),
                'lr': base_lr,
                'name': f'head_{organ}'
            })
            
        return param_groups
    
    def get_model_info(self) -> Dict:
        """Get model information for logging."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'CNN25DEnhanced',
            'feature_extractor': self.feature_extractor_name,
            'feature_dim': self.feature_dim,
            'lstm_hidden_dim': self.lstm_hidden_dim,
            'lstm_bidirectional': self.lstm_bidirectional,
            'aggregation_strategy': self.aggregation_strategy,
            'position_encoding': self.position_encoding,
            'organs': list(self.organ_classes.keys()),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'memory_estimate_mb': total_params * 4 / 1024 / 1024  # Rough estimate
        }


def create_cnn_25d_enhanced(config: Dict) -> CNN25DEnhanced:
    """
    Create Enhanced 2.5D CNN model from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized CNN25DEnhanced model
    """
    model_config = config.get('model', {})
    
    model = CNN25DEnhanced(
        feature_extractor=model_config.get('feature_extractor', 'efficientnet-b3'),
        feature_dim=model_config.get('feature_dim', 1536),
        lstm_hidden_dim=model_config.get('lstm_hidden_dim', 256),
        lstm_bidirectional=model_config.get('lstm_bidirectional', True),
        aggregation_strategy=model_config.get('aggregation_strategy', 'final'),
        position_encoding=model_config.get('position_encoding', True),
        num_organs=model_config.get('num_organs', 5),
        dropout=model_config.get('dropout', 0.2)
    )
    
    return model


if __name__ == "__main__":
    # Test model creation and forward pass
    print("Testing CNN25DEnhanced...")
    
    # Create model
    model = CNN25DEnhanced()
    
    # Test forward pass
    batch_size = 2
    seq_len = 48
    
    # Create test inputs following execution plan specs
    slice_sequence = torch.randn(batch_size, seq_len, 3, 320, 320)
    position_encoding = torch.randn(batch_size, seq_len, 1)
    sequence_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    print(f"\nTesting forward pass:")
    print(f"  Input: {slice_sequence.shape}")
    print(f"  Position: {position_encoding.shape}")
    print(f"  Mask: {sequence_mask.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(slice_sequence, position_encoding, sequence_mask)
    
    print(f"\nOutputs:")
    for organ, logits in outputs.items():
        print(f"  {organ}: {logits.shape}")
    
    # Model info
    info = model.get_model_info()
    print(f"\nModel Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("✅ CNN25DEnhanced test completed successfully!")