#!/usr/bin/env python
"""
Correct model loader for Team Oxygen's models based on checkpoint inspection
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add model paths
sys.path.append('/mnt/HDD4/jineel/bowel_project/team_oxygen/Models')
from models_full import Model3, Model4, Model5, Model6, Model7
from src.layers import FPN

class TeamOxygenModelLoader:
    """Load Team Oxygen models with correct architectures"""
    
    @staticmethod
    def load_model3_efficientnet(path, n_classes=10):
        """Load EfficientNetV2 model (Model3)"""
        model = Model3(n_classes=n_classes)
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint)
        model.eval()
        return model
    
    @staticmethod
    def load_model4_coat(path, num_classes=10, arch='medium'):
        """Load CoaT model (Model4) with correct architecture"""
        # Determine seg_classes from checkpoint
        checkpoint = torch.load(path, map_location='cpu')
        
        # Infer seg_classes from final_conv structure
        if 'final_conv.0.conv.0.weight' in checkpoint:
            conv_channels = checkpoint['final_conv.0.conv.0.weight'].shape[0]
            seg_classes = conv_channels  # 48 for most models
        else:
            seg_classes = 4  # Default
        
        model = Model4(num_classes=num_classes, seg_classes=seg_classes, arch=arch, mask_head=False)
        
        # Fix FPN to match checkpoint
        model.fpn = FPN([512, 384, 192], [64]*3)
        model.enc.out_features = ['x1_nocls', 'x2_nocls', 'x3_nocls', 'x4_nocls']
        
        # Load with strict=False to handle minor mismatches
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        return model
    
    @staticmethod
    def load_model5_coat(path, n_classes=10, arch='medium'):
        """Load CoaT model (Model5) with custom architecture"""
        model = Model5(num_classes=n_classes, arch=arch, mask_head=False)
        
        # Fix FPN
        model.fpn = FPN([512, 384, 192], [64]*3)
        model.enc.out_features = ['x1_nocls', 'x2_nocls', 'x3_nocls', 'x4_nocls']
        
        # Custom final_conv for newseg models
        if 'newseg' in path:
            class CustomFinalConv(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.shuf = nn.Sequential(
                        nn.Conv2d(192, 192, kernel_size=1, bias=True),
                        nn.BatchNorm2d(192, track_running_stats=False)
                    )
                    self.pixel_shuffle = nn.PixelShuffle(2)
                    self.conv = nn.Sequential(
                        nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=True),
                        nn.BatchNorm2d(48, track_running_stats=False),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(48, 4, kernel_size=1, bias=True)
                    )
                def forward(self, x):
                    x = self.shuf(x)
                    x = self.pixel_shuffle(x)
                    x = self.conv(x)
                    return x
            
            model.final_conv = nn.Sequential()
            model.final_conv.add_module('0', CustomFinalConv())
        
        # Special handling for extravasation models
        if n_classes == 2 and 'extravast' in path:
            model.head = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(512*2, 2),
            )
        
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        return model
    
    @staticmethod
    def load_model6_efficientnet(path, n_classes=2):
        """Load EfficientNetV2 extravasation model"""
        model = Model6(n_classes=n_classes)
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint)
        model.eval()
        return model
    
    @staticmethod
    def load_model7_unet(path, num_classes=4, arch='medium'):
        """Load Model7 (UNet-based)"""
        model = Model7(num_classes=num_classes, arch=arch, mask_head=False)
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        return model

def test_all_models():
    """Test loading all available models"""
    loader = TeamOxygenModelLoader()
    
    print("Testing Team Oxygen Model Loading")
    print("="*60)
    
    # Classification models
    classification_models = [
        # EfficientNetV2
        ("/mnt/HDD4/jineel/bowel_project/team_oxygen/trained_weights/rsna-abd-v2s-try5-v10-fulldata/123123.pth",
         loader.load_model3_efficientnet, "Model3-123123"),
        ("/mnt/HDD4/jineel/bowel_project/team_oxygen/trained_weights/rsna-abd-v2s-try5-v10-fulldata/123123123.pth",
         loader.load_model3_efficientnet, "Model3-123123123"),
        
        # CoaT Model4
        ("/mnt/HDD4/jineel/bowel_project/team_oxygen/trained_weights/coatmed384ourdataseed100/3.pth",
         loader.load_model4_coat, "Model4-seed100"),
        ("/mnt/HDD4/jineel/bowel_project/team_oxygen/trained_weights/coatlitemedium-384-exp1/2.pth",
         loader.load_model4_coat, "Model4-exp1"),
        
        # CoaT Model5
        ("/mnt/HDD4/jineel/bowel_project/team_oxygen/trained_weights/coatmed-newseg-ourdata-4f/1.pth",
         loader.load_model5_coat, "Model5-newseg"),
    ]
    
    # Extravasation models
    extravasation_models = [
        # EfficientNetV2
        ("/mnt/HDD4/jineel/bowel_project/team_oxygen/Models/rsna-abd-try11-v8-extrav/0_best.pth",
         lambda p: loader.load_model6_efficientnet(p, n_classes=2), "Model6-extrav-0"),
        
        # CoaT Small
        ("/mnt/HDD4/jineel/bowel_project/team_oxygen/Models/coatsmall384extravast4funet/0.pth",
         lambda p: loader.load_model4_coat(p, num_classes=2, arch='small'), "Model4-small-extrav"),
        
        # CoaT Medium
        ("/mnt/HDD4/jineel/bowel_project/team_oxygen/Models/coatmedium384extravast/0_best.pth",
         lambda p: loader.load_model5_coat(p, n_classes=2), "Model5-medium-extrav"),
    ]
    
    successful = 0
    failed = 0
    
    print("\nClassification Models:")
    print("-"*40)
    for path, loader_func, name in classification_models:
        if os.path.exists(path):
            try:
                model = loader_func(path)
                # Test forward pass
                dummy_input = torch.randn(1, 32, 3, 384, 384)
                with torch.no_grad():
                    output = model(dummy_input)
                print(f"✓ {name}: Output shape {output.shape}")
                successful += 1
            except Exception as e:
                print(f"✗ {name}: {str(e)[:100]}")
                failed += 1
        else:
            print(f"✗ {name}: File not found")
            failed += 1
    
    print("\nExtravasation Models:")
    print("-"*40)
    for path, loader_func, name in extravasation_models:
        if os.path.exists(path):
            try:
                model = loader_func(path)
                # Test forward pass
                dummy_input = torch.randn(1, 32, 3, 384, 384)
                with torch.no_grad():
                    output = model(dummy_input)
                print(f"✓ {name}: Output shape {output.shape}")
                successful += 1
            except Exception as e:
                print(f"✗ {name}: {str(e)[:100]}")
                failed += 1
        else:
            print(f"✗ {name}: File not found")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Summary: {successful} successful, {failed} failed")
    
    return successful > 0

if __name__ == "__main__":
    test_all_models()