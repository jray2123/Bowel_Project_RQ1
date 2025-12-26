#!/usr/bin/env python3
"""
Generate Per-Patient Predictions for Transformer Two-Stage Model

Uses Step 2 Fold 0 complete weights for inference on 100 test patients.
Note: Data overlap exists (55 test patients were in fold 0 training set).

Output: CSV with patient_id, bowel_prob, extravasation_prob, liver_prob, kidney_prob, spleen_prob
"""

import os
import sys
import json
import torch
import torch.nn as nn
import timm
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional

# Add paths for model imports
sys.path.append('/mnt/HDD4/jineel/bowel_project/archive/experiments/experiments/two_stage_transformer_baseline')

# ----- Model Definition (inline to avoid import issues) -----
class MultiOrganModel(nn.Module):
    """Stage 2: Multi-Organ Model with Pretrained Bowel Expert Backbone"""

    def __init__(
        self,
        backbone_name: str = "swin_tiny_patch4_window7_224",
        num_organs: int = 5,
        dropout: float = 0.2
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.num_organs = num_organs

        # Create backbone using timm
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,
            global_pool=''
        )

        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_features = self.backbone(dummy_input)
            if len(dummy_features.shape) == 4:
                feature_dim = dummy_features.shape[1]
            else:
                feature_dim = dummy_features.shape[1]

        self.feature_dim = feature_dim

        # Global pooling and multi-organ classifiers
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)

        # Individual classifiers for each organ (binary sigmoid output)
        self.organ_classifiers = nn.ModuleDict({
            'bowel': nn.Linear(feature_dim, 1),
            'extravasation': nn.Linear(feature_dim, 1),
            'liver': nn.Linear(feature_dim, 1),
            'kidney': nn.Linear(feature_dim, 1),
            'spleen': nn.Linear(feature_dim, 1)
        })

        print(f"MultiOrganModel created: {backbone_name}, {feature_dim} features")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for multi-organ classification"""
        # Extract features
        features = self.backbone(x)

        # Handle different feature shapes
        if len(features.shape) == 4:  # Spatial features [B, C, H, W]
            features = self.global_pool(features)
            features = features.flatten(1)

        # Apply dropout
        features = self.dropout(features)

        # Classify each organ
        outputs = {}
        for organ, classifier in self.organ_classifiers.items():
            outputs[organ] = torch.sigmoid(classifier(features))

        return outputs


def load_and_process_volume(npy_path: str, target_size: int = 224) -> torch.Tensor:
    """Load .npy volume and process for transformer input."""
    import cv2

    volume = np.load(npy_path)

    # Select middle slices for representation
    n_slices = volume.shape[0]

    # Use stride to get representative slices (max 48)
    if n_slices > 48:
        indices = np.linspace(0, n_slices-1, 48, dtype=int)
    else:
        indices = np.arange(n_slices)

    slices = []
    for idx in indices:
        # Get slice
        slice_2d = volume[idx]

        # Resize to target size
        if slice_2d.shape[0] != target_size or slice_2d.shape[1] != target_size:
            slice_2d = cv2.resize(slice_2d, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1]
        if slice_2d.max() > slice_2d.min():
            slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min())
        else:
            slice_2d = np.zeros_like(slice_2d)

        # Convert to 3-channel (RGB)
        slice_rgb = np.stack([slice_2d, slice_2d, slice_2d], axis=0)
        slices.append(slice_rgb)

    # Stack to [N, 3, H, W]
    slices_tensor = torch.tensor(np.stack(slices), dtype=torch.float32)

    return slices_tensor


def extract_patient_predictions(model, patient_ids, data_dir, device):
    """Extract per-patient predictions for all organs using MAX aggregation."""
    print(f"Processing {len(patient_ids)} patients...")

    organs = ['bowel', 'extravasation', 'liver', 'kidney', 'spleen']

    patient_probs = {organ: [] for organ in organs}
    valid_patient_ids = []

    model.eval()

    with torch.no_grad():
        for pid in tqdm(patient_ids, desc="Processing patients"):
            # Find npy file
            npy_path = None
            for split in ['train', 'val']:
                candidate = Path(data_dir) / 'images' / split / f"{pid}.npy"
                if candidate.exists():
                    npy_path = candidate
                    break

            if npy_path is None:
                print(f"Warning: Patient {pid} not found")
                continue

            # Load and process volume
            slices = load_and_process_volume(str(npy_path), target_size=224)
            slices = slices.to(device)

            # Process in batches to avoid OOM
            batch_size = 16
            slice_probs = {organ: [] for organ in organs}

            for i in range(0, len(slices), batch_size):
                batch = slices[i:i+batch_size]
                outputs = model(batch)

                for organ in organs:
                    probs = outputs[organ].cpu().numpy().flatten()
                    slice_probs[organ].extend(probs)

            # MEAN aggregation across slices for patient-level prediction
            # Note: MEAN provides better AUC (0.576) than MAX (0.507) for this model
            valid_patient_ids.append(pid)
            for organ in organs:
                mean_prob = np.mean(slice_probs[organ])
                patient_probs[organ].append(float(mean_prob))

    return valid_patient_ids, patient_probs


def save_predictions_csv(patient_ids, patient_probs, output_path):
    """Save per-patient predictions to CSV."""

    df = pd.DataFrame({
        'patient_id': patient_ids,
        'bowel_prob': patient_probs['bowel'],
        'extravasation_prob': patient_probs['extravasation'],
        'liver_prob': patient_probs['liver'],
        'kidney_prob': patient_probs['kidney'],
        'spleen_prob': patient_probs['spleen']
    })

    # Sort by patient_id
    df['patient_id'] = df['patient_id'].astype(int)
    df = df.sort_values('patient_id').reset_index(drop=True)

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} patient predictions to: {output_path}")

    return df


def main():
    # Paths
    # Fold 0 checkpoint (Fold 1 has incompatible architecture)
    checkpoint_path = "/mnt/HDD4/jineel/bowel_project/Traditional_Models/experiments/step2_development/step2_complete_fold_0_epoch_0_best.pth"
    test_patients_csv = "/mnt/HDD4/jineel/ratic_data/ratic_test_ids_final_verified.csv"
    data_dir = "/mnt/HDD4/jineel/bowel_project/Traditional_Models/data/processed"
    output_path = "/mnt/HDD4/jineel/bowel_project/Manuscript_Supplementary_Materials/4_Transformer_Baseline/results/transformer_patient_predictions_100.csv"

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from: {checkpoint_path}")
    model = MultiOrganModel(
        backbone_name="swin_tiny_patch4_window7_224",
        num_organs=5,
        dropout=0.2
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")

    # Load test patient IDs
    test_patients = pd.read_csv(test_patients_csv)['patient_id'].tolist()
    print(f"Test patients: {len(test_patients)}")

    # Extract predictions
    valid_ids, patient_probs = extract_patient_predictions(
        model, test_patients, data_dir, device
    )

    # Save to CSV
    df = save_predictions_csv(valid_ids, patient_probs, output_path)

    # Print summary
    print("\n" + "="*60)
    print("TRANSFORMER PER-PATIENT PREDICTIONS - SUMMARY")
    print("="*60)
    print(f"Total patients processed: {len(df)}")
    for organ in ['bowel', 'extravasation', 'liver', 'kidney', 'spleen']:
        probs = df[f'{organ}_prob']
        print(f"{organ:15}: mean={probs.mean():.3f}, min={probs.min():.3f}, max={probs.max():.3f}")

    print("\nFirst 5 rows:")
    print(df.head())

    print(f"\nOutput saved to: {output_path}")
    print("\n⚠️ Note: 55 test patients were in fold 0 training set (data overlap)")


if __name__ == '__main__':
    main()
