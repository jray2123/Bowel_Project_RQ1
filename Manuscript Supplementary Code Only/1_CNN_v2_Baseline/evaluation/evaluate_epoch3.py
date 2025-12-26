#!/usr/bin/env python3
"""
Evaluate CNN v2 Epoch 3 checkpoint (Best AUC: 0.642) with SIGMOID probability calculation.
This is the best performing checkpoint for bowel injury detection.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

# Add project paths - modify these if running from different location
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # Adjust as needed
sys.path.insert(0, str(PROJECT_ROOT / 'Traditional_Models'))
sys.path.insert(0, str(PROJECT_ROOT / 'Traditional_Models/src'))

# Paths - UPDATE THESE FOR YOUR ENVIRONMENT
TEST_IDS_PATH = "/mnt/HDD4/jineel/ratic_data/ratic_test_ids_final_verified.csv"
LABELS_PATH = "/mnt/HDD4/jineel/ratic_data/train_2024.csv"
DATA_DIRS = [
    "/mnt/HDD4/jineel/bowel_project/Traditional_Models/data/processed/images/train",
    "/mnt/HDD4/jineel/bowel_project/Traditional_Models/data/processed/images/val"
]

# Epoch 3 checkpoint (best AUC 0.642)
CHECKPOINT_PATH = Path(__file__).parent.parent / "weights" / "epoch_3_best.pth"

# Model config
CONFIG = {
    "model": {
        "feature_extractor": "efficientnet-b3",
        "feature_dim": 1536,
        "lstm_hidden_dim": 256,
        "lstm_bidirectional": True,
        "aggregation_strategy": "final",
        "position_encoding": True,
        "num_organs": 5,
        "dropout": 0.2
    }
}


def load_test_data():
    """Load test patient IDs and ground truth labels."""
    test_df = pd.read_csv(TEST_IDS_PATH)
    test_ids = test_df['patient_id'].tolist()

    labels_df = pd.read_csv(LABELS_PATH)
    labels_df = labels_df[labels_df['patient_id'].isin(test_ids)]

    labels_dict = {}
    for _, row in labels_df.iterrows():
        pid = row['patient_id']
        bowel_injury = 1 if row.get('bowel_injury', 0) > 0 else 0
        labels_dict[pid] = bowel_injury

    return test_ids, labels_dict


def load_patient_data(patient_id):
    """Load preprocessed patient data from any available directory."""
    npy_path = None
    for data_dir in DATA_DIRS:
        candidate = os.path.join(data_dir, f"{patient_id}.npy")
        if os.path.exists(candidate):
            npy_path = candidate
            break

    if npy_path is None:
        return None

    volume = np.load(npy_path)

    if volume.ndim == 3:
        volume = np.expand_dims(volume, axis=1)
        volume = np.repeat(volume, 3, axis=1)

    return volume


def sample_slices(volume, max_seq_len=48):
    """Uniformly sample slices from volume."""
    num_slices = volume.shape[0]

    if num_slices <= max_seq_len:
        if num_slices < max_seq_len:
            pad_size = max_seq_len - num_slices
            padding = np.zeros((pad_size,) + volume.shape[1:], dtype=volume.dtype)
            volume = np.concatenate([volume, padding], axis=0)
            positions = np.concatenate([
                np.linspace(0, 1, num_slices),
                np.zeros(pad_size)
            ])
            mask = np.array([True] * num_slices + [False] * pad_size)
        else:
            positions = np.linspace(0, 1, num_slices)
            mask = np.ones(num_slices, dtype=bool)
    else:
        indices = np.linspace(0, num_slices - 1, max_seq_len, dtype=int)
        volume = volume[indices]
        positions = np.linspace(0, 1, max_seq_len)
        mask = np.ones(max_seq_len, dtype=bool)

    return volume, positions, mask


def create_model(config):
    """Create CNN model."""
    from models.cnn_25d_enhanced import CNN25DEnhanced

    model_config = config['model']
    model = CNN25DEnhanced(
        feature_extractor=model_config['feature_extractor'],
        feature_dim=model_config['feature_dim'],
        lstm_hidden_dim=model_config['lstm_hidden_dim'],
        lstm_bidirectional=model_config['lstm_bidirectional'],
        aggregation_strategy=model_config['aggregation_strategy'],
        position_encoding=model_config['position_encoding'],
        num_organs=model_config['num_organs'],
        dropout=model_config['dropout']
    )
    return model


def evaluate():
    """Evaluate epoch 3 checkpoint with SIGMOID probabilities."""
    print("=" * 80)
    print("CNN v2 Epoch 3 Evaluation (Best Checkpoint)")
    print("Probability Method: SIGMOID")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load test data
    print("\nLoading test data...")
    test_ids, labels_dict = load_test_data()
    print(f"Test patients: {len(test_ids)}")
    print(f"Bowel injuries: {sum(labels_dict.values())}/{len(labels_dict)}")

    # Load model
    checkpoint_path = str(CHECKPOINT_PATH)
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return None

    model = create_model(CONFIG)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # Collect predictions
    patient_probs = {}
    patient_labels = {}

    with torch.no_grad():
        for pid in tqdm(test_ids, desc="Evaluating patients"):
            volume = load_patient_data(pid)
            if volume is None:
                continue

            volume, positions, mask = sample_slices(volume, max_seq_len=48)

            # Resize to 320x320 if needed
            if volume.shape[2] != 320 or volume.shape[3] != 320:
                from scipy.ndimage import zoom
                scale_h = 320 / volume.shape[2]
                scale_w = 320 / volume.shape[3]
                volume = zoom(volume, (1, 1, scale_h, scale_w), order=1)

            # Normalize
            volume = volume.astype(np.float32)
            if volume.max() > 1.0:
                volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

            # Convert to tensors
            slice_seq = torch.from_numpy(volume).unsqueeze(0).to(device)
            pos_enc = torch.from_numpy(positions).float().unsqueeze(0).unsqueeze(-1).to(device)
            seq_mask = torch.from_numpy(mask).unsqueeze(0).to(device)

            # Forward pass
            outputs = model(slice_seq, pos_enc, seq_mask)

            # Get bowel probability using SIGMOID (correct method)
            bowel_logits = outputs['bowel']
            bowel_prob = torch.sigmoid(bowel_logits[:, 1]).item()

            patient_probs[pid] = bowel_prob
            patient_labels[pid] = labels_dict.get(pid, 0)

    # Calculate metrics
    y_true = np.array([patient_labels[pid] for pid in patient_probs.keys()])
    y_prob = np.array([patient_probs[pid] for pid in patient_probs.keys()])

    # AUC
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5

    # Youden's J threshold optimization
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    best_threshold = thresholds[best_idx]

    # Metrics at optimal threshold
    y_pred = (y_prob >= best_threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    # Print results
    print("\n" + "=" * 60)
    print("EPOCH 3 RESULTS (Best Checkpoint)")
    print("=" * 60)
    print(f"  AUC: {auc:.4f}")
    print(f"  Optimal Threshold: {best_threshold:.4f}")
    print(f"  Sensitivity: {sensitivity:.3f}")
    print(f"  Specificity: {specificity:.3f}")
    print(f"  PPV: {ppv:.3f}")
    print(f"  NPV: {npv:.3f}")
    print(f"  F1 Score: {f1:.3f}")
    print(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"  Probability Range: [{y_prob.min():.3f}, {y_prob.max():.3f}]")

    # Save predictions
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.DataFrame([
        {'patient_id': pid, 'bowel_prob': prob}
        for pid, prob in patient_probs.items()
    ])
    predictions_df.to_csv(output_dir / "epoch3_patient_predictions.csv", index=False)
    print(f"\nSaved predictions to: {output_dir / 'epoch3_patient_predictions.csv'}")

    return {
        'auc': auc,
        'threshold': best_threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'f1': f1,
        'patient_probs': patient_probs
    }


if __name__ == "__main__":
    evaluate()
