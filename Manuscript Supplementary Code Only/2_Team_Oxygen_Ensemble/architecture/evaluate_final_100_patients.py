#!/usr/bin/env python
"""
Final evaluation on 100 test patients with intermediate saving
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2
import pydicom
from glob import glob
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import warnings
import pickle
warnings.filterwarnings('ignore')

# Add model paths
sys.path.append('/mnt/HDD4/jineel/bowel_project/team_oxygen/Models')
from team_oxygen_model_loader import TeamOxygenModelLoader

# Paths
DATA_DIR = '/mnt/HDD4/jineel/ratic_data'
SUBSET_DIR = '/mnt/HDD4/jineel/bowel_project/test_subset'
MODEL_BASE = '/mnt/HDD4/jineel/bowel_project/team_oxygen'

def standardize_pixel_array(dcm, pixel_array):
    """Team Oxygen's Theo preprocessing"""
    if dcm.PixelRepresentation == 1:
        bit_shift = dcm.BitsAllocated - dcm.BitsStored
        dtype = pixel_array.dtype 
        pixel_array = (pixel_array << bit_shift).astype(dtype) >> bit_shift

    intercept = float(dcm.RescaleIntercept)
    slope = float(dcm.RescaleSlope)
    center = int(dcm.WindowCenter)
    width = int(dcm.WindowWidth)
    low = center - width / 2
    high = center + width / 2    
    
    pixel_array = (pixel_array * slope) + intercept
    pixel_array = np.clip(pixel_array, low, high)

    return pixel_array

def get_windowed_image(img, WL=50, WW=400):
    """Standard windowing"""
    upper, lower = WL+WW//2, WL-WW//2
    X = np.clip(img.copy(), lower, upper)
    X = X - np.min(X)
    X = X / np.max(X) if np.max(X) > 0 else X
    X = (X*255.0).astype('uint8')
    return X

def load_available_models():
    """Load all available models"""
    loader = TeamOxygenModelLoader()
    
    print("="*70)
    print("LOADING AVAILABLE MODELS")
    print("="*70)
    
    # Classification - Standard
    classification_standard = []
    paths = [
        (f"{MODEL_BASE}/trained_weights/coatmed384ourdataseed100/3.pth", "CoaT-seed100"),
        (f"{MODEL_BASE}/trained_weights/coatmed-newseg-ourdata-4f/1.pth", "CoaT-newseg"),
        (f"{MODEL_BASE}/trained_weights/coatlitemedium-384-exp1/2.pth", "CoaT-exp1-2"),
        (f"{MODEL_BASE}/trained_weights/rsna-abd-v2s-try5-v10-fulldata/123123.pth", "V2S-123123"),
        (f"{MODEL_BASE}/trained_weights/rsna-abd-v2s-try5-v10-fulldata/123123123.pth", "V2S-123123123"),
        (f"{MODEL_BASE}/trained_weights/rsna-abd-v2s-try5-v10-fulldata/123.pth", "V2S-123"),
        (f"{MODEL_BASE}/trained_weights/rsna-abd-v2s-try5-v10-fulldata/3407.pth", "V2S-3407"),
    ]
    
    print("\nClassification Models (Standard Preprocessing):")
    for path, name in paths:
        if os.path.exists(path):
            try:
                if 'v2s' in path:
                    model = loader.load_model3_efficientnet(path)
                elif 'newseg' in path:
                    model = loader.load_model5_coat(path)
                else:
                    model = loader.load_model4_coat(path)
                model.cuda()
                model.eval()
                classification_standard.append(model)
                print(f"  ✓ {name}")
            except:
                print(f"  ✗ {name}")
    
    # Classification - Theo
    classification_theo = []
    paths = [
        (f"{MODEL_BASE}/trained_weights/coatlitemedium-384-exp1/0.pth", "CoaT-exp1-0"),
        (f"{MODEL_BASE}/trained_weights/coatlitemedium-384-exp1/1.pth", "CoaT-exp1-1"),
        (f"{MODEL_BASE}/trained_weights/coatlitemedium-384-exp1/3.pth", "CoaT-exp1-3"),
    ]
    
    print("\nClassification Models (Theo Preprocessing):")
    for path, name in paths:
        if os.path.exists(path):
            try:
                model = loader.load_model4_coat(path)
                model.cuda()
                model.eval()
                classification_theo.append(model)
                print(f"  ✓ {name}")
            except:
                print(f"  ✗ {name}")
    
    # Extravasation - Theo only (standard has issues)
    extra_models_theo = []
    paths = [
        (f"{MODEL_BASE}/Models/rsna-abd-try11-v8-extrav/0_best.pth", "V2S-extrav-0", False),
        (f"{MODEL_BASE}/Models/rsna-abd-try11-v8-extrav/1_best.pth", "V2S-extrav-1", False),
        (f"{MODEL_BASE}/Models/rsna-abd-try11-v8-extrav/2_best.pth", "V2S-extrav-2", False),
        (f"{MODEL_BASE}/Models/rsna-abd-try11-v8-extrav/3_best.pth", "V2S-extrav-3", False),
    ]
    
    print("\nExtravasation Models (Theo Preprocessing):")
    for path, name, needs_swap in paths:
        if os.path.exists(path):
            try:
                model = loader.load_model6_efficientnet(path, n_classes=2)
                model.cuda()
                model.eval()
                extra_models_theo.append((model, needs_swap))
                print(f"  ✓ {name}")
            except:
                print(f"  ✗ {name}")
    
    print(f"\n{'='*70}")
    print(f"TOTAL LOADED: {len(classification_standard)} + {len(classification_theo)} classification, {len(extra_models_theo)} extravasation")
    print(f"{'='*70}\n")
    
    return classification_standard, classification_theo, extra_models_theo

def process_patient_simplified(patient_id, models):
    """Simplified patient processing"""
    classification_standard, classification_theo, extra_models_theo = models
    
    # Load images
    image_dir = Path(DATA_DIR) / 'train_images' / str(patient_id)
    if not image_dir.exists():
        return None
    
    dcm_files = sorted(glob(f"{image_dir}/*/*.dcm"))
    if len(dcm_files) < 10:
        return None
    
    # Sample slices evenly
    if len(dcm_files) > 96:
        indices = np.linspace(0, len(dcm_files)-1, 96, dtype=int)
        dcm_files = [dcm_files[i] for i in indices]
    
    # Load with both preprocessing
    volumes_standard = []
    volumes_theo = []
    
    for dcm_path in dcm_files:
        try:
            dcm = pydicom.dcmread(dcm_path)
            img_orig = dcm.pixel_array.astype(np.float32)
            
            # Standard
            img = img_orig * dcm.RescaleSlope + dcm.RescaleIntercept
            img = get_windowed_image(img)
            volumes_standard.append(cv2.resize(img, (384, 384)))
            
            # Theo
            img2 = standardize_pixel_array(dcm, img_orig.copy())
            img2 = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-6)
            if dcm.PhotometricInterpretation == "MONOCHROME1":
                img2 = 1 - img2
            img2 = (img2 * 255).astype(np.uint8)
            volumes_theo.append(cv2.resize(img2, (384, 384)))
        except Exception as e:
            continue
    
    if len(volumes_standard) < 3:
        return None
    
    # Create 2.5D tensors
    def prepare_tensor(vols):
        if not vols:
            return None
            
        while len(vols) % 3 != 0:
            vols.append(vols[-1])
        
        batch = []
        for i in range(0, len(vols)-2, 3):
            vol = np.stack([vols[i], vols[i+1], vols[i+2]], axis=-1)
            vol = vol.astype(np.float32) / 255.0
            vol = np.transpose(vol, (2, 0, 1))
            batch.append(torch.from_numpy(vol))
        
        if not batch:
            return None
            
        return torch.stack(batch).unsqueeze(0).cuda()
    
    tensor_standard = prepare_tensor(volumes_standard)
    tensor_theo = prepare_tensor(volumes_theo)
    
    if tensor_standard is None and tensor_theo is None:
        return None
    
    # Get predictions
    with torch.no_grad():
        # Standard classification
        preds_standard = []
        if tensor_standard is not None:
            for model in classification_standard:
                try:
                    out = model(tensor_standard).sigmoid()
                    # Models output [1, num_windows, num_classes], take max across windows
                    out = out.squeeze(0).max(dim=0)[0] if out.dim() > 2 else out.squeeze()
                    preds_standard.append(out.cpu().numpy())
                except Exception as e:
                    pass
        
        # Theo classification
        preds_theo = []
        if tensor_theo is not None:
            for model in classification_theo:
                try:
                    out = model(tensor_theo).sigmoid()
                    # Models output [1, num_windows, num_classes], take max across windows
                    out = out.squeeze(0).max(dim=0)[0] if out.dim() > 2 else out.squeeze()
                    preds_theo.append(out.cpu().numpy())
                except Exception as e:
                    pass
        
        # Theo extravasation
        preds_extrav = []
        if tensor_theo is not None:
            for model, needs_swap in extra_models_theo:
                try:
                    out = model(tensor_theo).sigmoid()
                    # Models output [1, num_windows, num_classes], take max across windows
                    out = out.squeeze(0).max(dim=0)[0] if out.dim() > 2 else out.squeeze()
                    preds_extrav.append(out.cpu().numpy())
                except Exception as e:
                    pass
    
    if not preds_standard and not preds_theo:
        return None
    
    # Average predictions
    pred_standard = np.stack(preds_standard).mean(axis=0).squeeze() if preds_standard else np.zeros(10)
    pred_theo = np.stack(preds_theo).mean(axis=0).squeeze() if preds_theo else np.zeros(10)
    pred_extrav = np.stack(preds_extrav).mean(axis=0).squeeze() if preds_extrav else np.zeros(2)
    
    # Combine
    if np.any(pred_standard > 0) and np.any(pred_theo > 0):
        pred_combined = (pred_standard * 0.5) + (pred_theo * 0.5)
    else:
        pred_combined = pred_standard if np.any(pred_standard > 0) else pred_theo
    
    # Bowel score (simplified without standard extravasation)
    pred_combined[9] = (pred_combined[9] * 0.85) + (pred_extrav[1] * 0.15)
    
    return pred_combined, pred_extrav

def main():
    """Main evaluation"""
    
    # Load models
    models = load_available_models()
    
    # Load test data
    patient_ids = pd.read_csv('/mnt/HDD4/jineel/bowel_project/ratic_test_ids_final_verified.csv')['patient_id'].values
    labels_df = pd.read_csv(f'{SUBSET_DIR}/patient_labels.csv')
    
    # Check for intermediate results
    intermediate_file = 'intermediate_results.pkl'
    if os.path.exists(intermediate_file):
        print(f"\nLoading intermediate results from {intermediate_file}")
        with open(intermediate_file, 'rb') as f:
            saved_data = pickle.load(f)
        predictions = saved_data['predictions']
        labels = saved_data['labels']
        processed_ids = saved_data['processed_ids']
        print(f"Resuming from patient {len(processed_ids)}")
    else:
        predictions = []
        labels = []
        processed_ids = set()
    
    # Process patients
    print("Processing patients...")
    try:
        for patient_id in tqdm(patient_ids):
            if patient_id in processed_ids:
                continue
                
            result = process_patient_simplified(patient_id, models)
            
            if result is None:
                continue
            
            pred_class, pred_extrav = result
            
            # Apply weights
            bowel_w = 1.75
            extrav_w = 8
            low_w = 1.75
            high_w = 3.5
            
            # Create prediction dict
            pred = {
                'patient_id': patient_id,
                'bowel_injury': pred_class[9] * bowel_w,
                'extravasation_injury': pred_extrav[1] * extrav_w,
                'liver_healthy': 1 - pred_class[0],
                'liver_injury': pred_class[3] * low_w + pred_class[4] * high_w,
                'spleen_healthy': 1 - pred_class[1],
                'spleen_injury': pred_class[5] * low_w + pred_class[6] * high_w,
                'kidney_healthy': 1 - pred_class[2],
                'kidney_injury': pred_class[7] * low_w + pred_class[8] * high_w,
            }
            predictions.append(pred)
            
            # Get labels
            label = labels_df[labels_df['patient_id'] == patient_id].iloc[0]
            labels.append({
                'patient_id': patient_id,
                'bowel_injury': label['bowel_injury'],
                'extravasation_injury': label['extravasation_injury'],
                'liver_injury': int(label['liver_low'] + label['liver_high'] > 0),
                'spleen_injury': int(label['spleen_low'] + label['spleen_high'] > 0),
                'kidney_injury': int(label['kidney_low'] + label['kidney_high'] > 0),
            })
            
            processed_ids.add(patient_id)
            
            # Save intermediate results every 10 patients
            if len(processed_ids) % 10 == 0:
                with open(intermediate_file, 'wb') as f:
                    pickle.dump({
                        'predictions': predictions,
                        'labels': labels,
                        'processed_ids': processed_ids
                    }, f)
                print(f"\nSaved intermediate results ({len(processed_ids)} patients)")
                
    except KeyboardInterrupt:
        print("\nInterrupted! Saving intermediate results...")
        with open(intermediate_file, 'wb') as f:
            pickle.dump({
                'predictions': predictions,
                'labels': labels,
                'processed_ids': processed_ids
            }, f)
        print(f"Saved {len(processed_ids)} processed patients")
        return
    
    # Convert to dataframes
    pred_df = pd.DataFrame(predictions)
    true_df = pd.DataFrame(labels)
    
    print(f"\nProcessed {len(pred_df)} patients successfully")
    
    # Calculate metrics
    print("\n" + "="*70)
    print("EVALUATION METRICS")
    print("="*70)
    
    metrics = {}
    for organ in ['bowel', 'extravasation', 'liver', 'spleen', 'kidney']:
        col = f'{organ}_injury'
        if col in true_df.columns and col in pred_df.columns:
            y_true = true_df[col].values
            y_pred = pred_df[col].values
            
            if len(np.unique(y_true)) > 1:
                auc = roc_auc_score(y_true, y_pred)
                metrics[organ] = auc
                n_pos = y_true.sum()
                n_total = len(y_true)
                print(f"\n{organ.upper()}")
                print(f"  AUC: {auc:.4f}")
                print(f"  Positive samples: {n_pos}/{n_total} ({n_pos/n_total*100:.1f}%)")
            else:
                print(f"\n{organ.upper()}: Not enough positive samples")
    
    # Plot ROC curves
    if metrics:
        plt.figure(figsize=(10, 8))
        for organ, auc in metrics.items():
            col = f'{organ}_injury'
            y_true = true_df[col].values
            y_pred = pred_df[col].values
            
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            plt.plot(fpr, tpr, label=f'{organ.capitalize()} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Team Oxygen Models (100 Test Patients)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('roc_curves_100_patients.png')
        print("\nROC curves saved to roc_curves_100_patients.png")
    
    # Save predictions
    pred_df.to_csv('final_predictions_100_patients.csv', index=False)
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Models used:")
    print(f"  - {len(models[0])} standard classification")
    print(f"  - {len(models[1])} theo classification") 
    print(f"  - {len(models[2])} extravasation")
    print(f"\nLimitations:")
    print(f"  - Missing 4/14 classification models")
    print(f"  - Missing ALL standard extravasation models")
    print(f"  - Bowel score missing 15% contribution from standard extravasation")
    
    if 'bowel' in metrics:
        print(f"\nBowel AUC: {metrics['bowel']:.4f} (expected ~0.65-0.70 with all models)")
    
    # Clean up intermediate file
    if os.path.exists(intermediate_file):
        os.remove(intermediate_file)
        print(f"\nCleaned up intermediate file")

if __name__ == "__main__":
    main()