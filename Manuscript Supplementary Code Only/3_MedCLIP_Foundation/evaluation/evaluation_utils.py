#!/usr/bin/env python3
"""
MedCLIP Evaluation Utilities
============================

Shared utility functions for MedCLIP evaluation scripts to eliminate code duplication.
Contains common functions for:
- DICOM image loading and preprocessing
- Patient slice management
- Model evaluation with slice-level aggregation
- Threshold optimization for F1 score
- Metrics calculation (AUC, precision, recall, etc.)

Usage:
    from evaluation_utils import (
        load_and_preprocess_image,
        get_patient_slices,
        evaluate_patient_with_query,
        optimize_threshold_for_f1,
        calculate_auc_scores
    )
"""

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, f1_score,
    precision_recall_curve, auc
)
import logging
import sys

# MedCLIP import - will be imported when needed in initialize_medclip
# This avoids import errors when evaluation_utils is imported but MedCLIP isn't needed

logger = logging.getLogger(__name__)


def load_and_preprocess_image(patient_id: int, series_id: int, instance_number: int, data_dir: Path):
    """
    Load and preprocess a DICOM slice for MedCLIP evaluation.
    
    Args:
        patient_id (int): Patient identifier
        series_id (int): DICOM series identifier  
        instance_number (int): DICOM instance number
        data_dir (Path): Root data directory containing train_images
        
    Returns:
        PIL.Image or None: Preprocessed RGB image ready for MedCLIP, or None if loading fails
        
    Note:
        Uses min-max normalization to convert DICOM pixel values to 8-bit RGB.
        Consider applying medical windowing for optimal tissue contrast.
    """
    try:
        dicom_path = data_dir / "train_images" / str(patient_id) / str(series_id) / f"{instance_number}.dcm"
        
        if not dicom_path.exists():
            return None
            
        dicom = pydicom.dcmread(dicom_path)
        pixel_array = dicom.pixel_array
        
        # Convert to 8-bit using min-max normalization
        pixel_array = pixel_array.astype(np.float32)
        if pixel_array.max() > pixel_array.min():
            pixel_array = ((pixel_array - pixel_array.min()) / 
                          (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
        else:
            pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
        
        # Convert to RGB (MedCLIP expects 3-channel input)
        image = Image.fromarray(pixel_array).convert('RGB')
        return image
        
    except Exception as e:
        logger.debug(f"Failed to load image {patient_id}/{series_id}/{instance_number}: {e}")
        return None


def get_patient_slices(patient_id: int, data_dir: Path) -> list:
    """
    Get all DICOM slice coordinates for a patient.
    
    Args:
        patient_id (int): Patient identifier
        data_dir (Path): Root data directory containing train_images
        
    Returns:
        list: List of tuples (patient_id, series_id, instance_number) sorted by series and instance
        
    Note:
        Returns empty list if patient directory doesn't exist.
        Slices are sorted by (series_id, instance_number) for consistent ordering.
    """
    patient_dir = data_dir / "train_images" / str(patient_id)
    slices = []
    
    if not patient_dir.exists():
        return slices
        
    for series_dir in patient_dir.iterdir():
        if series_dir.is_dir():
            series_id = int(series_dir.name)
            for dicom_file in series_dir.glob('*.dcm'):
                instance_number = int(dicom_file.stem)
                slices.append((patient_id, series_id, instance_number))
    
    return sorted(slices, key=lambda x: (x[1], x[2]))


def evaluate_patient_with_query(model, patient_id: int, query: str, data_dir: Path):
    """
    Evaluate a patient with a query using slice-level max aggregation.
    
    Args:
        model: MedCLIP model instance (should have predict method)
        patient_id (int): Patient identifier
        query (str): Text query for medical image evaluation
        data_dir (Path): Root data directory containing train_images
        
    Returns:
        float or None: Maximum similarity score across all patient slices, or None if no valid slices
        
    Note:
        Uses max aggregation across all slices - the highest scoring slice determines patient-level score.
        This is the validated methodology for patient-level evaluation from slice-level predictions.
    """
    slices = get_patient_slices(patient_id, data_dir)
    
    if not slices:
        return None
    
    slice_scores = []
    
    for patient_id_slice, series_id, instance_number in slices:
        image = load_and_preprocess_image(patient_id_slice, series_id, instance_number, data_dir)
        
        if image is not None:
            try:
                prediction = model.predict(image, [query])
                score = prediction['max_similarity']
                slice_scores.append(score)
            except Exception as e:
                logger.debug(f"Prediction failed for {patient_id}/{series_id}/{instance_number}: {e}")
                continue
    
    if not slice_scores:
        return None
        
    # Use max aggregation (validated methodology)
    return max(slice_scores)


def evaluate_patient_with_query_detailed(model, patient_id: int, query: str, data_dir: Path):
    """
    Enhanced version that returns both aggregated score AND all slice scores.
    
    Args:
        model: MedCLIP model instance
        patient_id (int): Patient identifier
        query (str): Text query for medical image evaluation
        data_dir (Path): Root data directory containing train_images
        
    Returns:
        tuple: (max_score, slice_scores) where:
            - max_score (float or None): Maximum similarity score across slices
            - slice_scores (list): All individual slice similarity scores
            
    Note:
        Useful for aggregation strategy analysis and slice-level data collection.
    """
    slices = get_patient_slices(patient_id, data_dir)
    
    if not slices:
        return None, []
    
    slice_scores = []
    
    for patient_id_slice, series_id, instance_number in slices:
        image = load_and_preprocess_image(patient_id_slice, series_id, instance_number, data_dir)
        
        if image is not None:
            try:
                prediction = model.predict(image, [query])
                score = prediction['max_similarity']
                slice_scores.append(score)
            except Exception as e:
                logger.debug(f"Prediction failed for {patient_id}/{series_id}/{instance_number}: {e}")
                continue
    
    if not slice_scores:
        return None, []
        
    # Return both max aggregation AND all slice scores
    max_score = max(slice_scores)
    return max_score, slice_scores


def optimize_threshold_for_f1(y_true, y_scores):
    """
    Find threshold that maximizes F1 score using exhaustive search.
    
    Args:
        y_true (array-like): True binary labels (0/1)
        y_scores (array-like): Prediction scores (higher = more positive)
        
    Returns:
        dict or None: Dictionary with optimal metrics:
            - threshold: Optimal threshold value
            - f1: F1 score at optimal threshold
            - sensitivity: True positive rate (recall)
            - specificity: True negative rate
            - precision: Positive predictive value
            - npv: Negative predictive value
            - tp, tn, fp, fn: Confusion matrix values
            
    Note:
        Tests all unique score values as potential thresholds.
        Prediction logic: score >= threshold → positive prediction.
        Returns None if optimization fails (e.g., all same class).
    """
    thresholds = np.unique(y_scores)
    best_f1 = 0
    best_metrics = None
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        
        if len(np.unique(y_pred)) < 2:
            continue
            
        try:
            f1 = f1_score(y_true, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                
                # Calculate all metrics
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                
                best_metrics = {
                    'threshold': threshold,
                    'f1': f1,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'precision': precision,
                    'npv': npv,
                    'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
                }
        except (ValueError, ZeroDivisionError) as e:
            logger.debug(f"Skipping threshold {threshold} due to metric calculation error: {e}")
            continue
    
    return best_metrics


def optimize_threshold_for_f1_corrected(y_true, y_scores, query_name):
    """
    Enhanced threshold optimization with support for control queries (inverted logic).
    
    Args:
        y_true (array-like): True binary labels (0/1)
        y_scores (array-like): Prediction scores
        query_name (str): Query text to determine if it's a control query
        
    Returns:
        dict or None: Same as optimize_threshold_for_f1 plus:
            - is_control_query: Boolean indicating if control query logic was used
            - threshold_logic: String describing logic type ('normal' or 'inverted')
            
    Note:
        Control queries (containing 'normal', 'intact', 'no evidence') use inverted logic:
        high similarity = normal finding → score < threshold = injured (inverted prediction)
    """
    def is_control_query(query):
        """Determine if a query describes normal/control findings."""
        control_keywords = ['normal', 'intact', 'no evidence']
        return any(keyword in query.lower() for keyword in control_keywords)
    
    thresholds = np.unique(y_scores)
    best_f1 = 0
    best_metrics = None
    is_control = is_control_query(query_name)
    
    logger.debug(f"Query type: {'CONTROL' if is_control else 'INJURY'} - "
                f"Using {'inverted' if is_control else 'normal'} threshold logic")
    
    for threshold in thresholds:
        # Apply appropriate threshold logic
        if is_control:
            y_pred = (y_scores < threshold).astype(int)  # INVERTED for control queries
        else:
            y_pred = (y_scores >= threshold).astype(int)  # NORMAL for injury queries
        
        if len(np.unique(y_pred)) < 2:
            continue
            
        try:
            f1 = f1_score(y_true, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                
                # Calculate all metrics
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                
                best_metrics = {
                    'threshold': threshold,
                    'f1': f1,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'precision': precision,
                    'npv': npv,
                    'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
                    'is_control_query': is_control,
                    'threshold_logic': 'inverted' if is_control else 'normal'
                }
        except (ValueError, ZeroDivisionError) as e:
            logger.debug(f"Skipping threshold {threshold} due to metric calculation error: {e}")
            continue
    
    return best_metrics


def calculate_auc_scores(y_true, y_scores):
    """
    Calculate AUC-ROC and AUC-PR scores.
    
    Args:
        y_true (array-like): True binary labels (0/1)
        y_scores (array-like): Prediction scores
        
    Returns:
        tuple: (auc_roc, auc_pr) where:
            - auc_roc (float): Area under ROC curve
            - auc_pr (float): Area under precision-recall curve
            
    Note:
        Returns (0.5, 0.5) if calculation fails (e.g., single class).
    """
    try:
        auc_roc = roc_auc_score(y_true, y_scores)
    except (ValueError, ZeroDivisionError):
        auc_roc = 0.5
    
    try:
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
        auc_pr = auc(recall_vals, precision_vals)
    except (ValueError, ZeroDivisionError):
        auc_pr = 0.5
        
    return auc_roc, auc_pr


def initialize_medclip():
    """
    Initialize MedCLIP model with error handling and correct paths.
    
    Returns:
        MedCLIPOfficial: Loaded MedCLIP model instance
        
    Raises:
        Exception: If model loading fails
        
    Note:
        Loads the pretrained MedCLIP model from the reorganized location.
        Includes comprehensive error handling and logging.
    """
    logger.info("🤖 Initializing MedCLIP model...")
    
    try:
        # Import REAL MedCLIP implementation (not dummy!)
        sys.path.append('/mnt/HDD4/jineel/bowel_project/Foundation_Models/MedCLIP/utils')
        from medclip_official import MedCLIPOfficial
        
        # Use official MedCLIP with real learned features
        model = MedCLIPOfficial()
        model.load_pretrained()
        logger.info("✅ MedCLIP model loaded successfully from reorganized path")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load MedCLIP: {e}")
        raise


def load_test_patient_labels(labels_path=None):
    """
    Load test patient labels with multi-organ annotations.
    
    Args:
        labels_path (str or Path, optional): Path to labels CSV file.
            Defaults to standard location if not provided.
            
    Returns:
        tuple: (patients, organ_labels) where:
            - patients (list): List of patient IDs
            - organ_labels (dict): Dictionary mapping organ names to label lists
            
    Note:
        Standard organs: liver, kidney, spleen, extravasation, bowel
        Logs case counts and prevalence for each organ.
    """
    if labels_path is None:
        labels_path = Path("/mnt/HDD4/jineel/bowel_project/Foundation_Models/MedCLIP/final_results/test_patient_labels.csv")
    
    logger.info("📋 Loading test patient labels...")
    
    labels_df = pd.read_csv(labels_path)
    patients = labels_df['patient_id'].tolist()
    
    # Extract data for each organ
    organ_labels = {}
    standard_organs = ['liver', 'kidney', 'spleen', 'extravasation', 'bowel']
    
    for organ in standard_organs:
        if organ in labels_df.columns:
            organ_labels[organ] = labels_df[organ].astype(int).tolist()
            positive_cases = sum(organ_labels[organ])
            total_cases = len(organ_labels[organ])
            logger.info(f"✅ {organ.upper()}: {positive_cases}/{total_cases} positive cases "
                       f"({positive_cases/total_cases:.1%})")
    
    return patients, organ_labels


def apply_medical_windowing(pixel_array, window_center=50, window_width=400):
    """
    Apply medical windowing to DICOM pixel data for better tissue contrast.
    
    Args:
        pixel_array (np.ndarray): Raw DICOM pixel values
        window_center (int): Window center (typically 50 for soft tissue)
        window_width (int): Window width (typically 400 for abdominal CT)
        
    Returns:
        np.ndarray: Windowed pixel array
        
    Note:
        Standard abdominal soft tissue window is WC=50, WW=400.
        This may provide better tissue contrast than simple min-max normalization.
    """
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    windowed_array = np.clip(pixel_array, min_val, max_val)
    return windowed_array


def evaluate_single_query_complete(model, query, patients, labels, data_dir, organ_name="unknown"):
    """
    Complete evaluation of a single query with comprehensive metrics and error handling.
    
    Args:
        model: MedCLIP model instance
        query (str): Text query for evaluation
        patients (list): List of patient IDs
        labels (list): List of binary labels (0/1)
        data_dir (Path): Root data directory
        organ_name (str): Name of organ being evaluated (for logging)
        
    Returns:
        dict or None: Complete evaluation result with all metrics, or None if evaluation fails
        
    Note:
        Includes progress logging and comprehensive error handling.
        Calculates both AUC scores and optimal F1 metrics.
    """
    logger.info(f"🔍 {organ_name.upper()}: {query[:60]}...")
    
    try:
        scores = []
        valid_labels = []
        valid_patients = []
        
        logger.info(f"   Processing {len(patients)} patients...")
        for i, (patient_id, label) in enumerate(zip(patients, labels), 1):
            if i % 10 == 0 or i == 1:  # Progress logging
                logger.info(f"   Patient {i}/{len(patients)} (ID: {patient_id})...")
            
            score = evaluate_patient_with_query(model, patient_id, query, data_dir)
            
            if score is not None:
                scores.append(score)
                valid_labels.append(label)
                valid_patients.append(patient_id)
        
        if len(scores) == 0:
            logger.error("No valid similarity scores obtained")
            return None
        
        scores = np.array(scores)
        valid_labels = np.array(valid_labels)
        
        logger.info(f"✅ Obtained {len(scores)} valid similarity scores")
        
        # Calculate AUC scores
        auc_roc, auc_pr = calculate_auc_scores(valid_labels, scores)
        
        # Optimize threshold for F1
        optimal_metrics = optimize_threshold_for_f1(valid_labels, scores)
        
        if optimal_metrics is not None:
            # Combine results
            final_result = {
                'organ': organ_name,
                'query': query,
                'auc_roc': auc_roc,
                'auc_pr': auc_pr,
                'n_patients': len(valid_labels),
                'patients': valid_patients,
                'labels': valid_labels.tolist(),
                'scores': scores.tolist(),
                **optimal_metrics
            }
            
            logger.info(f"✅ F1={final_result['f1']:.3f}, "
                       f"Sens={final_result['sensitivity']:.3f}, "
                       f"Spec={final_result['specificity']:.3f}, "
                       f"AUC={final_result['auc_roc']:.3f}")
            
            return final_result
        else:
            logger.warning(f"⚠️  Failed to optimize threshold for {organ_name} query")
            return None
        
    except Exception as e:
        logger.error(f"❌ Error evaluating {organ_name} query: {e}")
        return None


# Version information
__version__ = "1.0.0"
__author__ = "MedCLIP Evaluation Team"
__description__ = "Shared utilities for MedCLIP medical image evaluation"