# Supplementary Materials - Bowel Injury Detection

## Setup

```bash
conda create -n bowel_env python=3.8
conda activate bowel_env
pip install -r requirements.txt
```

## Directory Structure

```
├── 1_CNN_v2_Baseline/          # EfficientNet-B3 + LSTM
├── 2_Team_Oxygen_Ensemble/     # CoaT + EfficientNetV2 ensemble
├── 3_MedCLIP_Foundation/       # Zero-shot vision-language model
├── 4_Transformer_Baseline/     # Swin Transformer
├── test_dataset/               # 100-patient test set
└── manuscript_figures/         # Analysis scripts and outputs
```

## Model Weights

Download weights separately and place in each model's `weights/` directory.

## Test Dataset

- `test_dataset/ratic_test_ids_final_verified.csv` - Patient IDs
- `test_dataset/ground_truth_labels.csv` - Ground truth labels

## Usage

See README.md in each model directory for specific instructions.
