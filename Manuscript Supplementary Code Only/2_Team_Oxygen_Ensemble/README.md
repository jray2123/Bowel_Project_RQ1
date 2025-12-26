# Team Oxygen Ensemble

Multi-model ensemble combining CoaT and EfficientNetV2 architectures.

## Files

- `architecture/models.py` - Core model classes
- `architecture/team_oxygen_model_loader.py` - Ensemble loading utilities
- `architecture/evaluate_final_100_patients.py` - Evaluation script
- `weights/` - 26 model checkpoints (~4.1 GB total)
- `results/final_predictions_100_patients.csv` - Predictions

## Weight Structure

```
weights/
├── coat_medium_exp1/       # 4 folds
├── coat_medium_seed100/    # 1 fold
├── coat_newseg_4fold/      # 4 folds
├── efficientnet_v2s/       # 4 seeds
└── extravasation_models/   # 13 models
```

## Usage

```python
import torch
from architecture.team_oxygen_model_loader import load_team_oxygen_ensemble

device = torch.device('cuda')
ensemble = load_team_oxygen_ensemble('weights/', device)

# Inference
for model_name, model in ensemble.items():
    with torch.no_grad():
        pred = model(images)
```

## Evaluation

```bash
python architecture/evaluate_final_100_patients.py
```
