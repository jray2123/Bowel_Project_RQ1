# Transformer Baseline

Two-stage Swin Transformer Tiny for multi-organ injury detection.

## Files

- `architecture/pure_transformer_model.py` - Model definition
- `weights/step2_fold1_best.pth` - Trained weights (107 MB)
- `evaluation/generate_transformer_patient_predictions.py` - Evaluation script
- `results/transformer_patient_predictions_fold1.csv` - Predictions

## Usage

```python
import torch
from architecture.pure_transformer_model import PureTransformerMultiOrgan

device = torch.device('cuda')

# Load model (requires step1 checkpoint path for initialization)
model = PureTransformerMultiOrgan(step1_path, device)
checkpoint = torch.load('weights/step2_fold1_best.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
# Input shape: [batch_size, 3, 224, 224]
with torch.no_grad():
    predictions = model(images)
    # predictions: dict with 'bowel', 'liver', 'kidney', 'spleen', 'extravasation'
    bowel_prob = torch.sigmoid(predictions['bowel'])
```

## Patient-Level Aggregation

```python
# Use MEAN aggregation across slices
import numpy as np
slice_probs = [model_predict(slice) for slice in patient_slices]
patient_prob = np.mean(slice_probs)
```

## Evaluation

```bash
python evaluation/generate_transformer_patient_predictions.py
```
