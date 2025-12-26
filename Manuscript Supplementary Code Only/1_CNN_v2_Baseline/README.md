# CNN Baseline

EfficientNet-B3 backbone with LSTM temporal aggregation (2.5D approach).

## Files

- `architecture/cnn_25d_enhanced.py` - Model definition
- `weights/epoch_3_best.pth` - Trained weights (173 MB)
- `evaluation/evaluate_epoch3.py` - Evaluation script
- `results/epoch3_patient_predictions_sigmoid.csv` - Predictions

## Usage

```python
import torch
import json
from architecture.cnn_25d_enhanced import CNN25DEnhanced

# Load model
with open('config/config.json', 'r') as f:
    config = json.load(f)

model = CNN25DEnhanced(config)
checkpoint = torch.load('weights/epoch_3_best.pth', map_location='cuda')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
# Input shape: [batch_size, seq_len, channels, height, width]
with torch.no_grad():
    outputs = model(image_sequences)
    bowel_prob = torch.sigmoid(outputs['bowel'])
```

## Input Requirements

- Image size: 320x320
- Sequence length: up to 48 slices
- HU windowing: Center=50, Width=400
