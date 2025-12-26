# MedCLIP Foundation Model

Zero-shot classification using a CLIP-style vision-language model with 5-query ensemble.

## Files

- `architecture/medclip_model.py` - Model wrapper
- `weights/pytorch_model.bin` - Pre-trained weights (509 MB)
- `evaluation/evaluation_utils.py` - Evaluation utilities
- `results/medclip_patient_predictions_100.csv` - Predictions

## Ensemble Queries

```python
QUERIES = [
    "bowel wall discontinuity visible on CT scan",
    "transmural bowel injury with spillage",
    "evidence of bowel injury on abdominal CT",
    "grade 4 bowel injury with full thickness tear",
    "ligamentum teres sign suggesting upper bowel perforation"
]
```

## Usage

```python
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor

# Load model
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.from_pretrained('weights/')
model.eval()

processor = MedCLIPProcessor()

# Single query inference
image = Image.open('ct_slice.png').convert('RGB')
inputs = processor(images=image, return_tensors='pt')
text_inputs = processor(text="bowel wall discontinuity visible on CT scan", return_tensors='pt')

with torch.no_grad():
    image_features = model.encode_image(inputs['pixel_values'])
    text_features = model.encode_text(text_inputs['input_ids'])
    similarity = (image_features @ text_features.T).item()
```

## Ensemble Method

1. Run each query against the image
2. Apply per-query threshold for binary prediction
3. Majority vote: positive if >=4/5 queries agree
4. Patient-level: MAX aggregation across slices
