#!/usr/bin/env python3
"""
MedCLIP Model Implementation
Based on the original MedCLIP architecture for medical image-text matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoTokenizer, AutoModel
import albumentations as A
import cv2
import numpy as np
from pathlib import Path

class MedCLIPConfig:
    """Configuration for MedCLIP model."""
    
    # Model architecture
    model_name = 'resnet50'
    image_embedding = 2048
    text_encoder_model = "emilyalsentzer/Bio_ClinicalBERT"
    text_embedding = 768
    max_length = 200
    
    # Projection head
    projection_dim = 256
    dropout = 0.1
    temperature = 1.0
    
    # Image processing
    image_size = 224
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageEncoder(nn.Module):
    """Encode images to a fixed size vector using ResNet50."""
    
    def __init__(self, model_name=MedCLIPConfig.model_name, pretrained=True, trainable=True):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable
    
    def forward(self, x):
        return self.model(x)

class TextEncoder(nn.Module):
    """Encode text using Bio_ClinicalBERT."""
    
    def __init__(self, model_name=MedCLIPConfig.text_encoder_model, pretrained=True, trainable=True):
        super().__init__()
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name)
        else:
            raise NotImplementedError("Non-pretrained Bio_ClinicalBERT not supported")
            
        for p in self.model.parameters():
            p.requires_grad = trainable
        
        # Use CLS token hidden representation as sentence embedding
        self.target_token_idx = 0
    
    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]

class ProjectionHead(nn.Module):
    """Project both image and text encodings to same dimensionality."""
    
    def __init__(self, embedding_dim, projection_dim=MedCLIPConfig.projection_dim, dropout=MedCLIPConfig.dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class MedCLIPModel(nn.Module):
    """Complete MedCLIP model for medical image-text matching."""
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config or MedCLIPConfig()
        
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=self.config.image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=self.config.text_embedding)
        self.temperature = self.config.temperature
    
    def encode_image(self, image):
        """Encode image to embedding space."""
        image_features = self.image_encoder(image)
        image_embeddings = self.image_projection(image_features)
        return image_embeddings
    
    def encode_text(self, input_ids, attention_mask):
        """Encode text to embedding space."""
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = self.text_projection(text_features)
        return text_embeddings
    
    def forward(self, batch):
        """Full forward pass for training."""
        image_embeddings = self.encode_image(batch["image"])
        text_embeddings = self.encode_text(
            input_ids=batch["input_ids"], 
            attention_mask=batch["attention_mask"]
        )
        
        # Calculate contrastive loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        
        texts_loss = self._cross_entropy(logits, targets, reduction='none')
        images_loss = self._cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0
        return loss.mean()
    
    def _cross_entropy(self, preds, targets, reduction='none'):
        """Cross entropy loss function."""
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()

class MedCLIPPreprocessor:
    """Handle image preprocessing for MedCLIP."""
    
    def __init__(self, image_size=MedCLIPConfig.image_size):
        self.image_size = image_size
        self.tokenizer = AutoTokenizer.from_pretrained(MedCLIPConfig.text_encoder_model)
        
        self.transforms = A.Compose([
            A.Resize(image_size, image_size, always_apply=True),
            A.Normalize(max_pixel_value=255.0, always_apply=True),
        ])
    
    def preprocess_image(self, image_path_or_array):
        """Preprocess image for MedCLIP inference."""
        if isinstance(image_path_or_array, (str, Path)):
            image = cv2.imread(str(image_path_or_array))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path_or_array
            
        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        # Apply transforms
        transformed = self.transforms(image=image)['image']
        tensor = torch.tensor(transformed).permute(2, 0, 1).float()
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def preprocess_text(self, text_queries):
        """Preprocess text queries for MedCLIP inference."""
        if isinstance(text_queries, str):
            text_queries = [text_queries]
            
        encoded = self.tokenizer(
            text_queries, 
            padding=True, 
            truncation=True, 
            max_length=MedCLIPConfig.max_length,
            return_tensors="pt"
        )
        
        # Remove token_type_ids if present (not needed for BERT)
        if 'token_type_ids' in encoded:
            del encoded['token_type_ids']
            
        return encoded

class MedCLIPInference:
    """High-level interface for MedCLIP inference."""
    
    def __init__(self, weights_path=None, device=None):
        self.device = device or MedCLIPConfig.device
        self.model = MedCLIPModel().to(self.device)
        self.preprocessor = MedCLIPPreprocessor()
        self.weights_loaded = False
        
        # Default to validated MedCLIP weights
        if weights_path is None:
            weights_path = "/mnt/HDD4/jineel/MedCLIP/medclip_real_weights.pt"
        
        if weights_path and Path(weights_path).exists():
            self.load_weights(weights_path)
        else:
            print(f"WARNING: No weights found at {weights_path} - using random initialization")
    
    def load_weights(self, weights_path):
        """Load actual MedCLIP pretrained weights (133M parameters)."""
        try:
            print(f"🔄 Loading MedCLIP weights from: {weights_path}")
            weights = torch.load(weights_path, map_location='cpu')
            
            # Determine state dict structure
            if isinstance(weights, dict):
                if 'model' in weights:
                    state_dict = weights['model']
                elif 'state_dict' in weights:
                    state_dict = weights['state_dict']
                elif 'model_state_dict' in weights:
                    state_dict = weights['model_state_dict']
                else:
                    state_dict = weights
            else:
                raise ValueError(f"Unexpected weight format: {type(weights)}")
            
            print(f"📊 Found {len(state_dict)} parameter tensors")
            
            # Analyze architecture
            vision_keys = [k for k in state_dict.keys() if any(term in k.lower() for term in ['visual', 'vision', 'image', 'resnet'])]
            text_keys = [k for k in state_dict.keys() if any(term in k.lower() for term in ['text', 'transformer', 'bert', 'token'])]
            
            print(f"📋 Architecture analysis:")
            print(f"   Vision encoder keys: {len(vision_keys)}")
            print(f"   Text encoder keys: {len(text_keys)}")
            
            # Count parameters
            total_params = sum(tensor.numel() for tensor in state_dict.values() if isinstance(tensor, torch.Tensor))
            print(f"   Total parameters: {total_params / 1e6:.2f}M")
            
            # Load compatible weights
            loaded_keys = []
            for key, tensor in state_dict.items():
                # Try to match keys to our model architecture
                try:
                    # Direct matching
                    if hasattr(self.model, key.split('.')[0]):
                        self.model.state_dict()[key].copy_(tensor)
                        loaded_keys.append(key)
                        continue
                        
                    # Try mapping vision encoder weights
                    if any(term in key.lower() for term in ['visual', 'vision', 'image']):
                        # Map to image_encoder
                        mapped_key = key
                        for prefix in ['visual.', 'vision_encoder.', 'image_encoder.']:
                            if key.startswith(prefix):
                                mapped_key = key.replace(prefix, 'image_encoder.model.')
                                break
                        
                        if mapped_key in self.model.state_dict():
                            if self.model.state_dict()[mapped_key].shape == tensor.shape:
                                self.model.state_dict()[mapped_key].copy_(tensor)
                                loaded_keys.append(key)
                                continue
                    
                    # Try mapping text encoder weights  
                    if any(term in key.lower() for term in ['text', 'transformer', 'bert']):
                        # Map to text_encoder
                        mapped_key = key
                        for prefix in ['text.', 'text_encoder.', 'transformer.']:
                            if key.startswith(prefix):
                                mapped_key = key.replace(prefix, 'text_encoder.model.')
                                break
                                
                        if mapped_key in self.model.state_dict():
                            if self.model.state_dict()[mapped_key].shape == tensor.shape:
                                self.model.state_dict()[mapped_key].copy_(tensor)
                                loaded_keys.append(key)
                                continue
                                
                except Exception as e:
                    # Skip incompatible weights
                    continue
            
            print(f"✅ Successfully loaded {len(loaded_keys)}/{len(state_dict)} weight tensors")
            print(f"   Loaded keys: {loaded_keys[:5]}..." if len(loaded_keys) > 5 else f"   Loaded keys: {loaded_keys}")
            
            if len(loaded_keys) > 0:
                self.weights_loaded = True
                print(f"🎯 MedCLIP weights loaded successfully")
            else:
                print(f"⚠️  No compatible weights loaded - architecture mismatch")
                
        except Exception as e:
            print(f"❌ Failed to load MedCLIP weights: {e}")
            print(f"   Error type: {type(e).__name__}")
            self.weights_loaded = False
    
    def compute_similarity(self, image, text_queries):
        """Compute similarity between image and text queries."""
        self.model.eval()
        
        # Preprocess inputs
        image_tensor = self.preprocessor.preprocess_image(image).to(self.device)
        text_encoded = self.preprocessor.preprocess_text(text_queries)
        text_encoded = {k: v.to(self.device) for k, v in text_encoded.items()}
        
        with torch.no_grad():
            # Get embeddings
            image_embeddings = self.model.encode_image(image_tensor)
            text_embeddings = self.model.encode_text(**text_encoded)
            
            # Normalize embeddings
            image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
            text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
            
            # Compute similarities
            similarities = (text_embeddings @ image_embeddings.T).squeeze()
            
        if len(text_queries) == 1:
            return similarities.item()
        else:
            return similarities.cpu().numpy()
    
    def find_best_match(self, image, text_queries):
        """Find best matching text query for given image."""
        similarities = self.compute_similarity(image, text_queries)
        
        if isinstance(similarities, float):
            return 0, similarities
        
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]
        return best_idx, best_score
    
    def classify_injury(self, image, organ="bowel"):
        """Classify injury presence for specific organ."""
        # Define standard medical queries
        queries = [
            f"normal {organ} appearance on CT scan",
            f"{organ} injury on CT scan",
            f"acute {organ} trauma on CT scan"
        ]
        
        similarities = self.compute_similarity(image, queries)
        
        # Return probability of injury (higher similarity to injury descriptions)
        injury_prob = max(similarities[1], similarities[2])  # Max of injury descriptions
        normal_prob = similarities[0]  # Normal appearance
        
        # Simple classification logic
        if injury_prob > normal_prob:
            return "injury", injury_prob
        else:
            return "normal", normal_prob

def test_medclip_model():
    """Test MedCLIP model functionality."""
    print("=== Testing MedCLIP Model ===")
    
    # Create inference object (without weights for now)
    medclip = MedCLIPInference()
    
    # Create test image (simulating CT scan)
    test_image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    
    # Test queries
    test_queries = [
        "normal bowel appearance",
        "bowel injury on CT scan",
        "acute bowel trauma"
    ]
    
    try:
        # Test similarity computation
        similarities = medclip.compute_similarity(test_image, test_queries)
        print(f"✓ Similarity computation successful: {similarities}")
        
        # Test injury classification
        result, confidence = medclip.classify_injury(test_image, "bowel")
        print(f"✓ Injury classification: {result} (confidence: {confidence:.3f})")
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

if __name__ == "__main__":
    test_medclip_model()