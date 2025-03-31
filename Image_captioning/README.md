# Assignment-3: Image Caption Consistency

**Authors**:  
Goda Nagakalyani(214050010) 
**Date**: 06 May 2022  

---

## Problem Statement
Develop a two-stage DNN system combining CNN (for image processing), RNN/Transformer (for caption processing), and FFNN (to evaluate image-caption consistency). The system should output a value between 0 (inconsistent) and 1 (consistent).

**Example**:  
- Consistent: Image of "tiger chasing deer" with caption "wild predator hunting prey"  
- Inconsistent: Same image with caption "peaceful nature scene"  

**Dataset**: [MS-COCO 2017](https://cocodataset.org/#home)  
- Positive samples: 25,014  
- Negative samples: 25,014 (created using BERT embeddings' lowest cosine similarity)  

---

## System Architecture
### 1. Feed-Forward Neural Network (FFNN)
- **Structure**:
- Image Encoding (2048) → 5-layer FFN → [2048,1500,1024,768,512]
- Caption Encoding (512) → 4-layer FFN → [512,512,512,512]
- Merged → 10-layer FFN → [1400,1400,...,20,2] → Softmax  
- **Hyperparameters**:  
- Learning rate: 1e-4 (Adam: 1e-3)  

### 2. CNN (ResNet152)
- **Key Features**:  
- 152-layer deep residual network  
- Identity shortcut connections  
- Convolution pooling + classifier head  

### 3. RNN/Transformer (CLIP ViT-B/32)
- **Text Transformer**:  
- Layers: 12  
- Embedding dim: 512  
- Heads: 8  
- **Hyperparameters**:  
- Learning rate: 5e-4  
- Batch size: 32,768  

---

## Training Details
- **Epochs**: 200  
- **Early Stopping**: Based on validation loss  
- **Optimizer**: Adam (β1=0.9, β2=0.999/0.98, ε=1e-8/1e-6)  

---

## Performance Metrics
| Metric          | Value |
|-----------------|-------|
| True Positives  | 39    |
| True Negatives  | 43    |
| False Positives | 7     |
| False Negatives | 11    |

**Accuracy**: ~82%  

---

## Error Analysis
### False Negatives
- **Example**:  
- *Caption*: "Girl blowing out a candle on an ice-cream"  
- *Actual*: 1 (consistent), *Predicted*: 0  

### False Positives
- **Example**:  
- *Caption*: "Young boy barefoot holding umbrella touching cow horn"  
- *Actual*: 0 (inconsistent), *Predicted*: 1  

**Key Insight**:  
- Low accuracy may stem from ambiguous ground truth labels.  
- Model confusion occurs with semantically similar captions (e.g., cosine similarity = 0.503).  

---

## Good Predictions
### True Positives
- "Food cooks in a pot on a stove" ↔ "Food sits in a pot in a kitchen"  

### True Negatives
- "Woman in a room with a cat" ↔ "Group using computers"  

---

## References
1. [MS-COCO Dataset](https://arxiv.org/pdf/1405.0312.pdf)  
2. [CLIP: Contrastive Language-Image Pre-training](https://arxiv.org/abs/2103.00020)  
3. [ResNet Architecture](https://arxiv.org/abs/1512.03385)
