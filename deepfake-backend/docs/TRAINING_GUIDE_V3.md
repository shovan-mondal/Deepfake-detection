# Deepfake Detector V3 - Training Guide

## 🎯 Target Performance
- **Accuracy**: >90%
- **AUC-ROC**: >0.90
- **F1-Score**: >0.88

## 🏗️ Architecture Overview

### Dual-Stream Gated Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input (224x224x3)                         │
└─────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
┌─────────────────────────┐     ┌─────────────────────────────┐
│   Stream A: Visual      │     │   Stream B: Forensic         │
│   (MobileNetV2)         │     │   (SRM + Bayar Filters)      │
│   - Semantic features   │     │   - Noise residuals          │
│   - 1280-D output       │     │   - High-freq artifacts      │
└─────────────────────────┘     └─────────────────────────────┘
              │                           │
              │         ┌─────────────────┘
              │         │ (Gate Vector: sigmoid)
              ▼         ▼
┌─────────────────────────────────────────────────────────────┐
│              Gated Fusion: visual × gate                     │
│              + Skip connection (concat)                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Classification Head                             │
│         Dense(512) → BN → Dropout(0.3)                      │
│         Dense(256) → BN → Dropout(0.3)                      │
│         Dense(1) → Sigmoid                                   │
└─────────────────────────────────────────────────────────────┘
```

### Key V3 Improvements

1. **Extended Forensic Filter Bank** (5 filters instead of 3):
   - KV (Ker-Vass): Second-order residuals
   - SPAM: Edge detection
   - MinMax: Center-surround contrast
   - **Bayar**: Constrained high-pass (NEW - great for GANs)
   - **Laplacian**: Second derivative (NEW - great for diffusion)

2. **Learnable Bayar Convolution**: Adaptive forensic features with high-pass constraint

3. **Channel Attention**: Squeeze-and-Excitation on visual features

4. **Mixed Precision Training**: 2x speedup on T4 GPU

5. **Cosine Learning Rate with Warmup**: Better convergence

6. **Label Smoothing (0.1)**: Better calibrated probabilities

---

## 🚀 Training on Google Colab T4

### Step 1: Setup Environment

```python
# Mount Google Drive (to save models)
from google.colab import drive
drive.mount('/content/drive')

# Clone or upload your project
# Option A: Clone from GitHub
!git clone https://github.com/YOUR_USERNAME/deepfake-image-detection.git
%cd deepfake-image-detection

# Option B: Upload from Drive
# !cp -r /content/drive/MyDrive/deepfake-image-detection /content/

# Install dependencies
!pip install -q tensorflow matplotlib seaborn scikit-learn tqdm
```

### Step 2: Verify GPU

```python
import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Should show: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### Step 3: Upload/Prepare Data

```python
# If data is on Google Drive
!mkdir -p data/processed
!cp -r /content/drive/MyDrive/YOUR_DATA_FOLDER/* data/processed/

# Verify data structure
!find data/processed -type d
# Should show: train/real, train/fake, val/real, val/fake, test/real, test/fake
```

### Step 4: Phase 1 Training (Frozen Backbone)

```python
import sys
sys.path.insert(0, '.')

from src.models.train_model import train_model_v3

# Train Phase 1
model, history = train_model_v3(
    data_dir="data/processed",
    batch_size=64,           # T4 can handle 64 with mixed precision
    epochs=15,
    model_save_path="models/deepfake_detector_v3.keras"
)
```

**Expected Results (Phase 1):**
- Val Accuracy: ~85-88%
- Val AUC: ~0.88-0.92

### Step 5: Phase 2 Fine-Tuning (Progressive Unfreezing)

```python
from src.models.fine_tune import fine_tune_v3

# Run full 3-stage fine-tuning
model, all_history = fine_tune_v3(
    model_path="models/deepfake_detector_v3_best.keras",
    data_dir="data/processed",
    output_path="models/deepfake_detector_v3_final.keras",
    run_all_stages=True
)
```

**Expected Results (After Fine-Tuning):**
- Val Accuracy: >90%
- Val AUC: >0.92

### Step 6: Evaluation

```python
from src.models.evaluate_model import evaluate_model_v3

results = evaluate_model_v3(
    model_path="models/deepfake_detector_v3_final.keras",
    data_dir="data/processed",
    optimize_threshold=True,
    generate_plots=True
)

# View results
print(f"Test Accuracy: {results['default_threshold']['accuracy']:.4f}")
print(f"Test AUC-ROC:  {results['default_threshold']['auc_roc']:.4f}")
print(f"Test F1:       {results['default_threshold']['f1_score']:.4f}")
```

### Step 7: Save to Google Drive

```python
# Copy trained models to Drive
!cp models/*.keras /content/drive/MyDrive/deepfake_models/
!cp reports/figures/*.png /content/drive/MyDrive/deepfake_models/figures/

print("✅ Models saved to Google Drive!")
```

---

## ⚙️ Configuration Tuning

### For Lower VRAM (if OOM errors):
```python
# In train_model.py, change Config class:
BATCH_SIZE = 32  # Reduce from 64
```

### For Higher Performance (if you have more time):
```python
# In train_model.py:
EPOCHS_PHASE1 = 20    # More epochs
INITIAL_LR = 5e-4     # Slightly higher LR

# In fine_tune.py:
STAGE1_EPOCHS = 7
STAGE2_EPOCHS = 7
STAGE3_EPOCHS = 7
```

### For Quick Experimentation:
```python
from src.models.fine_tune import quick_fine_tune

model = quick_fine_tune(
    model_path="models/deepfake_detector_v3_best.keras",
    epochs=10,
    learning_rate=5e-5
)
```

---

## 📊 Training Tips for 90%+ Accuracy

1. **Data Quality**: Ensure balanced train/val/test splits
2. **Augmentation**: V3 uses careful augmentation that preserves forensic features
3. **Learning Rate**: Cosine schedule with warmup prevents early divergence
4. **Label Smoothing**: 0.1 smoothing helps calibration
5. **Progressive Unfreezing**: Don't unfreeze everything at once
6. **Early Stopping**: Monitor `val_auc`, not just `val_accuracy`
7. **Class Weights**: Automatic balancing for imbalanced data

---

## 🔧 Troubleshooting

### OOM (Out of Memory)
```python
# Reduce batch size
BATCH_SIZE = 32  # or even 16

# Clear memory between runs
import gc
import tensorflow as tf
gc.collect()
tf.keras.backend.clear_session()
```

### Training Loss Not Decreasing
```python
# Check data preprocessing
# Images should be scaled to [-1, 1] by MobileNetV2 preprocess_input()
```

### Low AUC but High Accuracy
```python
# This indicates class imbalance issues
# Ensure class_weights is being used in model.fit()
```

### Model Not Saving
```python
# Ensure models/ directory exists
import os
os.makedirs("models", exist_ok=True)
```

---

## 📁 File Structure

```
src/models/
├── srm_model.py          # SRMLayerV3, ForensicStreamV3, BayarConv2D
├── train_model.py        # build_model_v3(), train_model_v3()
├── fine_tune.py          # fine_tune_v3(), quick_fine_tune()
└── evaluate_model.py     # evaluate_model_v3()
```

---

## 🎓 Key Classes & Functions

### srm_model.py
- `SRMLayerV3`: 5-filter forensic feature extractor
- `BayarConv2D`: Learnable constrained high-pass filters
- `ForensicStreamV3`: Complete forensic stream with gating

### train_model.py
- `build_model_v3()`: Build the dual-stream architecture
- `train_model_v3()`: Phase 1 training with frozen backbone
- `Config`: Training hyperparameters
- `WarmupCosineDecay`: Learning rate schedule

### fine_tune.py
- `fine_tune_v3()`: 3-stage progressive fine-tuning
- `quick_fine_tune()`: Single-stage fast fine-tuning
- `FineTuneConfig`: Fine-tuning hyperparameters

### evaluate_model.py
- `evaluate_model_v3()`: Full evaluation with plots
- `find_optimal_threshold()`: Threshold optimization
- `compute_metrics()`: Comprehensive metrics

---

## 📈 Expected Timeline on Colab T4

| Phase | Epochs | Time | Expected Val AUC |
|-------|--------|------|------------------|
| Phase 1 (Frozen) | 15 | ~45 min | 0.88-0.92 |
| Fine-tune Stage 1 | 5 | ~20 min | 0.90-0.93 |
| Fine-tune Stage 2 | 5 | ~25 min | 0.92-0.94 |
| Fine-tune Stage 3 | 5 | ~30 min | 0.93-0.95 |
| **Total** | **30** | **~2 hours** | **>0.93** |

---

Good luck with training! 🚀
