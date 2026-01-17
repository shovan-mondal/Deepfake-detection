"""
Deepfake Detection Model V3 - Training Pipeline
================================================
Dual-Stream Gated Architecture for GAN and Diffusion Model Detection

Optimized for Google Colab T4 GPU:
- Mixed precision training (FP16)
- Batch size 64 with gradient accumulation option
- Advanced data augmentation
- Label smoothing for better calibration
- Cosine annealing with warm restarts

Target: >90% accuracy, >0.90 AUC
"""

import sys
import os
import gc
import math
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional

import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ============================================
# 1. GPU & MEMORY CONFIGURATION
# ============================================

gc.collect()
tf.keras.backend.clear_session()

# Enable mixed precision for T4 GPU (2x speedup)
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print("✅ Mixed Precision (FP16) Enabled")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Detected: {len(gpus)} GPU(s)")
        # Print GPU info
        for gpu in gpus:
            print(f"   - {gpu.name}")
    except RuntimeError as e:
        print(f"⚠️ GPU Config Error: {e}")
else:
    print("⚠️ No GPU detected, running on CPU")

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.srm_model import SRMLayerV3, ForensicStreamV3, BayarConv2D

# ============================================
# 2. CONFIGURATION
# ============================================

class Config:
    """Training configuration for V3 model."""
    # Image settings
    IMG_SIZE = (224, 224)
    IMG_SHAPE = (224, 224, 3)
    
    # Training settings - optimized for T4 (16GB VRAM)
    BATCH_SIZE = 64          # T4 can handle 64 with mixed precision
    EPOCHS_PHASE1 = 15       # Frozen backbone
    EPOCHS_PHASE2 = 10       # Fine-tuning
    
    # Learning rate schedule
    INITIAL_LR = 1e-3        # Higher initial LR with warmup
    MIN_LR = 1e-6
    WARMUP_EPOCHS = 2
    
    # Regularization
    LABEL_SMOOTHING = 0.1    # Helps with calibration
    DROPOUT_RATE = 0.3
    L2_REG = 1e-4
    
    # Data pipeline
    SHUFFLE_BUFFER = 2048
    PREFETCH_BUFFER = tf.data.AUTOTUNE
    PARALLEL_CALLS = tf.data.AUTOTUNE
    
    # Model architecture
    MOBILENET_ALPHA = 1.0    # Full MobileNetV2
    FREEZE_UNTIL = 100       # Freeze first 100 layers initially
    
    # Paths
    MODEL_DIR = "models"
    LOG_DIR = "logs"


# ============================================
# 3. DATA AUGMENTATION PIPELINE
# ============================================

def get_augmentation_pipeline_v3(training: bool = True):
    """
    Advanced augmentation pipeline V3.
    
    Key principles:
    - Geometric transforms are safe (don't destroy forensic artifacts)
    - NO Gaussian noise (confuses SRM)
    - Careful with JPEG compression simulation
    - Color jitter helps generalization
    """
    if training:
        return tf.keras.Sequential([
            # Geometric (safe for forensics)
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15, fill_mode='reflect'),
            layers.RandomZoom((-0.15, 0.15), fill_mode='reflect'),
            layers.RandomTranslation(0.1, 0.1, fill_mode='reflect'),
            
            # Color augmentation (helps generalization)
            layers.RandomContrast(0.3),
            layers.RandomBrightness(0.2),
            
            # Cutout/Random erasing (regularization)
            RandomCutout(probability=0.3, max_holes=3, max_size=30),
            
            # Ensure valid range
            layers.Lambda(lambda x: tf.clip_by_value(x, 0, 255)),
        ], name="augmentation_v3")
    else:
        # No augmentation for validation/test
        return tf.keras.Sequential([
            layers.Lambda(lambda x: x)
        ], name="identity")


class RandomCutout(layers.Layer):
    """
    Random Cutout (Erasing) augmentation.
    Randomly erases rectangular regions to improve robustness.
    """
    def __init__(self, probability=0.5, max_holes=1, max_size=40, **kwargs):
        super().__init__(**kwargs)
        self.probability = probability
        self.max_holes = max_holes
        self.max_size = max_size
    
    def call(self, images, training=None):
        if not training:
            return images
        
        def apply_cutout(img):
            if tf.random.uniform([]) > self.probability:
                return img
            
            h = tf.shape(img)[0]
            w = tf.shape(img)[1]
            
            for _ in range(self.max_holes):
                # Random hole size
                hole_h = tf.random.uniform([], 10, self.max_size, dtype=tf.int32)
                hole_w = tf.random.uniform([], 10, self.max_size, dtype=tf.int32)
                
                # Random position
                top = tf.random.uniform([], 0, h - hole_h, dtype=tf.int32)
                left = tf.random.uniform([], 0, w - hole_w, dtype=tf.int32)
                
                # Create mask
                mask = tf.ones_like(img)
                # Set hole region to 0
                indices = tf.reshape(tf.range(top, top + hole_h), [-1, 1])
                indices = tf.tile(indices, [1, hole_w])
                col_indices = tf.reshape(tf.range(left, left + hole_w), [1, -1])
                col_indices = tf.tile(col_indices, [hole_h, 1])
                
                # Use scatter to create hole (fill with mean color)
                mean_color = tf.reduce_mean(img)
                
                # Simple approximation: zero out the region
                pad_top = top
                pad_bottom = h - top - hole_h
                pad_left = left
                pad_right = w - left - hole_w
                
                hole = tf.ones([hole_h, hole_w, 3]) * mean_color
                
                # Create padded hole mask
                hole_padded = tf.pad(hole, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
                mask_region = tf.pad(
                    tf.zeros([hole_h, hole_w, 3]), 
                    [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
                    constant_values=1.0
                )
                
                img = img * mask_region + hole_padded * (1 - mask_region)
            
            return img
        
        return tf.map_fn(apply_cutout, images)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'probability': self.probability,
            'max_holes': self.max_holes,
            'max_size': self.max_size
        })
        return config


# ============================================
# 4. DATA PIPELINE
# ============================================

def get_class_weights(data_dir: str) -> Dict[int, float]:
    """Calculate class weights for imbalanced data."""
    train_dir = os.path.join(data_dir, 'train')
    try:
        n_real = len(os.listdir(os.path.join(train_dir, 'real')))
        n_fake = len(os.listdir(os.path.join(train_dir, 'fake')))
    except FileNotFoundError:
        print("⚠️ Data directory not found, using equal weights")
        return {0: 1.0, 1: 1.0}
    
    total = n_real + n_fake
    if total == 0:
        return {0: 1.0, 1: 1.0}
    
    # Inverse frequency weighting
    weight_for_0 = (1 / n_real) * (total / 2.0)
    weight_for_1 = (1 / n_fake) * (total / 2.0)
    
    print(f"⚖️ Class Balance: {n_real:,} Real vs {n_fake:,} Fake")
    print(f"   Weights: Real={weight_for_0:.3f}, Fake={weight_for_1:.3f}")
    
    return {0: weight_for_0, 1: weight_for_1}


def load_dataset(
    data_dir: str, 
    img_size: Tuple[int, int] = Config.IMG_SIZE,
    batch_size: int = Config.BATCH_SIZE
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load train/val/test datasets."""
    
    print(f"📂 Loading dataset...")
    print(f"   Image Size: {img_size}")
    print(f"   Batch Size: {batch_size}")
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/train",
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary',
        shuffle=True,
        seed=42
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/val",
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary',
        shuffle=False
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/test",
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary',
        shuffle=False
    )
    
    print(f"   Train batches: {tf.data.experimental.cardinality(train_ds).numpy()}")
    print(f"   Val batches: {tf.data.experimental.cardinality(val_ds).numpy()}")
    print(f"   Test batches: {tf.data.experimental.cardinality(test_ds).numpy()}")
    
    return train_ds, val_ds, test_ds


def prepare_datasets(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    augment_train: bool = True
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Prepare datasets with preprocessing and augmentation.
    
    Pipeline: Augment -> Preprocess ([-1, 1]) -> Cache/Shuffle -> Prefetch
    """
    
    # Get augmentation pipelines
    train_aug = get_augmentation_pipeline_v3(training=True) if augment_train else None
    
    def preprocess_train(x, y):
        """Augment then preprocess to [-1, 1]."""
        if train_aug:
            x = train_aug(x, training=True)
        x = preprocess_input(x)  # Scale to [-1, 1]
        return x, y
    
    def preprocess_eval(x, y):
        """Only preprocess to [-1, 1]."""
        x = preprocess_input(x)
        return x, y
    
    # Apply preprocessing
    train_ds = train_ds.map(preprocess_train, num_parallel_calls=Config.PARALLEL_CALLS)
    val_ds = val_ds.map(preprocess_eval, num_parallel_calls=Config.PARALLEL_CALLS)
    test_ds = test_ds.map(preprocess_eval, num_parallel_calls=Config.PARALLEL_CALLS)
    
    # Optimize pipeline
    train_ds = train_ds.shuffle(Config.SHUFFLE_BUFFER).prefetch(Config.PREFETCH_BUFFER)
    val_ds = val_ds.cache().prefetch(Config.PREFETCH_BUFFER)
    test_ds = test_ds.cache().prefetch(Config.PREFETCH_BUFFER)
    
    return train_ds, val_ds, test_ds


# ============================================
# 5. MODEL ARCHITECTURE V3
# ============================================

def unnormalize_image(x):
    """Convert from [-1, 1] back to [0, 255] for SRM."""
    return (x + 1.0) * 127.5


class ChannelAttention(layers.Layer):
    """
    Squeeze-and-Excitation style channel attention.
    Used to help the model focus on important channels.
    """
    def __init__(self, reduction_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
    
    def build(self, input_shape):
        channels = input_shape[-1]
        self.gap = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(channels // self.reduction_ratio, activation='relu')
        self.fc2 = layers.Dense(channels, activation='sigmoid')
        super().build(input_shape)
    
    def call(self, inputs):
        x = self.gap(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        x = tf.reshape(x, [-1, 1, 1, inputs.shape[-1]])
        return inputs * x
    
    def get_config(self):
        config = super().get_config()
        config.update({'reduction_ratio': self.reduction_ratio})
        return config


def build_model_v3(
    input_shape: Tuple[int, int, int] = Config.IMG_SHAPE,
    freeze_backbone: bool = True,
    freeze_until: int = Config.FREEZE_UNTIL,
    dropout_rate: float = Config.DROPOUT_RATE
) -> Model:
    """
    Build Dual-Stream Gated Architecture V3.
    
    Architecture:
    ┌─────────────────────────────────────────────────┐
    │                    Input (224x224x3)            │
    └─────────────────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
    ┌─────────────────────┐     ┌─────────────────────┐
    │   Visual Stream     │     │   Forensic Stream   │
    │   (MobileNetV2)     │     │   (SRM + Bayar)     │
    │   1280-D features   │     │   Gate Vector       │
    └─────────────────────┘     └─────────────────────┘
              │                           │
              │         ┌─────────────────┘
              │         │ (Gating)
              ▼         ▼
    ┌─────────────────────────────────────────────────┐
    │              Gated Fusion                        │
    │         visual_feat × gate_vector               │
    └─────────────────────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────┐
    │              Classification Head                 │
    │         Dense(512) -> Dense(256) -> Dense(1)    │
    └─────────────────────────────────────────────────┘
    """
    
    print("🏗️ Building Dual-Stream Gated Architecture V3...")
    
    # Input
    inputs = layers.Input(shape=input_shape, name="image_input")
    
    # ========================================
    # STREAM A: Visual Branch (MobileNetV2)
    # ========================================
    
    # Load pretrained MobileNetV2
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        alpha=Config.MOBILENET_ALPHA
    )
    base_model._name = "mobilenetv2_backbone"
    
    # Freeze strategy
    if freeze_backbone:
        base_model.trainable = True
        for i, layer in enumerate(base_model.layers):
            if i < freeze_until:
                layer.trainable = False
        
        trainable_count = sum(1 for l in base_model.layers if l.trainable)
        print(f"   Visual Stream: {trainable_count}/{len(base_model.layers)} layers trainable")
    else:
        base_model.trainable = True
        print(f"   Visual Stream: All {len(base_model.layers)} layers trainable")
    
    # Extract visual features
    visual_map = base_model(inputs, training=True)  # (batch, 7, 7, 1280)
    
    # Add channel attention
    visual_attended = ChannelAttention(reduction_ratio=16, name="visual_attention")(visual_map)
    
    # Global pooling
    visual_vec = layers.GlobalAveragePooling2D(name="visual_gap")(visual_attended)  # (batch, 1280)
    
    # ========================================
    # STREAM B: Forensic Branch (SRM + Bayar)
    # ========================================
    
    # Convert back to [0, 255] for SRM filters
    x_raw = layers.Lambda(unnormalize_image, name="unnormalize_for_srm")(inputs)
    
    # Forensic stream generates gate vector
    forensic_stream = ForensicStreamV3(gate_dim=1280, name="forensic_stream")
    gate_vector = forensic_stream(x_raw, training=True)  # (batch, 1280) with values in [0, 1]
    
    print(f"   Forensic Stream: SRM + Bayar with gating")
    
    # ========================================
    # GATED FUSION
    # ========================================
    
    # Apply gating: element-wise multiplication
    gated_visual = layers.Multiply(name="gated_fusion")([visual_vec, gate_vector])
    
    # Skip connection: add ungated features for robustness
    fused = layers.Concatenate(name="feature_concat")([gated_visual, visual_vec])
    
    # ========================================
    # CLASSIFICATION HEAD
    # ========================================
    
    # Dense layers with regularization
    x = layers.Dense(
        512, 
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REG),
        name="head_dense1"
    )(fused)
    x = layers.BatchNormalization(name="head_bn1")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout1")(x)
    
    x = layers.Dense(
        256,
        activation='relu', 
        kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REG),
        name="head_dense2"
    )(x)
    x = layers.BatchNormalization(name="head_bn2")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout2")(x)
    
    # Output layer (float32 for numerical stability with mixed precision)
    x = layers.Dense(1, name="logits")(x)
    outputs = layers.Activation('sigmoid', dtype='float32', name="prediction")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="deepfake_detector_v3")
    
    print(f"✅ Model built: {model.count_params():,} total parameters")
    
    return model


# ============================================
# 6. LEARNING RATE SCHEDULES
# ============================================

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Cosine decay with linear warmup.
    
    - Warmup: Linear increase from 0 to initial_lr
    - Decay: Cosine decay from initial_lr to min_lr
    """
    
    def __init__(
        self, 
        initial_lr: float,
        decay_steps: int,
        warmup_steps: int,
        min_lr: float = 1e-7
    ):
        super().__init__()
        self.initial_lr = initial_lr
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        
        # Warmup phase
        warmup_lr = self.initial_lr * (step / self.warmup_steps)
        
        # Cosine decay phase
        decay_step = step - self.warmup_steps
        decay_steps = self.decay_steps - self.warmup_steps
        cosine_decay = 0.5 * (1 + tf.cos(math.pi * decay_step / decay_steps))
        decayed_lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
        
        # Select based on step
        return tf.where(step < self.warmup_steps, warmup_lr, decayed_lr)
    
    def get_config(self):
        return {
            'initial_lr': self.initial_lr,
            'decay_steps': self.decay_steps,
            'warmup_steps': self.warmup_steps,
            'min_lr': self.min_lr
        }


# ============================================
# 7. LOSS FUNCTIONS
# ============================================

def get_loss_function(label_smoothing: float = Config.LABEL_SMOOTHING):
    """
    Binary crossentropy with label smoothing.
    
    Label smoothing helps:
    - Better calibrated probabilities
    - Reduced overconfidence
    - Improved generalization
    """
    return tf.keras.losses.BinaryCrossentropy(
        from_logits=False,
        label_smoothing=label_smoothing
    )


# ============================================
# 8. CALLBACKS
# ============================================

def get_callbacks(
    model_path: str = "models/best_model_v3.keras",
    log_path: str = "models/training_log_v3.csv",
    patience_early_stop: int = 7,
    patience_reduce_lr: int = 3
) -> list:
    """Get training callbacks."""
    
    callbacks = [
        # Save best model
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=patience_early_stop,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce LR on plateau (backup to cosine schedule)
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience_reduce_lr,
            min_lr=Config.MIN_LR,
            verbose=1
        ),
        
        # CSV logging
        tf.keras.callbacks.CSVLogger(log_path, append=False),
        
        # TensorBoard (optional)
        tf.keras.callbacks.TensorBoard(
            log_dir=Config.LOG_DIR,
            histogram_freq=0,
            update_freq='epoch'
        ),
    ]
    
    return callbacks


# ============================================
# 9. COMPILE MODEL
# ============================================

def compile_model(
    model: Model,
    steps_per_epoch: int,
    epochs: int = Config.EPOCHS_PHASE1,
    initial_lr: float = Config.INITIAL_LR,
    warmup_epochs: int = Config.WARMUP_EPOCHS,
    label_smoothing: float = Config.LABEL_SMOOTHING
) -> Model:
    """Compile model with optimizer, loss, and metrics."""
    
    # Calculate schedule parameters
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    
    # Learning rate schedule
    lr_schedule = WarmupCosineDecay(
        initial_lr=initial_lr,
        decay_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr=Config.MIN_LR
    )
    
    # Optimizer with gradient clipping
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=Config.L2_REG,
        clipnorm=1.0  # Gradient clipping
    )
    
    # Loss
    loss = get_loss_function(label_smoothing)
    
    # Metrics
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.AUC(name='auc', curve='ROC'),
        tf.keras.metrics.AUC(name='auc_pr', curve='PR'),  # Precision-Recall AUC
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ]
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    print(f"✅ Model compiled:")
    print(f"   Optimizer: AdamW with cosine LR decay")
    print(f"   Initial LR: {initial_lr}, Warmup: {warmup_epochs} epochs")
    print(f"   Label Smoothing: {label_smoothing}")
    
    return model


# ============================================
# 10. TRAINING FUNCTION
# ============================================

def train_model_v3(
    data_dir: str = "data/processed",
    batch_size: int = Config.BATCH_SIZE,
    epochs: int = Config.EPOCHS_PHASE1,
    model_save_path: str = "models/deepfake_detector_v3.keras"
) -> Tuple[Model, dict]:
    """
    Main training function for V3 model.
    
    Phase 1: Train with frozen backbone
    """
    
    print("=" * 60)
    print("🚀 DEEPFAKE DETECTOR V3 - TRAINING")
    print("=" * 60)
    print(f"   Batch Size: {batch_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Mixed Precision: FP16")
    print("=" * 60)
    
    # Load data
    class_weights = get_class_weights(data_dir)
    train_ds, val_ds, test_ds = load_dataset(data_dir, batch_size=batch_size)
    train_ds, val_ds, test_ds = prepare_datasets(train_ds, val_ds, test_ds)
    
    # Calculate steps
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    
    # Build model
    model = build_model_v3(freeze_backbone=True)
    
    # Compile
    model = compile_model(model, steps_per_epoch, epochs)
    
    # Print summary
    model.summary()
    
    # Callbacks
    callbacks = get_callbacks(
        model_path=model_save_path.replace('.keras', '_best.keras'),
        log_path=model_save_path.replace('.keras', '_log.csv')
    )
    
    # Train
    print("\n🚀 Starting Phase 1 Training (Frozen Backbone)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )
    
    # Save final model
    model.save(model_save_path)
    print(f"\n✅ Model saved to {model_save_path}")
    
    # Quick evaluation
    print("\n📊 Quick Evaluation on Test Set:")
    results = model.evaluate(test_ds, verbose=1)
    for name, value in zip(model.metrics_names, results):
        print(f"   {name}: {value:.4f}")
    
    return model, history.history


# ============================================
# 11. MAIN ENTRY POINT
# ============================================

def main():
    """Main entry point for training."""
    
    # Create directories
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    # Train
    model, history = train_model_v3(
        data_dir="data/processed",
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS_PHASE1,
        model_save_path="models/deepfake_detector_v3.keras"
    )
    
    print("\n" + "=" * 60)
    print("✅ PHASE 1 TRAINING COMPLETE")
    print("=" * 60)
    print("Next step: Run fine_tune.py for Phase 2 (unfreeze backbone)")


if __name__ == "__main__":
    main()


# ============================================
# EXPORTS (for imports from other modules)
# ============================================

# Re-export for backward compatibility
def build_model(*args, **kwargs):
    """Backward compatible wrapper."""
    return build_model_v3(*args, **kwargs)