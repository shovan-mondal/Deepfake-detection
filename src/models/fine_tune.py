"""
Deepfake Detection Model V3 - Fine-Tuning Pipeline
===================================================
Phase 2: Progressive unfreezing with discriminative learning rates

Strategy:
1. Unfreeze MobileNetV2 backbone in stages
2. Use layer-wise learning rate decay (lower layers = lower LR)
3. Continue training with careful regularization
4. Target: >90% accuracy, >0.90 AUC
"""

import sys
import os
import gc
import math
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List

import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ============================================
# 1. GPU & MEMORY CONFIGURATION
# ============================================

gc.collect()
tf.keras.backend.clear_session()

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print("✅ Mixed Precision (FP16) Enabled")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Detected: {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"⚠️ GPU Config Error: {e}")

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.srm_model import SRMLayerV3, ForensicStreamV3, BayarConv2D
from src.models.train_model import (
    Config,
    build_model_v3,
    load_dataset,
    prepare_datasets,
    get_class_weights,
    get_callbacks,
    ChannelAttention,
    RandomCutout,
    WarmupCosineDecay,
    unnormalize_image
)


# ============================================
# 2. FINE-TUNING CONFIGURATION
# ============================================

class FineTuneConfig:
    """Fine-tuning specific configuration."""
    
    # Learning rates for different stages
    STAGE1_LR = 1e-4       # Head + top layers
    STAGE2_LR = 5e-5       # More backbone unfrozen
    STAGE3_LR = 1e-5       # Full fine-tune
    
    MIN_LR = 1e-7
    
    # Epochs per stage
    STAGE1_EPOCHS = 5
    STAGE2_EPOCHS = 5
    STAGE3_EPOCHS = 5
    
    # Batch size (can be smaller for fine-tuning stability)
    BATCH_SIZE = 48
    
    # Regularization (increase during fine-tuning)
    DROPOUT_RATE = 0.4
    LABEL_SMOOTHING = 0.1
    L2_REG = 2e-4
    
    # Progressive unfreezing thresholds
    # MobileNetV2 has ~155 layers
    STAGE1_UNFREEZE = 30   # Last 30 layers
    STAGE2_UNFREEZE = 80   # Last 80 layers
    STAGE3_UNFREEZE = -1   # All layers (-1 = all)


# ============================================
# 3. LAYER-WISE LEARNING RATE
# ============================================

def get_layer_wise_lr_multipliers(model: Model, decay_factor: float = 0.9) -> Dict[str, float]:
    """
    Generate learning rate multipliers for each layer.
    
    Lower layers (closer to input) get smaller multipliers.
    This preserves pretrained features while allowing adaptation.
    
    Args:
        model: Keras model
        decay_factor: Multiplicative decay per layer group
    
    Returns:
        Dict mapping layer name to LR multiplier
    """
    
    multipliers = {}
    
    # Find MobileNetV2 backbone
    backbone = None
    for layer in model.layers:
        if 'mobilenet' in layer.name.lower():
            backbone = layer
            break
    
    if backbone is None:
        # Fallback: uniform multipliers
        for layer in model.layers:
            multipliers[layer.name] = 1.0
        return multipliers
    
    # Group backbone layers
    backbone_layers = backbone.layers
    n_layers = len(backbone_layers)
    
    # Divide into groups of ~10 layers
    group_size = 10
    n_groups = n_layers // group_size
    
    for i, layer in enumerate(backbone_layers):
        group_idx = i // group_size
        # Exponential decay from output to input
        multiplier = decay_factor ** (n_groups - group_idx)
        multipliers[f"{backbone.name}/{layer.name}"] = multiplier
    
    # Head layers get full learning rate
    for layer in model.layers:
        if layer.name not in multipliers and 'mobilenet' not in layer.name.lower():
            multipliers[layer.name] = 1.0
    
    return multipliers


class LayerWiseOptimizer(tf.keras.optimizers.Optimizer):
    """
    Wrapper optimizer that applies layer-wise learning rates.
    
    Note: This is a simplified implementation. For production,
    consider using tfa.optimizers.MultiOptimizer or similar.
    """
    
    def __init__(
        self, 
        base_optimizer: tf.keras.optimizers.Optimizer,
        lr_multipliers: Dict[str, float],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_optimizer = base_optimizer
        self.lr_multipliers = lr_multipliers
    
    def apply_gradients(self, grads_and_vars, **kwargs):
        modified_grads_and_vars = []
        
        for grad, var in grads_and_vars:
            if grad is None:
                continue
            
            # Find multiplier for this variable
            multiplier = 1.0
            for layer_name, mult in self.lr_multipliers.items():
                if layer_name in var.name:
                    multiplier = mult
                    break
            
            # Scale gradient
            scaled_grad = grad * multiplier
            modified_grads_and_vars.append((scaled_grad, var))
        
        return self.base_optimizer.apply_gradients(modified_grads_and_vars, **kwargs)
    
    def get_config(self):
        return {
            'base_optimizer': self.base_optimizer.get_config(),
            'lr_multipliers': self.lr_multipliers
        }


# ============================================
# 4. UNFREEZING FUNCTIONS
# ============================================

def unfreeze_model_layers(
    model: Model,
    num_layers_to_unfreeze: int = -1,
    verbose: bool = True
) -> Model:
    """
    Progressively unfreeze backbone layers.
    
    Args:
        model: Keras model
        num_layers_to_unfreeze: Number of layers from top to unfreeze (-1 = all)
        verbose: Print information
    
    Returns:
        Model with updated trainable status
    """
    
    # Find MobileNetV2 backbone
    backbone = None
    for layer in model.layers:
        if 'mobilenet' in layer.name.lower():
            backbone = layer
            break
    
    if backbone is None:
        print("⚠️ MobileNetV2 backbone not found")
        return model
    
    n_layers = len(backbone.layers)
    
    if num_layers_to_unfreeze == -1:
        # Unfreeze all
        backbone.trainable = True
        for layer in backbone.layers:
            layer.trainable = True
        if verbose:
            print(f"🔓 Unfroze ALL {n_layers} backbone layers")
    else:
        # Unfreeze last N layers
        backbone.trainable = True
        freeze_until = n_layers - num_layers_to_unfreeze
        
        for i, layer in enumerate(backbone.layers):
            if i < freeze_until:
                layer.trainable = False
            else:
                layer.trainable = True
        
        if verbose:
            trainable_count = sum(1 for l in backbone.layers if l.trainable)
            print(f"🔓 Unfroze {trainable_count}/{n_layers} backbone layers")
    
    return model


def count_trainable_params(model: Model) -> Tuple[int, int]:
    """Count trainable and non-trainable parameters."""
    trainable = sum(np.prod(v.shape) for v in model.trainable_weights)
    non_trainable = sum(np.prod(v.shape) for v in model.non_trainable_weights)
    return trainable, non_trainable


# ============================================
# 5. FINE-TUNING STAGES
# ============================================

def compile_for_finetuning(
    model: Model,
    learning_rate: float,
    steps_per_epoch: int,
    epochs: int,
    use_layer_wise_lr: bool = True,
    lr_decay_factor: float = 0.9
) -> Model:
    """Compile model for fine-tuning with appropriate settings."""
    
    # Learning rate schedule (cosine decay without warmup)
    total_steps = steps_per_epoch * epochs
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=total_steps,
        alpha=FineTuneConfig.MIN_LR / learning_rate
    )
    
    # Base optimizer
    base_optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=FineTuneConfig.L2_REG,
        clipnorm=1.0
    )
    
    # Layer-wise LR (optional)
    if use_layer_wise_lr:
        lr_multipliers = get_layer_wise_lr_multipliers(model, lr_decay_factor)
        # Note: For simplicity, we use base optimizer directly
        # Full layer-wise LR requires custom training loop
        optimizer = base_optimizer
    else:
        optimizer = base_optimizer
    
    # Loss with label smoothing
    loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=False,
        label_smoothing=FineTuneConfig.LABEL_SMOOTHING
    )
    
    # Metrics
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.AUC(name='auc', curve='ROC'),
        tf.keras.metrics.AUC(name='auc_pr', curve='PR'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ]
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model


def run_finetuning_stage(
    model: Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_weights: Dict[int, float],
    stage: int,
    learning_rate: float,
    epochs: int,
    unfreeze_layers: int,
    model_save_prefix: str = "models/deepfake_v3"
) -> Tuple[Model, dict]:
    """Run a single fine-tuning stage."""
    
    print(f"\n{'='*60}")
    print(f"🔧 FINE-TUNING STAGE {stage}")
    print(f"{'='*60}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Epochs: {epochs}")
    print(f"   Layers to Unfreeze: {unfreeze_layers}")
    
    # Unfreeze layers
    model = unfreeze_model_layers(model, unfreeze_layers)
    
    # Count parameters
    trainable, non_trainable = count_trainable_params(model)
    print(f"   Trainable params: {trainable:,}")
    print(f"   Non-trainable params: {non_trainable:,}")
    
    # Calculate steps
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    
    # Compile
    model = compile_for_finetuning(
        model, 
        learning_rate, 
        steps_per_epoch, 
        epochs
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            f"{model_save_prefix}_stage{stage}_best.keras",
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=FineTuneConfig.MIN_LR,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            f"{model_save_prefix}_stage{stage}_log.csv",
            append=False
        ),
    ]
    
    # Train
    print(f"\n🚀 Starting Stage {stage} Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )
    
    # Report
    best_val_auc = max(history.history.get('val_auc', [0]))
    best_val_acc = max(history.history.get('val_accuracy', [0]))
    print(f"\n📊 Stage {stage} Results:")
    print(f"   Best Val AUC: {best_val_auc:.4f}")
    print(f"   Best Val Accuracy: {best_val_acc:.4f}")
    
    return model, history.history


# ============================================
# 6. MAIN FINE-TUNING FUNCTION
# ============================================

def fine_tune_v3(
    model_path: str = "models/deepfake_detector_v3_best.keras",
    data_dir: str = "data/processed",
    output_path: str = "models/deepfake_detector_v3_finetuned.keras",
    run_all_stages: bool = True
) -> Tuple[Model, Dict]:
    """
    Progressive fine-tuning for V3 model.
    
    Stage 1: Unfreeze top 30 layers, moderate LR
    Stage 2: Unfreeze top 80 layers, lower LR
    Stage 3: Unfreeze all layers, lowest LR
    
    Args:
        model_path: Path to Phase 1 trained model
        data_dir: Data directory
        output_path: Where to save final model
        run_all_stages: Whether to run all 3 stages
    
    Returns:
        (fine-tuned model, combined history)
    """
    
    print("=" * 60)
    print("🚀 DEEPFAKE DETECTOR V3 - FINE-TUNING PIPELINE")
    print("=" * 60)
    print(f"   Base Model: {model_path}")
    print(f"   Output: {output_path}")
    print("=" * 60)
    
    # Load data
    class_weights = get_class_weights(data_dir)
    train_ds, val_ds, test_ds = load_dataset(
        data_dir, 
        batch_size=FineTuneConfig.BATCH_SIZE
    )
    train_ds, val_ds, test_ds = prepare_datasets(train_ds, val_ds, test_ds)
    
    # Load model
    print(f"\n🔄 Loading base model...")
    
    try:
        custom_objects = {
            'SRMLayerV3': SRMLayerV3,
            'ForensicStreamV3': ForensicStreamV3,
            'BayarConv2D': BayarConv2D,
            'ChannelAttention': ChannelAttention,
            'RandomCutout': RandomCutout,
            'WarmupCosineDecay': WarmupCosineDecay,
            'unnormalize_image': unnormalize_image
        }
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print("   ✅ Full model loaded")
    except Exception as e:
        print(f"   ⚠️ Full load failed: {e}")
        print("   🔄 Rebuilding and loading weights...")
        model = build_model_v3(freeze_backbone=True)
        model.load_weights(model_path)
        print("   ✅ Weights loaded")
    
    # Combined history
    all_history = {}
    
    # ========================================
    # STAGE 1: Unfreeze top 30 layers
    # ========================================
    model, history1 = run_finetuning_stage(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        class_weights=class_weights,
        stage=1,
        learning_rate=FineTuneConfig.STAGE1_LR,
        epochs=FineTuneConfig.STAGE1_EPOCHS,
        unfreeze_layers=FineTuneConfig.STAGE1_UNFREEZE,
        model_save_prefix="models/deepfake_v3"
    )
    all_history['stage1'] = history1
    
    if not run_all_stages:
        model.save(output_path)
        return model, all_history
    
    # ========================================
    # STAGE 2: Unfreeze top 80 layers
    # ========================================
    model, history2 = run_finetuning_stage(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        class_weights=class_weights,
        stage=2,
        learning_rate=FineTuneConfig.STAGE2_LR,
        epochs=FineTuneConfig.STAGE2_EPOCHS,
        unfreeze_layers=FineTuneConfig.STAGE2_UNFREEZE,
        model_save_prefix="models/deepfake_v3"
    )
    all_history['stage2'] = history2
    
    # ========================================
    # STAGE 3: Unfreeze all layers
    # ========================================
    model, history3 = run_finetuning_stage(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        class_weights=class_weights,
        stage=3,
        learning_rate=FineTuneConfig.STAGE3_LR,
        epochs=FineTuneConfig.STAGE3_EPOCHS,
        unfreeze_layers=FineTuneConfig.STAGE3_UNFREEZE,
        model_save_prefix="models/deepfake_v3"
    )
    all_history['stage3'] = history3
    
    # ========================================
    # FINAL EVALUATION
    # ========================================
    print("\n" + "=" * 60)
    print("📊 FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    
    results = model.evaluate(test_ds, verbose=1)
    print("\nTest Metrics:")
    for name, value in zip(model.metrics_names, results):
        print(f"   {name}: {value:.4f}")
    
    # Save final model
    model.save(output_path)
    print(f"\n✅ Final model saved to: {output_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📈 FINE-TUNING SUMMARY")
    print("=" * 60)
    
    for stage_name, history in all_history.items():
        best_auc = max(history.get('val_auc', [0]))
        best_acc = max(history.get('val_accuracy', [0]))
        print(f"   {stage_name}: Val AUC = {best_auc:.4f}, Val Acc = {best_acc:.4f}")
    
    print("\n✅ FINE-TUNING COMPLETE")
    print("Next step: Run evaluate_model.py for comprehensive evaluation")
    
    return model, all_history


# ============================================
# 7. QUICK FINE-TUNE (SINGLE STAGE)
# ============================================

def quick_fine_tune(
    model_path: str = "models/deepfake_detector_v3_best.keras",
    data_dir: str = "data/processed",
    epochs: int = 10,
    learning_rate: float = 5e-5
) -> Model:
    """
    Quick single-stage fine-tuning.
    
    Useful for fast iteration or when computational resources are limited.
    """
    
    print("=" * 60)
    print("🚀 QUICK FINE-TUNING (Single Stage)")
    print("=" * 60)
    
    # Load data
    class_weights = get_class_weights(data_dir)
    train_ds, val_ds, test_ds = load_dataset(data_dir, batch_size=48)
    train_ds, val_ds, test_ds = prepare_datasets(train_ds, val_ds, test_ds)
    
    # Load model
    custom_objects = {
        'SRMLayerV3': SRMLayerV3,
        'ForensicStreamV3': ForensicStreamV3,
        'BayarConv2D': BayarConv2D,
        'ChannelAttention': ChannelAttention,
        'RandomCutout': RandomCutout,
        'WarmupCosineDecay': WarmupCosineDecay,
        'unnormalize_image': unnormalize_image
    }
    
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    except:
        model = build_model_v3(freeze_backbone=True)
        model.load_weights(model_path)
    
    # Unfreeze all
    model = unfreeze_model_layers(model, num_layers_to_unfreeze=-1)
    
    # Compile
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    model = compile_for_finetuning(model, learning_rate, steps_per_epoch, epochs)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "models/deepfake_v3_quickft_best.keras",
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),
    ]
    
    # Train
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )
    
    # Evaluate
    print("\n📊 Test Set Evaluation:")
    model.evaluate(test_ds)
    
    model.save("models/deepfake_v3_quickft_final.keras")
    
    return model


# ============================================
# BACKWARD COMPATIBILITY
# ============================================

def fine_tune():
    """Backward compatible wrapper."""
    return fine_tune_v3()


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fine-tune Deepfake Detector V3')
    parser.add_argument('--quick', action='store_true', help='Run quick single-stage fine-tuning')
    parser.add_argument('--model', type=str, default='models/deepfake_detector_v3_best.keras',
                        help='Path to base model')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs for quick fine-tune')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_fine_tune(model_path=args.model, epochs=args.epochs)
    else:
        fine_tune_v3(model_path=args.model)