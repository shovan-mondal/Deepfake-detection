# src/models/train_model.py
import sys
import os
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Allow running this file directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import get_preprocessing_pipeline
# Ensure this matches the filename you created (srm_layers.py)
from srm_model import SRMLayer 

def get_class_weights(data_dir):
    """Calculates class weights to handle imbalanced datasets (Merged FF++/Celeb/CiFAKE)."""
    # Count files in training set
    train_dir = os.path.join(data_dir, 'train')
    n_real = len(os.listdir(os.path.join(train_dir, 'real')))
    n_fake = len(os.listdir(os.path.join(train_dir, 'fake')))
    
    total = n_real + n_fake
    # Formula: Total / (2 * Count)
    weight_for_0 = (1 / n_real) * (total / 2.0) # Real
    weight_for_1 = (1 / n_fake) * (total / 2.0) # Fake
    
    print(f"⚖️  Class Balance: {n_real} Real vs {n_fake} Fake")
    print(f"⚖️  Calculated Weights: Real={weight_for_0:.2f}, Fake={weight_for_1:.2f}")
    
    return {0: weight_for_0, 1: weight_for_1}

def load_dataset(data_dir, img_size=(224, 224), batch_size=32):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/train",
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary',
        shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/val",
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary',
        shuffle=False # Don't shuffle val for consistent metrics
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/test",
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary',
        shuffle=False
    )

    return train_ds, val_ds, test_ds

def prepare_datasets(train_ds, val_ds, test_ds):
    aug = get_preprocessing_pipeline()
    AUTOTUNE = tf.data.AUTOTUNE

    # CRITICAL FIX: Use MobileNetV2 specific preprocessing ([-1, 1] range)
    # Note: 'aug' typically outputs [0, 255] or [0, 1]. Ensure preprocess_input receives what it expects.
    # Assuming standard image load is [0, 255].
    
    train_ds = train_ds.map(lambda x, y: (preprocess_input(aug(x)), y), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)

    return (
        train_ds.prefetch(AUTOTUNE),
        val_ds.prefetch(AUTOTUNE),
        test_ds.prefetch(AUTOTUNE),
    )

def build_model(input_shape=(224, 224, 3)):
    inputs = layers.Input(shape=input_shape, name="image")

    # --- Stream 1: Visual Branch ---
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    
    # CHANGE: Unfreeze the last block so it can learn "Deepfake" textures
    base_model.trainable = True
    # Freeze all layers except the last 30 (fine-tuning)
    for layer in base_model.layers[:-30]:
        layer.trainable = False
        
    visual_map = base_model(inputs, training=False) # Keep BatchNorm in inference mode
    visual_vec = layers.GlobalAveragePooling2D(name="visual_gap")(visual_map)

    # --- Stream 2: Forensic SRM Branch ---
    # 1. Fixed Forensic Filters (Extract Noise)
    srm_maps = SRMLayer(name="srm_fixed")(inputs) 
    # 2. Normalize (Important because SRM output is small float, RGB is large)
    srm_norm = layers.BatchNormalization(name="srm_bn")(srm_maps)
    # 3. Learnable Processing (Combine 9 channels -> 16 features)
    srm_feat = layers.Conv2D(16, 3, padding="same", activation="relu", name="srm_learnable_conv")(srm_norm)
    # 4. Squeeze to vector (Batch, 16)
    srm_vec = layers.GlobalAveragePooling2D(name="srm_gap")(srm_feat)

    # --- Gating Mechanism ---
    # Project SRM vector to match MobileNet dimensions (16 -> 1280)
    gate = layers.Dense(1280, activation="sigmoid", name="forensic_gate")(srm_vec)
    
    # Apply Gate: Visual Features * Forensic Confidence
    gated_visual = layers.Multiply(name="gated_fusion")([visual_vec, gate])

    # --- Classifier Head ---
    x = layers.Dense(256, activation="relu", name="head_dense")(gated_visual)
    x = layers.Dropout(0.4, name="head_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="mobile_srm_detector")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), # Lower LR for fine-tuning
                  loss="binary_crossentropy", 
                  metrics=["accuracy"])
    return model

def main():
    data_dir = "data/processed"
    
    # 1. Calculate Class Weights (Fix for Imbalance)
    class_weights = get_class_weights(data_dir)

    # 2. Load & Prepare
    train_ds, val_ds, test_ds = load_dataset(data_dir)
    train_ds, val_ds, test_ds = prepare_datasets(train_ds, val_ds, test_ds)

    # 3. Build & Train
    model = build_model()
    print(model.summary())
    
    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=15, # Increased slightly to let the Gate converge
        class_weight=class_weights, # CRITICAL
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint("models/best_srm_model.h5", save_best_only=True)
        ]
    )

    model.save("models/deepfake_detector_final.h5")
    print("🚀 Model training complete and saved!")

if __name__ == "__main__":
    main()