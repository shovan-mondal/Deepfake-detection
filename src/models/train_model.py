import sys
import os
import gc
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- 1. MEMORY SAFETY FIRST ---
gc.collect()
tf.keras.backend.clear_session()

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Detected: {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(e)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import get_preprocessing_pipeline
from src.models.srm_model import SRMLayer 

# --- 2. DATA PIPELINE ---

def get_class_weights(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    try:
        n_real = len(os.listdir(os.path.join(train_dir, 'real')))
        n_fake = len(os.listdir(os.path.join(train_dir, 'fake')))
    except FileNotFoundError:
        print("⚠️ Data directory not found.")
        return {0: 1.0, 1: 1.0}
    
    total = n_real + n_fake
    if total == 0: return {0: 1.0, 1: 1.0}

    weight_for_0 = (1 / n_real) * (total / 2.0)
    weight_for_1 = (1 / n_fake) * (total / 2.0)
    
    print(f"⚖️  Class Balance: {n_real} Real vs {n_fake} Fake")
    return {0: weight_for_0, 1: weight_for_1}

def load_dataset(data_dir, img_size=(224, 224), batch_size=16):
    print(f"📂 Loading dataset with Batch Size: {batch_size}...")
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
        shuffle=False
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
    PARALLEL = 4
    def preprocess(x, y):
        return preprocess_input(x), y

    # 1. Map
    train_ds = train_ds.map(preprocess, num_parallel_calls=PARALLEL)
    val_ds = val_ds.map(preprocess, num_parallel_calls=PARALLEL)
    test_ds = test_ds.map(preprocess, num_parallel_calls=PARALLEL)

    # 2. NO CACHING
    # val_ds = val_ds.cache() # Optional: Uncomment if you have RAM to spare
    
    # 3. Buffer
    train_ds = train_ds.shuffle(20).prefetch(buffer_size=1) 
    val_ds = val_ds.prefetch(buffer_size=1)
    test_ds = test_ds.prefetch(buffer_size=1)

    return train_ds, val_ds, test_ds

# --- 3. MODEL ARCHITECTURE ---

# FIX: Define this OUTSIDE build_model so it can be saved/pickled
def unnormalize_image(t):
    # Converts [-1, 1] back to [0, 255]
    return (t + 1.0) * 127.5

def build_model(input_shape=(224, 224, 3)):
    inputs = layers.Input(shape=input_shape, name="image")

    # 1. Augmentation
    aug_pipeline = get_preprocessing_pipeline()
    x = aug_pipeline(inputs)

    # 2. Visual Branch
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False
        
    visual_map = base_model(x, training=False)
    visual_vec = layers.GlobalAveragePooling2D(name="visual_gap")(visual_map)

    # 3. SRM Branch
    # FIX: Use the named function instead of lambda
    x_srm_raw = layers.Lambda(unnormalize_image, name="unnormalize_for_srm")(x)
    
    srm_maps = SRMLayer(name="srm_fixed")(x_srm_raw) 
    srm_norm = layers.BatchNormalization(name="srm_bn")(srm_maps)
    srm_feat = layers.Conv2D(16, 3, padding="same", activation="relu", name="srm_learnable_conv")(srm_norm)
    srm_vec = layers.GlobalAveragePooling2D(name="srm_gap")(srm_feat)

    # 4. Gating
    gate = layers.Dense(1280, activation="sigmoid", name="forensic_gate")(srm_vec)
    gated_visual = layers.Multiply(name="gated_fusion")([visual_vec, gate])

    head = layers.Dense(256, activation="relu", name="head_dense")(gated_visual)
    head = layers.Dropout(0.4)(head)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(head)

    model = models.Model(inputs=inputs, outputs=outputs, name="mobile_srm_v2")
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
                  loss="binary_crossentropy", 
                  metrics=[
                      'accuracy', 
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall'),
                      tf.keras.metrics.AUC(name='auc')])
    return model

# --- 4. MAIN ---

def main():
    data_dir = "data/processed"
    
    class_weights = get_class_weights(data_dir)
    train_ds, val_ds, test_ds = load_dataset(data_dir, batch_size=16)
    train_ds, val_ds, test_ds = prepare_datasets(train_ds, val_ds, test_ds)

    model = build_model()
    model.summary()
    
    callbacks = [
        # FIX: Changed .h5 -> .keras (Native Format prevents Pickle errors)
        tf.keras.callbacks.ModelCheckpoint(
            "models/best_srm_model_v2.keras", 
            save_best_only=True,
            monitor='val_auc', 
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, verbose=1),
        tf.keras.callbacks.CSVLogger("models/training_log_v2.csv")
    ]

    print("🚀 Starting V2 Training (Stable .keras mode)...")
    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=15, 
        class_weight=class_weights, 
        callbacks=callbacks
    )

    # FIX: Changed .h5 -> .keras
    model.save("models/deepfake_detector_v2_final.keras")
    print("✅ Training Complete.")

if __name__ == "__main__":
    main()