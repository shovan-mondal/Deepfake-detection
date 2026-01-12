import sys
import os
import tensorflow as tf
from pathlib import Path

# --- SETUP ---
# Memory Safety
import gc
gc.collect()
tf.keras.backend.clear_session()

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import necessary components
from src.models.train_model import build_model, load_dataset, prepare_datasets, get_class_weights, unnormalize_image
from src.models.srm_model import SRMLayer

def fine_tune():
    # 1. Setup Data
    print("🔄 Loading Data for Fine-Tuning...")
    data_dir = "data/processed"
    class_weights = get_class_weights(data_dir)
    
    # We use the SAME batch size (16) to fit in memory
    train_ds, val_ds, test_ds = load_dataset(data_dir, batch_size=16)
    train_ds, val_ds, test_ds = prepare_datasets(train_ds, val_ds, test_ds)

    # 2. Rebuild & Load Weights
    print("🏗️  Rebuilding V2 Model...")
    model = build_model()
    
    # Load the best weights you just trained
    weight_path = "models/best_srm_model_v2.keras"
    print(f"📥 Loading weights from {weight_path}...")
    model.load_weights(weight_path)

    # 3. THE SURGERY: Unfreeze MobileNet
    print("🔓 Unfreezing MobileNet layers for Deep Learning...")
    
    # Find the MobileNet layer (it's usually a Functional layer inside our model)
    # We iterate to find the backbone
    mobilenet_layer = None
    for layer in model.layers:
        if "mobilenet" in layer.name or "model" in layer.name:
            mobilenet_layer = layer
            break
    
    if mobilenet_layer:
        mobilenet_layer.trainable = True
        print(f"   ✅ Unfroze layer: {mobilenet_layer.name}")
    else:
        # Fallback if MobileNet isn't wrapped (depends on Keras version)
        # We unfreeze everything
        print("   ⚠️ specific layer not found, unfreezing WHOLE model.")
        model.trainable = True

    # 4. Compile with TINY Learning Rate
    # CRITICAL: Use 1e-5 (10x smaller) to avoid destroying existing knowledge
    FINE_TUNE_LR = 1e-5
    
    print(f"⚙️ Re-compiling with Low LR: {FINE_TUNE_LR}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss="binary_crossentropy",
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    model.summary()

    # 5. Fine-Tune Training
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "models/best_srm_model_v2_finetuned.keras",  # New filename
            save_best_only=True,
            monitor='val_auc', 
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor='val_auc'),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=2, verbose=1, min_lr=1e-7),
        tf.keras.callbacks.CSVLogger("models/training_log_finetune.csv")
    ]

    print("🚀 Starting Phase 3: Fine-Tuning (Target > 88%)...")
    
    # Train for 10 more epochs (Total effectively 25)
    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=10, 
        class_weight=class_weights, 
        callbacks=callbacks
    )

    print("✅ Fine-Tuning Complete.")
    model.save("models/deepfake_detector_v2_FINAL_SOTA.keras")

if __name__ == "__main__":
    fine_tune()