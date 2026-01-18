import sys
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- SETUP PATHS ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 1. Import Model Builder
from src.models.train_model import build_model

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

def plot_training_results(log_path="models/training_log_v2.csv", output_dir="reports"):
    """ Reads the CSV log from training and plots Accuracy, Loss, and AUC curves. """
    if not os.path.exists(log_path):
        print(f"⚠️ Log file not found at {log_path}. Did training finish?")
        return

    df = pd.read_csv(log_path)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Accuracy & AUC Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(df['accuracy'], label='Train Accuracy', linewidth=2)
    plt.plot(df['val_accuracy'], label='Val Accuracy', linewidth=2, linestyle='--')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    if 'auc' in df.columns:
        plt.plot(df['auc'], label='Train AUC', color='purple', linewidth=2)
        plt.plot(df['val_auc'], label='Val AUC', color='orange', linewidth=2, linestyle='--')
        plt.title('Area Under Curve (AUC)')
        plt.xlabel('Epochs')
        plt.ylabel('AUC Score')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_performance.png", dpi=300)
    plt.close()

    # 2. Loss Plot
    plt.figure(figsize=(6, 5))
    plt.plot(df['loss'], label='Train Loss', color='red', linewidth=2)
    plt.plot(df['val_loss'], label='Val Loss', color='blue', linewidth=2, linestyle='--')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{output_dir}/training_loss.png", dpi=300)
    print(f"✅ Training Plots saved to {output_dir}")

def visualize_srm_layer(model_path, image_path, output_dir="reports"):
    print(f"🔍 Extracting SRM features from {image_path}...")
    try:
        full_model = build_model()
        full_model.load_weights(model_path)
        
        srm_model = tf.keras.Model(inputs=full_model.input, 
                                   outputs=full_model.get_layer("srm_fixed").output)
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Load Image
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    
    # Preprocess for Model (Scale to -1, 1)
    img_preprocessed = preprocess_input(img_array.copy())
    img_batch = np.expand_dims(img_preprocessed, axis=0)

    # Get SRM Output
    noise_map = srm_model.predict(img_batch) 
    
    # Visualization Logic
    srm_viz = noise_map[0]
    srm_viz = np.mean(srm_viz, axis=-1)
    srm_viz = (srm_viz - srm_viz.min()) / (srm_viz.max() - srm_viz.min()) 
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image (RGB)")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(srm_viz, cmap='inferno')
    plt.title("SRM Noise Residuals (What Model Sees)")
    plt.axis('off')
    
    os.makedirs(output_dir, exist_ok=True)
    save_name = Path(image_path).stem
    plt.savefig(f"{output_dir}/srm_viz_{save_name}.png", dpi=300, bbox_inches='tight')
    print(f"✅ SRM Visualization saved.")
    plt.close()

def visualize_prediction_grid(model_path, data_dir, output_dir="reports"):
    print("🔮 Generating Prediction Grid...")
    
    full_model = build_model()
    full_model.load_weights(model_path)
    
    # Load raw images [0, 255]
    test_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/test",
        image_size=(224, 224),
        batch_size=9,
        shuffle=True
    ).take(1)
    
    plt.figure(figsize=(10, 10))
    
    for images, labels in test_ds:
        # 1. Preprocess for Prediction ONLY [-1, 1]
        images_processed = preprocess_input(images.numpy().copy())
        preds = full_model.predict(images_processed)
        
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            
            # 2. Use Raw Image for Display [0, 255]
            # Convert float32 0-255 to uint8 0-255
            img_disp = images[i].numpy().astype("uint8")
            plt.imshow(img_disp)
            
            confidence = preds[i][0]
            true_label = int(labels[i])
            pred_label = 1 if confidence > 0.5 else 0
            label_map = {0: "Real", 1: "Fake"}
            color = 'green' if true_label == pred_label else 'red'
            
            plt.title(f"True: {label_map[true_label]}\nPred: {label_map[pred_label]} ({confidence:.2f})", 
                      color=color, fontsize=10)
            plt.axis("off")
            
    plt.savefig(f"{output_dir}/prediction_grid.png")
    print(f"✅ Prediction Grid saved to {output_dir}/prediction_grid.png")
    plt.close()

if __name__ == "__main__":
    plot_training_results()
    
    sample_fake = "data/processed/test/fake"
    if os.path.exists(sample_fake) and len(os.listdir(sample_fake)) > 0:
        first_img = os.listdir(sample_fake)[0]
        full_path = os.path.join(sample_fake, first_img)
        visualize_srm_layer("models/best_v3_model_best.keras", full_path)
    
    if os.path.exists("models/best_v3_model_best.keras"):
        visualize_prediction_grid("models/best_v3_model_best.keras", "data/processed")