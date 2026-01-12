import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # <--- IMPORT THIS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import build_model to rebuild architecture
from src.models.train_model import build_model

def load_test_data(data_dir):
    print("📂 Loading Test Data...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/test",
        image_size=(224, 224),
        batch_size=32,
        label_mode='binary',
        shuffle=False # CRITICAL: Must be False for evaluation
    )
    
    # --- CRITICAL FIX: APPLY PREPROCESSING ---
    # We must scale data to [-1, 1] or the model sees garbage.
    print("⚙️ Applying MobileNetV2 Preprocessing...")
    test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))
    
    return test_ds

def evaluate():
    # model_path = "models/best_srm_model_v2.keras"
    model_path = "models/best_srm_model_v2_finetuned.keras"
    print(f"🔄 Loading model architecture and weights...")
    
    try:
        # Rebuild Architecture + Load Weights
        model = build_model()
        model.load_weights(model_path)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    data_dir = "data/processed"
    test_ds = load_test_data(data_dir)
    
    print("🔮 Running Predictions on Test Set... (This takes a moment)")
    
    # Get True Labels
    # Note: We iterate over the UNBATCHED dataset to ensure alignment
    y_true = []
    for batch_x, batch_y in test_ds:
        y_true.extend(batch_y.numpy())
    y_true = np.array(y_true)
    
    # Get Predictions
    predictions = model.predict(test_ds)
    y_pred = (predictions > 0.5).astype("int32")

    print("\n" + "="*40)
    print("📊 FINAL RESEARCH METRICS REPORT")
    print("="*40)
    
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))
    
    auc = roc_auc_score(y_true, predictions)
    f1 = f1_score(y_true, y_pred)
    
    print(f"🏆 Final AUC Score: {auc:.4f}")
    print(f"🏆 Final F1 Score:  {f1:.4f}")
    print("="*40)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], 
                yticklabels=['Real', 'Fake'])
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix\nAUC: {auc:.3f} | F1: {f1:.3f}')
    
    os.makedirs("reports", exist_ok=True)
    save_path = "reports/confusion_matrix_v2.png"
    plt.savefig(save_path)
    print(f"✅ Confusion Matrix saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    evaluate()