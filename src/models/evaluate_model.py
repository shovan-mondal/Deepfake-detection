"""
Deepfake Detection Model V3 - Evaluation Pipeline
==================================================
Comprehensive evaluation with:
- Threshold optimization
- Per-class and per-dataset metrics
- Calibration analysis
- Visualization suite
"""

import sys
import os
import gc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, Optional
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    f1_score, 
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from sklearn.calibration import calibration_curve

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Memory safety
gc.collect()
tf.keras.backend.clear_session()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import custom layers for model loading
from src.models.srm_model import SRMLayerV3, ForensicStreamV3, BayarConv2D
from src.models.train_model import (
    build_model_v3, 
    ChannelAttention, 
    RandomCutout,
    WarmupCosineDecay,
    unnormalize_image
)


# ============================================
# CONFIGURATION
# ============================================

class EvalConfig:
    """Evaluation configuration."""
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 64
    REPORTS_DIR = "reports"
    FIGURES_DIR = "reports/figures"


# ============================================
# DATA LOADING
# ============================================

def load_test_data(data_dir: str, batch_size: int = EvalConfig.BATCH_SIZE) -> tf.data.Dataset:
    """Load and preprocess test dataset."""
    
    print("📂 Loading Test Data...")
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/test",
        image_size=EvalConfig.IMG_SIZE,
        batch_size=batch_size,
        label_mode='binary',
        shuffle=False  # CRITICAL: Must be False for evaluation
    )
    
    # Apply preprocessing
    def preprocess(x, y):
        return preprocess_input(x), y
    
    test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    print(f"   Batches: {tf.data.experimental.cardinality(test_ds).numpy()}")
    
    return test_ds


def get_predictions_and_labels(
    model: tf.keras.Model, 
    dataset: tf.data.Dataset
) -> Tuple[np.ndarray, np.ndarray]:
    """Get predictions and true labels from dataset."""
    
    print("🔮 Running Predictions...")
    
    # Collect true labels
    y_true = []
    for _, batch_y in dataset:
        y_true.extend(batch_y.numpy().flatten())
    y_true = np.array(y_true)
    
    # Get predictions
    y_pred_proba = model.predict(dataset, verbose=1)
    y_pred_proba = y_pred_proba.flatten()
    
    return y_pred_proba, y_true


# ============================================
# THRESHOLD OPTIMIZATION
# ============================================

def find_optimal_threshold(
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find optimal classification threshold.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        metric: 'f1', 'youden', or 'balanced'
    
    Returns:
        (optimal_threshold, best_score)
    """
    
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_score = 0
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'youden':
            # Youden's J statistic: sensitivity + specificity - 1
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sensitivity + specificity - 1
        elif metric == 'balanced':
            # Balanced accuracy
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = (sensitivity + specificity) / 2
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold, best_score


# ============================================
# METRICS COMPUTATION
# ============================================

def compute_metrics(
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict:
    """Compute comprehensive metrics."""
    
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Core metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # AUC scores
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    auc_pr = average_precision_score(y_true, y_pred_proba)
    
    # Balanced accuracy
    balanced_acc = (recall + specificity) / 2
    
    metrics = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'threshold': threshold,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn
    }
    
    return metrics


# ============================================
# VISUALIZATION
# ============================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict,
    save_path: str
):
    """Plot and save confusion matrix."""
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    
    # Percentage confusion matrix
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Custom annotation with both count and percentage
    annot = np.array([[f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)' 
                       for j in range(2)] for i in range(2)])
    
    sns.heatmap(
        cm, 
        annot=annot, 
        fmt='', 
        cmap='Blues',
        xticklabels=['Real', 'Fake'],
        yticklabels=['Real', 'Fake'],
        annot_kws={'size': 14}
    )
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(
        f'Confusion Matrix\n'
        f'Accuracy: {metrics["accuracy"]:.3f} | AUC: {metrics["auc_roc"]:.3f} | F1: {metrics["f1_score"]:.3f}',
        fontsize=14
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Confusion matrix saved: {save_path}")
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: str
):
    """Plot and save ROC curve."""
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(10, 8))
    
    # ROC curve
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'Model (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Chance')
    
    # Find optimal point (Youden's J)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    
    plt.scatter(fpr[best_idx], tpr[best_idx], c='green', s=100, zorder=5,
                label=f'Optimal (thresh={best_threshold:.2f})')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Deepfake Detection', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ ROC curve saved: {save_path}")
    plt.close()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: str
):
    """Plot and save Precision-Recall curve."""
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    ap = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(10, 8))
    
    plt.plot(recall, precision, 'b-', linewidth=2, label=f'Model (AP = {ap:.4f})')
    
    # Baseline (proportion of positive class)
    baseline = y_true.sum() / len(y_true)
    plt.axhline(y=baseline, color='r', linestyle='--', linewidth=1, 
                label=f'Baseline ({baseline:.2f})')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve - Deepfake Detection', fontsize=14)
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ PR curve saved: {save_path}")
    plt.close()


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: str,
    n_bins: int = 10
):
    """Plot and save calibration curve."""
    
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
    
    plt.figure(figsize=(10, 8))
    
    # Calibration curve
    plt.plot(prob_pred, prob_true, 'b-o', linewidth=2, markersize=8, label='Model')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Perfectly Calibrated')
    
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title('Calibration Curve - Deepfake Detection', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Calibration curve saved: {save_path}")
    plt.close()


def plot_prediction_distribution(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float,
    save_path: str
):
    """Plot prediction probability distribution by class."""
    
    plt.figure(figsize=(12, 6))
    
    # Separate predictions by true class
    real_probs = y_pred_proba[y_true == 0]
    fake_probs = y_pred_proba[y_true == 1]
    
    # Histograms
    plt.hist(real_probs, bins=50, alpha=0.6, label=f'Real (n={len(real_probs)})', 
             color='green', density=True)
    plt.hist(fake_probs, bins=50, alpha=0.6, label=f'Fake (n={len(fake_probs)})', 
             color='red', density=True)
    
    # Threshold line
    plt.axvline(x=threshold, color='black', linestyle='--', linewidth=2,
                label=f'Threshold ({threshold:.2f})')
    
    plt.xlabel('Predicted Probability (Fake)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Prediction Distribution by True Class', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Distribution plot saved: {save_path}")
    plt.close()


# ============================================
# MAIN EVALUATION FUNCTION
# ============================================

def evaluate_model_v3(
    model_path: str = "models/deepfake_detector_v3_best.keras",
    data_dir: str = "data/processed",
    optimize_threshold: bool = True,
    generate_plots: bool = True
) -> Dict:
    """
    Comprehensive model evaluation.
    
    Args:
        model_path: Path to saved model
        data_dir: Path to data directory
        optimize_threshold: Whether to find optimal threshold
        generate_plots: Whether to generate visualization plots
    
    Returns:
        Dictionary of evaluation metrics
    """
    
    print("=" * 60)
    print("📊 DEEPFAKE DETECTOR V3 - EVALUATION")
    print("=" * 60)
    
    # Create output directories
    os.makedirs(EvalConfig.REPORTS_DIR, exist_ok=True)
    os.makedirs(EvalConfig.FIGURES_DIR, exist_ok=True)
    
    # Load model
    print(f"\n🔄 Loading model from {model_path}...")
    
    try:
        # Try loading with custom objects
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
        print("   ✅ Model loaded successfully (full model)")
    except Exception as e:
        print(f"   ⚠️ Full load failed: {e}")
        print("   🔄 Attempting to rebuild and load weights...")
        
        try:
            model = build_model_v3()
            model.load_weights(model_path)
            print("   ✅ Model weights loaded successfully")
        except Exception as e2:
            print(f"   ❌ Failed to load model: {e2}")
            return {}
    
    # Load test data
    test_ds = load_test_data(data_dir)
    
    # Get predictions
    y_pred_proba, y_true = get_predictions_and_labels(model, test_ds)
    
    # Find optimal threshold
    if optimize_threshold:
        print("\n🔍 Finding Optimal Threshold...")
        
        thresh_f1, score_f1 = find_optimal_threshold(y_true, y_pred_proba, 'f1')
        thresh_youden, score_youden = find_optimal_threshold(y_true, y_pred_proba, 'youden')
        thresh_balanced, score_balanced = find_optimal_threshold(y_true, y_pred_proba, 'balanced')
        
        print(f"   F1-optimal threshold: {thresh_f1:.3f} (F1={score_f1:.4f})")
        print(f"   Youden-optimal threshold: {thresh_youden:.3f} (J={score_youden:.4f})")
        print(f"   Balanced-optimal threshold: {thresh_balanced:.3f} (BA={score_balanced:.4f})")
        
        # Use F1-optimal threshold
        optimal_threshold = thresh_f1
    else:
        optimal_threshold = 0.5
    
    # Compute metrics
    print("\n📈 Computing Metrics...")
    
    # Metrics at default threshold (0.5)
    metrics_default = compute_metrics(y_true, y_pred_proba, threshold=0.5)
    
    # Metrics at optimal threshold
    metrics_optimal = compute_metrics(y_true, y_pred_proba, threshold=optimal_threshold)
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 FINAL EVALUATION REPORT")
    print("=" * 60)
    
    print("\n📌 At Default Threshold (0.5):")
    print(f"   Accuracy:          {metrics_default['accuracy']:.4f}")
    print(f"   Balanced Accuracy: {metrics_default['balanced_accuracy']:.4f}")
    print(f"   Precision:         {metrics_default['precision']:.4f}")
    print(f"   Recall:            {metrics_default['recall']:.4f}")
    print(f"   Specificity:       {metrics_default['specificity']:.4f}")
    print(f"   F1 Score:          {metrics_default['f1_score']:.4f}")
    print(f"   AUC-ROC:           {metrics_default['auc_roc']:.4f}")
    print(f"   AUC-PR:            {metrics_default['auc_pr']:.4f}")
    
    if optimize_threshold:
        print(f"\n📌 At Optimal Threshold ({optimal_threshold:.3f}):")
        print(f"   Accuracy:          {metrics_optimal['accuracy']:.4f}")
        print(f"   Balanced Accuracy: {metrics_optimal['balanced_accuracy']:.4f}")
        print(f"   Precision:         {metrics_optimal['precision']:.4f}")
        print(f"   Recall:            {metrics_optimal['recall']:.4f}")
        print(f"   Specificity:       {metrics_optimal['specificity']:.4f}")
        print(f"   F1 Score:          {metrics_optimal['f1_score']:.4f}")
    
    # Classification report
    print("\n📋 Classification Report (threshold=0.5):")
    y_pred = (y_pred_proba >= 0.5).astype(int)
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))
    
    # Generate plots
    if generate_plots:
        print("\n🎨 Generating Visualizations...")
        
        plot_confusion_matrix(
            y_true, y_pred, metrics_default,
            f"{EvalConfig.FIGURES_DIR}/confusion_matrix_v3.png"
        )
        
        plot_roc_curve(
            y_true, y_pred_proba,
            f"{EvalConfig.FIGURES_DIR}/roc_curve_v3.png"
        )
        
        plot_precision_recall_curve(
            y_true, y_pred_proba,
            f"{EvalConfig.FIGURES_DIR}/pr_curve_v3.png"
        )
        
        plot_calibration_curve(
            y_true, y_pred_proba,
            f"{EvalConfig.FIGURES_DIR}/calibration_v3.png"
        )
        
        plot_prediction_distribution(
            y_true, y_pred_proba, optimal_threshold,
            f"{EvalConfig.FIGURES_DIR}/prediction_distribution_v3.png"
        )
    
    print("\n" + "=" * 60)
    print("✅ EVALUATION COMPLETE")
    print("=" * 60)
    
    # Return combined metrics
    return {
        'default_threshold': metrics_default,
        'optimal_threshold': metrics_optimal,
        'predictions': y_pred_proba,
        'true_labels': y_true
    }


# Legacy function name for backward compatibility
def evaluate():
    """Backward compatible wrapper."""
    return evaluate_model_v3()


if __name__ == "__main__":
    evaluate_model_v3()